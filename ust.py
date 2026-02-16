"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
"""

from collections import defaultdict
from copy import deepcopy
from scipy.special import softmax
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import time 
from torchmetrics.classification import MulticlassCalibrationError
import torch.nn.functional as F
import torch.nn as nn
from custom_dataset import CustomDataset, CustomDataset_tracked

import logging
import math
import numpy as np
import pandas as pd 
import os
import sampler
import torch
import json 
import statistics 
import random
from multiprocessing import Process, Pool
from torch.multiprocessing import Pool, Process, set_start_method

logger = logging.getLogger('UST')

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("The device is : ", device)

def multigpu(model):
    model = nn.DataParallel(model).to(device)
    return model 

class BertModel(torch.nn.Module):
    def __init__(self, checkpoint, num_labels=10):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
        #self.model.classifier.dropout.p = 0.5
        self.T = torch.nn.Parameter(torch.ones(1) * 1.0)


    def forward(self, input_ids, token_type_ids, attention_mask, temperature_scaling=False):
        if temperature_scaling:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            temperature = self.T.unsqueeze(1).expand(outputs.logits.size(0), outputs.logits.size(1))
            outputs.logits /= temperature
        else:
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return outputs


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler


def mc_dropout_evaluate(model_dir, n_classes, pt_teacher_checkpoint, X_new_unlabeled_dataset, cfg, linear_dropout=0.5, T=30, data_dir="data"):

    cfg.return_dict = True

    model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
    state_dict = torch.load(os.path.join(data_dir, model_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()

    #print(len(X_new_unlabeled_dataset.text_list))

    y_T = np.zeros((T, len(X_new_unlabeled_dataset), n_classes))
    acc = None
    data_loader = torch.utils.data.DataLoader(
        X_new_unlabeled_dataset, batch_size=64, shuffle=False)   

    logger.info ("Yielding predictions looping over ...")
    with torch.no_grad():
        for i in tqdm(range(T)):
            y_pred = []
            for elem in data_loader:
                x = {key: elem[key].to(device)
                for key in elem if key not in ['idx']}
                pred = model(
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
                y_pred.extend(pred.logits.cpu().numpy())

            #converting logits to probabilities
            y_T[i] = softmax(np.array(y_pred), axis=-1)
    logger.info (y_T)

    #compute mean
    y_mean = np.mean(y_T, axis=0)
    assert y_mean.shape == (len(X_new_unlabeled_dataset), n_classes)

    #compute majority prediction
    y_pred = np.array([np.argmax(np.bincount(row)) for row in np.transpose(np.argmax(y_T, axis=-1))])
    assert y_pred.shape == (len(X_new_unlabeled_dataset),)

    #compute variance
    y_var = np.var(y_T, axis=0)
    assert y_var.shape == (len(X_new_unlabeled_dataset), 10)

    return y_mean, y_var, y_pred, y_T



def evaluate(model, n_classes, test_dataloader, criterion, batch_size, temp_scaling=False):
    full_predictions = []
    true_labels = []
    probabilities = []

    model.eval()
    crt_loss = 0

    with torch.no_grad():
        for elem in tqdm(test_dataloader):
            x = {key: elem[key].to(device)
                for key in elem if key not in ['idx']}
            logits = model(
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'], temperature_scaling=temp_scaling)
            results = torch.argmax(logits.logits, dim=1)
            prob = F.softmax(logits.logits.to('cpu'), dim=1)
            probabilities += list(prob)

            crt_loss += criterion(logits.logits, x['lbl']
                                ).cpu().detach().numpy()
            full_predictions = full_predictions + \
                list(results.cpu().detach().numpy())
            true_labels = true_labels + list(elem['lbl'].cpu().detach().numpy())


    model.train()

    metric = MulticlassCalibrationError(num_classes=n_classes, n_bins=10, norm='l1')
    metric = metric.to(device)

    preds = torch.stack(probabilities)
    preds = preds.to(device)

    orig = torch.tensor(true_labels, dtype=torch.float, device=device)

    ece_metric = metric(preds, orig).to(device)

    return f1_score(true_labels, full_predictions, average='macro'), crt_loss / len(test_dataloader), ece_metric



def	train_model(ds_train, ds_dev, ds_test, ds_unlabeled, pt_teacher_checkpoint, cfg, model_dir, sup_batch_size=16, unsup_batch_size=64, unsup_size=4096, sample_size=16384,
	            sample_scheme='easy_bald_class_conf', T=30, alpha=0.1, sup_epochs=20, unsup_epochs=25, N_base=10, dense_dropout=0.5, attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3,
                results_file="", results_dir="results", data_dir="data", temp_scaling=False, ls=0.0, n_classes=10):

    start_time = time.time()
    patience = 5
    load_best = False
    logger_dict = {}
    logger_dict["Temperature Scaling"] = temp_scaling
    logger_dict["Label Smoothing"]= ls

    train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_size=sup_batch_size, shuffle=True)   
    validation_dataloader = torch.utils.data.DataLoader(
        ds_dev, batch_size=sup_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        ds_test, batch_size=128, shuffle=False)
    
    cfg.num_labels = n_classes
    copy_cfg = deepcopy(cfg)
    copy_cfg.attention_probs_dropout_prob = 0.1
    copy_cfg.hidden_dropout_prob = 0.1
    #run the base model n times with different initialization to select best base model based on validation loss
    best_f1_overall = 0
    crt_patience = 0
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=ls)

    if load_best == False:
        for counter in range(N_base):
            best_f1 = 0
            copy_cfg.return_dict  = True
            model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
            model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
            if counter == 0:
                logger.info(model)
            for epoch in range(sup_epochs):
                for data in tqdm(train_dataloader):
                    cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                    optimizer.zero_grad()
                    logits = model(input_ids=cuda_tensors['input_ids'], token_type_ids=cuda_tensors['token_type_ids'], attention_mask=cuda_tensors['attention_mask'])
                    #print(logits)
                    loss = loss_fn(logits.logits, cuda_tensors['lbl'])
                    loss.backward()
                    optimizer.step()

                f1_macro_validation, loss_validation, ece = evaluate(
                    model, n_classes, validation_dataloader, loss_fn, unsup_batch_size)

                if f1_macro_validation >= best_f1:
                    crt_patience = 0
                    best_f1 = f1_macro_validation
                    if best_f1 > best_f1_overall:
                        #model.save_pretrained(model_dir+"/ust")
                        torch.save(model.state_dict(), os.path.join(data_dir, model_dir, "pytorch_model.bin"))
                        best_f1_overall = best_f1
                    print('New best macro validation', best_f1, 'Epoch', epoch)
                    continue
            
                if crt_patience == 3:
                    crt_patience = 0
                    print('Exceeding max patience; Exiting..')
                    break

                crt_patience += 1

        del model

    cfg.return_dict = True

    best_model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
    best_model.to(device)
    state_dict = torch.load(os.path.join(data_dir, model_dir, "pytorch_model.bin"))
    best_model.load_state_dict(state_dict)

    for epoch in range(unsup_epochs):


        if sample_size < len(ds_unlabeled):
            logger.info ("Evaluating uncertainty on {} number of instances sampled from {} unlabeled instances".format(sample_size, len(ds_unlabeled)))
            indices = np.random.choice(len(ds_unlabeled), min(sample_size, len(ds_unlabeled)), replace=False)
            X_new_unlabeled_dataset = ds_unlabeled.get_subset_dataset(indices)
        else:
            logger.info ("Evaluating uncertainty on {} number of instances".format(len(ds_unlabeled)))
            X_new_unlabeled_dataset = ds_unlabeled

        if 'uni' in sample_scheme:
            y_mean, y_var, y_T = None, None, None
        elif 'bald' in sample_scheme:
            y_mean, y_var, y_pred, y_T = mc_dropout_evaluate(model_dir, n_classes, pt_teacher_checkpoint, X_new_unlabeled_dataset, cfg, dense_dropout, T=T, data_dir=data_dir)
        else:
            logger.info ("Error in specifying sample_scheme: One of the 'uni' or 'bald' schemes need to be specified")
            exit(0)

        if 'soft' not in sample_scheme:
            copy_cfg.return_dict = True
            # model = AutoModelForSequenceClassification.from_pretrained(model_dir+"/ust", config=copy_cfg)
            model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
            state_dict = torch.load(os.path.join(data_dir, model_dir, "pytorch_model.bin"))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            data_loader = torch.utils.data.DataLoader(
                X_new_unlabeled_dataset, batch_size=128, shuffle=False)   
            y_pred = []
            with torch.no_grad():
                for elem in data_loader:
                    x = {key: elem[key].to(device)
                    for key in elem if key not in ['idx', 'weights']}
                    pred = model(
                    input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
                    y_pred.extend(pred.logits.cpu().numpy())
            del model

            y_pred = np.array(y_pred)
            y_pred = np.argmax(y_pred, axis=-1).flatten()

        # sample from unlabeled set
        if 'conf' in sample_scheme:
            conf = True
        else:
            conf = False

        if 'bald' in sample_scheme and 'eas' in sample_scheme:
            f_ = sampler.sample_by_bald_easiness

        if 'bald' in sample_scheme and 'eas' in sample_scheme and 'clas' in sample_scheme:
            f_ = sampler.sample_by_bald_class_easiness

        if 'bald' in sample_scheme and 'dif' in sample_scheme:
            f_ = sampler.sample_by_bald_difficulty

        if 'bald' in sample_scheme and 'dif' in sample_scheme and 'clas' in sample_scheme:
            f_ = sampler.sample_by_bald_class_difficulty

        if 'uni' in sample_scheme:
            if unsup_size < len(X_new_unlabeled_dataset):
                indices = np.random.choice(len(X_new_unlabeled_dataset), unsup_size, replace=False)
                text_data = [X_new_unlabeled_dataset.text_list[i] for i in indices]
                X_new_unlabeled_dataset = CustomDataset(text_data, y_pred[indices], X_new_unlabeled_dataset.tokenizer, labeled=True)
            else:
                X_new_unlabeled_dataset = CustomDataset(X_new_unlabeled_dataset.text_list, y_pred, X_new_unlabeled_dataset.tokenizer, labeled=True)
        else:
            X_new_unlabeled_dataset = f_(X_new_unlabeled_dataset, y_mean, y_var, y_pred, unsup_size, 10, y_T=y_T)

        if not conf:
            logger.info ("Not using confidence learning.")
            X_new_unlabeled_dataset.weights = np.ones(len(X_new_unlabeled_dataset))
        else:
            logger.info ("Using confidence learning ")
            X_new_unlabeled_dataset.weights = -np.log(np.array(X_new_unlabeled_dataset.weights)+1e-10)*alpha

        copy_cfg.return_dict = True
        # model = AutoModelForSequenceClassification.from_pretrained(model_dir+"/ust", config=copy_cfg)
        # model.classifier.dropout.p = 0.1
        # model.to(device)

        model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
        model.to(device)
        state_dict = torch.load(os.path.join(data_dir, model_dir, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
        model.train()

        unsup_dataloader = torch.utils.data.DataLoader(
        X_new_unlabeled_dataset, batch_size=unsup_batch_size, shuffle=True)   
        loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=ls)
        loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=ls)

        data_sampler = torch.utils.data.RandomSampler(ds_train, num_samples=10**5, replacement=True)
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, sup_batch_size, drop_last=False)
        train_dataloader = torch.utils.data.DataLoader(ds_train, batch_sampler=batch_sampler)
        crt_patience = 0
        for epoch in range(sup_epochs):
            for data_supervised, data_unsupervised in tqdm(zip(train_dataloader, unsup_dataloader)):
                cuda_tensors_supervised = {key: data_supervised[key].to(
                    device) for key in data_supervised if key not in ['idx', 'weights']}

                cuda_tensors_unsupervised = {key: data_unsupervised[key].to(
                    device) for key in data_unsupervised if key not in ['idx']}
                    
                merged_tensors = {}
                for k in cuda_tensors_supervised:
                    merged_tensors[k] = torch.cat((cuda_tensors_supervised[k], cuda_tensors_unsupervised[k]))

                num_lb = cuda_tensors_supervised['input_ids'].shape[0]
                num_ulb = cuda_tensors_unsupervised['input_ids'].shape[0]

                optimizer.zero_grad()
                logits = model(input_ids=merged_tensors['input_ids'], token_type_ids=merged_tensors['token_type_ids'], attention_mask=merged_tensors['attention_mask'])

                logits_lbls = logits.logits[:num_lb]
                logits_ulbl = logits.logits[num_lb:]

                loss_sup = loss_fn_supervised(logits_lbls, cuda_tensors_supervised['lbl'])
                loss_unsup = cuda_tensors_unsupervised['weights'] * loss_fn_unsupervised(logits_ulbl, cuda_tensors_unsupervised['lbl'])
                loss = 0.5 * loss_sup + 0.5 * torch.mean(loss_unsup)
                loss.backward()
                optimizer.step()

            f1_macro_validation, loss_validation, ece = evaluate(
                model, n_classes, validation_dataloader, loss_fn, unsup_batch_size)
            print('Confident learning metrics', f1_macro_validation)

            if f1_macro_validation >= best_f1:
                crt_patience = 0
                best_f1 = f1_macro_validation
                if best_f1 > best_f1_overall:
                    #model.save_pretrained(model_dir+"/ust")
                    torch.save(model.state_dict(), os.path.join(data_dir, model_dir, "pytorch_model.bin"))
                    best_f1_overall = best_f1
                print('New best macro validation', best_f1, 'Epoch', epoch)
                continue
        
            if crt_patience == 3:
                crt_patience = 0
                print('Exceeding max patience; Exiting..')
                break

            crt_patience += 1

    copy_cfg.return_dict = True
    # model = AutoModelForSequenceClassification.from_pretrained(model_dir+"/ust", config=copy_cfg)
    model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
    model.to(device)
    state_dict = torch.load(os.path.join(data_dir, model_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict)

    rel_file = model_dir + "-" + results_file

    f1_macro_test, loss_test, ece_metric = evaluate(model, n_classes, test_dataloader, loss_fn, unsup_batch_size)

    logger.info ("Test macro F1 based on best validation f1 : {}".format(f1_macro_test))

    logger_dict["Best ST model"] = {}
    logger_dict["Best ST model"]["F1 before temp scaling"] = str(f1_macro_test)
    logger_dict["Best ST model"]["ECE before temp scaling"] = str(ece_metric)


    logger_dict["Best ST model"]["T before temp scaling"] = str(model.T.detach().cpu().numpy()[0])

    if temp_scaling:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-02)

        for epoch in range(20):
            for data in tqdm(validation_dataloader):
                cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                optimizer.zero_grad()
  
                result = model(cuda_tensors['input_ids'], cuda_tensors['token_type_ids'], cuda_tensors['attention_mask'], True)

                loss = loss_fn(result.logits, cuda_tensors['lbl'])
                loss.backward()
                optimizer.step()

        # f1_macro_test, _, ece_metric = evaluate(model, test_dataloader, loss_fn, unsup_batch_size, True)

        f1_macro_test, loss_test, ece_metric = evaluate(model, n_classes, test_dataloader, loss_fn, unsup_batch_size, True)

        logger_dict["Best ST model"]["F1 after temp scaling"] = str(f1_macro_test)
        logger_dict["Best ST model"]["ECE after temp scaling"] = str(ece_metric)

        logger_dict["Best ST model"]["T  after temp scaling"] = str(model.T.detach().cpu().numpy()[0])

    print(json.dumps(logger_dict, indent=4))
    results_out_dir = os.path.join(results_dir, model_dir)
    os.makedirs(results_out_dir, exist_ok=True)
    with open(os.path.join(results_out_dir, results_file + '.txt'), 'w') as fp:
        fp.write(json.dumps(logger_dict, indent=4))





