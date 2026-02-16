from distutils.command.config import config
from sklearn.utils import shuffle

import argparse
import logging
import numpy as np
import os
import pandas as pd
import random
import sys
import transformers

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from custom_dataset import CustomDataset_tracked, CustomDataset
from ust import train_model

# logging
logger = logging.getLogger('UST')
logging.basicConfig(level = logging.INFO)

GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
logger.info ("Global seed {}".format(GLOBAL_SEED))

# Full 10-class humanitarian label mapping (superset)
# Not every disaster has all 10 classes — detect_classes() builds the actual mapping
FULL_LABEL_TO_ID = {
	"caution_and_advice":0,
	"displaced_people_and_evacuations":1,
	"infrastructure_and_utility_damage":2,
	"injured_or_dead_people":3,
	"missing_or_found_people":4,
	"not_humanitarian":5,
	"other_relevant_information":6,
	"requests_or_urgent_needs":7,
	"rescue_volunteering_or_donation_effort":8,
	"sympathy_and_support":9,
}


def detect_classes(disaster, train_file, data_dir="data"):
	"""Detect the actual classes present in a disaster dataset.

	Scans train, dev, and test TSV files to collect all unique class labels,
	then builds a contiguous label-to-id mapping (0, 1, 2, ...).
	"""
	base = os.path.join(data_dir, disaster)
	files = [
		os.path.join(base, "labeled_" + train_file + ".tsv"),
		os.path.join(base, disaster + "_dev.tsv"),
		os.path.join(base, disaster + "_test.tsv"),
	]
	all_labels = set()
	for f in files:
		if os.path.exists(f):
			df = pd.read_csv(f, sep='\t')
			all_labels.update(df['class_label'].dropna().unique())

	# Build a contiguous mapping, preserving the canonical order
	label_to_id = {}
	idx = 0
	for label in FULL_LABEL_TO_ID:
		if label in all_labels:
			label_to_id[label] = idx
			idx += 1

	return label_to_id, len(label_to_id)


def get_dataset(path, tokenizer, label_to_id, labeled=True):

    df = pd.read_csv(path, sep='\t')
    text_list = []
    labels_list = []
    ids_list = []
    for i, row in df.iterrows():
        if pd.isna(row['tweet_text']):
            continue
        text_list.append(row['tweet_text'])
        labels_list.append(label_to_id[row['class_label']])
        ids_list.append(row['tweet_id'])

    dataset = CustomDataset_tracked(text_list, labels_list, ids_list, tokenizer, labeled=labeled)
    return dataset



if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--disaster", required=True, help="path of the disaster directory containing train, test and unlabeled data files")
	parser.add_argument("--train_file", nargs="?", type=str, default="S1T_5", help="train file" )
	parser.add_argument("--dev_file", nargs="?", type=str, default="S1V_5", help="train file")
	parser.add_argument("--unlabeled_file", nargs="?", type=str, default="S1U", help="train file")
	parser.add_argument("--aum_save_dir", nargs="?", type=str, default="AUM0", help="Aum save directory")
	parser.add_argument("--seq_len", nargs="?", type=int, default=128, help="sequence length")
	parser.add_argument("--sup_batch_size", nargs="?", type=int, default=16, help="batch size for fine-tuning base model")
	parser.add_argument("--unsup_batch_size", nargs="?", type=int, default=64, help="batch size for self-training on pseudo-labeled data")
	parser.add_argument("--sample_size", nargs="?", type=int, default=1800, help="number of unlabeled samples for evaluating uncetainty on in each self-training iteration")
	parser.add_argument("--unsup_size", nargs="?", type=int, default=1000, help="number of pseudo-labeled instances drawn from sample_size and used in each self-training iteration")
	parser.add_argument("--sample_scheme", nargs="?", default="easy_bald_class_conf", help="Sampling scheme to use")
	# parser.add_argument("--sup_labels", nargs="?", type=int, default=60, help="number of labeled samples per class for training and validation (total)")
	parser.add_argument("--T", nargs="?", type=int, default=7, help="number of masked models for uncertainty estimation")
	parser.add_argument("--alpha", nargs="?", type=float, default=0.1, help="hyper-parameter for confident training loss")
	# parser.add_argument("--valid_split", nargs="?", type=float, default=0.5, help="percentage of sup_labels to use for validation for each class")
	parser.add_argument("--sup_epochs", nargs="?", type=int, default=18, help="number of epochs for fine-tuning base model")
	parser.add_argument("--unsup_epochs", nargs="?", type=int, default=12, help="number of self-training iterations")
	parser.add_argument("--N_base", nargs="?", type=int, default=3, help="number of times to randomly initialize and fine-tune few-shot base encoder to select the best starting configuration")
	parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="vinai/bertweet-base", help="teacher model checkpoint to load pre-trained weights")
	parser.add_argument("--results_file", nargs="?", default="result.txt", help="file name")
	parser.add_argument("--do_pairwise", action="store_true", default=False, help="whether to perform pairwise classification tasks like MNLI")
	parser.add_argument("--hidden_dropout_prob", nargs="?", type=float, default=0.3, help="dropout probability for hidden layer of teacher model")
	parser.add_argument("--attention_probs_dropout_prob", nargs="?", type=float, default=0.3, help="dropout probability for attention layer of teacher model")
	parser.add_argument("--dense_dropout", nargs="?", type=float, default=0.5, help="dropout probability for final layers of teacher model")
	parser.add_argument("--temp_scaling", nargs="?", type=bool, default=False, help="temp scaling" )
	parser.add_argument("--label_smoothing", nargs="?", type=float, default=0.0, help="label smoothing factor")

	args = vars(parser.parse_args())
	logger.info(args)

	disaster_name = args["disaster"]
	max_seq_length = args["seq_len"]
	sup_batch_size = args["sup_batch_size"]
	unsup_batch_size = args["unsup_batch_size"]
	unsup_size = args["unsup_size"]
	sample_size = args["sample_size"]
	model_dir = disaster_name
	aum_save_dir = args["aum_save_dir"]
	sample_scheme = args["sample_scheme"]
	# sup_labels = args["sup_labels"]
	T = args["T"]
	alpha = args["alpha"]
	# valid_split = args["valid_split"]
	sup_epochs = args["sup_epochs"]
	unsup_epochs = args["unsup_epochs"]
	N_base = args["N_base"]
	pt_teacher_checkpoint = args["pt_teacher_checkpoint"]
	do_pairwise = args["do_pairwise"]
	dense_dropout = args["dense_dropout"]
	attention_probs_dropout_prob = args["attention_probs_dropout_prob"]
	hidden_dropout_prob = args["hidden_dropout_prob"]
	results_file_name = args["results_file"]
	# num_aug = args["num_aug"]
	train_file = args["train_file"]
	dev_file = args["dev_file"]
	unlabeled_file = args["unlabeled_file"]
	temp_scaling = args["temp_scaling"]
	label_smoothing = args["label_smoothing"]
	# test_disaster = args["test_disaster"]


	cfg = AutoConfig.from_pretrained(pt_teacher_checkpoint)
	cfg.hidden_dropout_prob = hidden_dropout_prob
	cfg.attention_probs_dropout_prob = attention_probs_dropout_prob

	tokenizer = AutoTokenizer.from_pretrained(pt_teacher_checkpoint)

	# Detect actual classes for this disaster (not all have 10)
	label_to_id, n_classes = detect_classes(disaster_name, train_file)
	logger.info("Detected {} classes for {}: {}".format(n_classes, disaster_name, list(label_to_id.keys())))

	ds_train = get_dataset("data/" + disaster_name + "/labeled_" + train_file + ".tsv", tokenizer, label_to_id)
	ds_dev = get_dataset("data/" + disaster_name + "/" + disaster_name + "_dev.tsv", tokenizer, label_to_id)
	ds_test = get_dataset("data/" + disaster_name + "/" + disaster_name + "_test.tsv", tokenizer, label_to_id)
	ds_unlabeled = get_dataset("data/" + disaster_name + "/unlabeled_" + train_file + ".tsv", tokenizer, label_to_id, False)

	train_model(ds_train, ds_dev, ds_test, ds_unlabeled, pt_teacher_checkpoint, cfg, model_dir, sup_batch_size=sup_batch_size, unsup_batch_size=unsup_batch_size, unsup_size=unsup_size, sample_size=sample_size,
	            sample_scheme=sample_scheme, T=T, alpha=alpha, sup_epochs=sup_epochs, unsup_epochs=unsup_epochs, N_base=N_base, dense_dropout=dense_dropout, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob,
				results_file=results_file_name, temp_scaling=temp_scaling, ls=label_smoothing, n_classes=n_classes)

