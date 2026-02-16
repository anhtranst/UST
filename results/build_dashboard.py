"""
Build the results dashboard by reading all experiment JSON files and
injecting the data into index.html as an embedded JavaScript object.

Run from anywhere:
    python results/build_dashboard.py
"""

import json
import os
import re

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE    = os.path.join(RESULTS_DIR, "index.html")
OUTPUT      = os.path.join(RESULTS_DIR, "dashboard.html")

DISASTERS = [
    "california_wildfires_2018", "canada_wildfires_2016", "cyclone_idai_2019",
    "hurricane_dorian_2019", "hurricane_florence_2018", "hurricane_harvey_2017",
    "hurricane_irma_2017", "hurricane_maria_2017", "kaikoura_earthquake_2016",
    "kerala_floods_2018",
]
SIZES = [5, 10, 25, 50]
SETS  = [1, 2, 3]

FULL_LABEL_TO_ID = {
    "caution_and_advice": 0, "displaced_people_and_evacuations": 1,
    "infrastructure_and_utility_damage": 2, "injured_or_dead_people": 3,
    "missing_or_found_people": 4, "not_humanitarian": 5,
    "other_relevant_information": 6, "requests_or_urgent_needs": 7,
    "rescue_volunteering_or_donation_effort": 8, "sympathy_and_support": 9,
}


def detect_classes(disaster, train_file):
    """Return number of classes present in this disaster's data files."""
    try:
        import pandas as pd
        data_dir = os.path.join(RESULTS_DIR, "..", "data", disaster)
        files = [
            os.path.join(data_dir, f"labeled_{train_file}.tsv"),
            os.path.join(data_dir, f"{disaster}_dev.tsv"),
            os.path.join(data_dir, f"{disaster}_test.tsv"),
        ]
        all_labels = set()
        for f in files:
            if os.path.exists(f):
                df = pd.read_csv(f, sep="\t")
                all_labels.update(df["class_label"].dropna().unique())
        return sum(1 for l in FULL_LABEL_TO_ID if l in all_labels)
    except Exception:
        return None


def parse_ece(ece_str):
    """Parse ECE value from string like 'tensor(0.4798, device=\\'cuda:0\\')' or '0.4798'."""
    if ece_str is None:
        return None
    m = re.search(r"[\d]+\.[\d]+", str(ece_str))
    return float(m.group()) if m else None


def load_results():
    data = {}
    for disaster in DISASTERS:
        disaster_dir = os.path.join(RESULTS_DIR, disaster)
        if not os.path.isdir(disaster_dir):
            continue
        for size in SIZES:
            for s in SETS:
                train_file = f"{size}_set{s}"
                result_path = os.path.join(disaster_dir, f"st_uniform_{train_file}.txt")
                key = f"{disaster}__{train_file}"
                if not os.path.exists(result_path):
                    continue
                try:
                    with open(result_path) as f:
                        r = json.load(f)
                    best = r.get("Best ST model", {})
                    f1  = best.get("F1 before temp scaling")
                    ece = best.get("ECE before temp scaling")
                    n_classes = detect_classes(disaster, train_file)
                    data[key] = {
                        "f1":  float(f1)  if f1  is not None else None,
                        "ece": parse_ece(ece),
                        "n_classes": n_classes,
                        "temp_scaling": r.get("Temperature Scaling", False),
                        "label_smoothing": r.get("Label Smoothing", 0.0),
                    }
                    # include post-scaling metrics if available
                    if "F1 after temp scaling" in best:
                        data[key]["f1_scaled"]  = float(best["F1 after temp scaling"])
                        data[key]["ece_scaled"]  = parse_ece(best.get("ECE after temp scaling"))
                except Exception as e:
                    print(f"  Warning: could not parse {result_path}: {e}")
    return data


def build():
    results = load_results()
    print(f"Loaded {len(results)} result files.")

    with open(TEMPLATE) as f:
        html = f.read()

    json_str = json.dumps(results, indent=2)
    html = html.replace("__RESULTS_JSON__", json_str)

    with open(OUTPUT, "w") as f:
        f.write(html)

    print(f"Dashboard written to: {OUTPUT}")


if __name__ == "__main__":
    build()
