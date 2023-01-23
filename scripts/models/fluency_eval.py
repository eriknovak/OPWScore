import os
import sys
import numpy as np
from scipy import stats
from src.data.Datasets import WMT18
from src.utils.fs import read_json_file, save_scores_to_file

# =====================================
# Import Inputs
# =====================================

REL_PATH = "results"
datasets = sys.argv[1].split(",") if len(sys.argv) == 2 else None

# =====================================
# Define the Helper Functions
# =====================================


def run_statistical_tests(dataset, filename):
    """Evaluates the model's performance"""

    # read the test results
    results = read_json_file(os.path.join(REL_PATH, dataset), filename)

    # get all system and human scores
    diffs = [np.mean(k["test"]) - k["ref"] for k in results]

    results = stats.ttest_1samp(diffs, popmean=0)

    # calculate the correlations
    mean = np.mean(diffs)
    std = np.std(diffs)
    pvalue = results[1]

    return {
        "mean": mean,
        "std": std,
        "pvalue": pvalue,
    }


def evaluate_dataset(dataset):
    if dataset == "wmt18":
        Dataset = WMT18
    else:
        raise Exception("Unsupported dataset!")

    for lang_pair in Dataset.supported_languages:
        fluency_scores_file = f"fluency.{lang_pair}.json"
        if not os.path.exists(
            os.path.join(REL_PATH, dataset, "fluency", fluency_scores_file)
        ):
            continue
        evaluation = run_statistical_tests(dataset, fluency_scores_file)
        filename = f"eval.{dataset}.{lang_pair}.fluency.json"
        save_scores_to_file(
            REL_PATH, dataset, "fluency", "scores", filename, evaluation
        )


# =====================================
# Run the Evaluation
# =====================================

if datasets == None or "wmt18" in datasets:
    print("WMT18 dataset: Start evaluation")
    evaluate_dataset("wmt18")
