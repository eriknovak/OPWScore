import os
import sys
import numpy as np
from scipy import stats
from src.utils.fs import read_json_file, save_scores_to_file
from src.data.Datasets import WMT18

# =====================================
# Import Inputs
# =====================================

REL_PATH = "results/baselines"
datasets = sys.argv[1].split(",") if len(sys.argv) == 2 else None

# =====================================
# Define the Helper Functions
# =====================================


def run_statistical_tests(dataset, filename, model):
    """Evaluates the model's performance"""

    # read the test results
    results = read_json_file(os.path.join(REL_PATH, dataset, "fluency"), filename)

    # get all system and human scores
    diffs = [k[model]["ref"] - np.mean(k[model]["test"]) for k in results]

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
        for model in models:
            evaluation = run_statistical_tests(
                dataset, f"fluency.{lang_pair}.json", model
            )
            filename = f"eval.fluency.{model.replace('-', '_').lower()}.{dataset}.{lang_pair}.json"
            save_scores_to_file(
                os.path.join(REL_PATH, dataset, "fluency", "scores"),
                filename,
                evaluation,
            )


# =====================================
# Run the Evaluation
# =====================================

models = [
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "BERTScore",
    "COMET",
]

if datasets == None or "wmt18" in datasets:
    print("WMT18 dataset: Start evaluation")
    evaluate_dataset("wmt18")
