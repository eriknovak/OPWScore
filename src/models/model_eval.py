import os
import sys
import json
from src.utils.fs import read_json_file, save_scores_to_file
from src.utils.metrics import get_pearson_r, get_spearman_r, get_kendall_tau

# =====================================
# Import Inputs
# =====================================

REL_PATH = "results"
datasets = sys.argv[1].split(",") if len(sys.argv) == 2 else None

# =====================================
# Define the Helper Functions
# =====================================


def evaluate_model(dataset, filename):
    """Evaluates the model's performance"""

    # read the test results
    results = read_json_file(os.path.join(REL_PATH, dataset), filename)

    # get all system and human scores
    x = [k["system_score"] for k in results]
    y = [k["human_score"] for k in results]

    # calculate the correlations
    pearson_r = get_pearson_r(x, y)[0]
    spearman_r = get_spearman_r(x, y).correlation
    kendall_tau = get_kendall_tau(x, y).correlation

    return {
        "pearson_r": abs(pearson_r),
        "spearman_r": abs(spearman_r),
        "kendall_tau": abs(kendall_tau),
    }


# =====================================
# Run the Evaluation
# =====================================

if datasets == None or "sts" in datasets:
    # TODO: implement the evaluation
    print("STS Benchmarks dataset: Start evaluation")
    evaluation = evaluate_model("sts", "scores.json")
    save_scores_to_file(REL_PATH, "eval.sts.json", evaluation)


if datasets == None or "wmt18" in datasets:
    print("WMT18 dataset: Start evaluation")
    language_pairs = [
        "cs-en",
        "de-en",
        "et-en",
        "fi-en",
        "ru-en",
        "tr-en",
        "zh-en",
        "en-cs",
        "en-de",
        "en-et",
        "en-fi",
        "en-ru",
        "en-tr",
        "en-zh",
    ]
    for lang_pair in language_pairs:
        evaluation = evaluate_model("wmt18", f"scores.{lang_pair}.json")
        save_scores_to_file(REL_PATH, f"eval.wmt18.{lang_pair}.json", evaluation)
