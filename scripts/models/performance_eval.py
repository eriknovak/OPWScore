import os
import sys
from src.utils.fs import read_json_file, save_scores_to_file
from src.utils.metrics import get_pearson_r, get_spearman_r, get_kendall_tau
from src.data.Datasets import WMT18, WMT20

# =====================================
# Import Inputs
# =====================================

REL_PATH = "results"
datasets = sys.argv[1].split(",") if len(sys.argv) == 2 else None

# =====================================
# Define the Helper Functions
# =====================================


def evaluate(dataset, lang_pair):
    """Evaluates the model's performance"""

    # read the test results
    results = read_json_file(
        os.path.join(REL_PATH, dataset, "correlations"), f"scores.{lang_pair}.json"
    )
    # get all system and human scores
    x = [k["system_score"] for k in results]
    y = [k["score"] for k in results]

    # calculate the correlations
    pearson_r = get_pearson_r(x, y)[0]
    spearman_r = get_spearman_r(x, y).correlation
    kendall_tau = get_kendall_tau(x, y).correlation

    return {
        "pearson_r": abs(pearson_r),
        "spearman_r": abs(spearman_r),
        "kendall_tau": abs(kendall_tau),
    }


def evaluate_model(dataset):
    if dataset == "wmt18":
        Dataset = WMT18
    elif dataset == "wmt20":
        Dataset = WMT20
    else:
        raise Exception("Unsupported dataset!")

    for lang_pair in Dataset.supported_languages:
        try:
            evaluation = evaluate(dataset, lang_pair)
            save_scores_to_file(
                os.path.join(REL_PATH, dataset, "correlations", "scores"),
                f"eval.{dataset}.{lang_pair}.json",
                evaluation,
            )
        except:
            print("Unable to process for lang pair: ", lang_pair)


# =====================================
# Run the Evaluation
# =====================================
if datasets == None or "wmt18" in datasets:
    print("WMT18 dataset: Start evaluation")
    evaluate_model("wmt18")


if datasets == None or "wmt20" in datasets:
    print("WMT20 dataset: Start evaluation")
    evaluate_model("wmt20")
