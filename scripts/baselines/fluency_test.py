import os
import sys
from evaluate import load
from src.data.Datasets import WMT18

from src.utils.fs import save_scores_to_file
from tqdm import tqdm

# =====================================
# Import Inputs
# =====================================

datasets = sys.argv[1].split(",") if len(sys.argv) == 2 else None

# =====================================
# Define the Model Functions
# =====================================

# prepare the models
model_bleu = load("bleu")
model_meteor = load("meteor")
model_rouge = load("rouge")
model_bertscore = load("bertscore")
model_comet = load("comet")


def get_bleu_1_score(predictions, references, sources=None, lang=None):
    def _process_single_example(pred, ref):
        try:
            results = model_bleu.compute(predictions=[pred], references=[[ref]])
        except:
            results = {"precisions": [0, 0, 0, 0]}
        return results["precisions"][0]

    return [
        _process_single_example(pred, ref) for pred, ref in zip(predictions, references)
    ]


def get_bleu_2_score(predictions, references, sources=None, lang=None):
    def _process_single_example(pred, ref):
        try:
            results = model_bleu.compute(predictions=[pred], references=[[ref]])
        except:
            results = {"precisions": [0, 0, 0, 0]}
        return results["precisions"][1]

    return [
        _process_single_example(pred, ref) for pred, ref in zip(predictions, references)
    ]


def get_bleu_3_score(predictions, references, sources=None, lang=None):
    def _process_single_example(pred, ref):
        try:
            results = model_bleu.compute(predictions=[pred], references=[[ref]])
        except:
            results = {"precisions": [0, 0, 0, 0]}
        return results["precisions"][2]

    return [
        _process_single_example(pred, ref) for pred, ref in zip(predictions, references)
    ]


def get_bleu_4_score(predictions, references, sources=None, lang=None):
    def _process_single_example(pred, ref):
        try:
            results = model_bleu.compute(predictions=[pred], references=[[ref]])
        except:
            results = {"precisions": [0, 0, 0, 0]}
        return results["precisions"][3]

    return [
        _process_single_example(pred, ref) for pred, ref in zip(predictions, references)
    ]


def get_meteor_score(predictions, references, sources=None, lang=None):
    def _process_single_example(pred, ref):
        try:
            results = model_meteor.compute(predictions=[pred], references=[ref])
        except:
            results = {"meteor": 0}
        return results["meteor"]

    return [
        _process_single_example(pred, ref) for pred, ref in zip(predictions, references)
    ]


def get_rougel_score(predictions, references, sources=None, lang=None):
    def _process_single_example(pred, ref):
        try:
            results = model_rouge.compute(predictions=[pred], references=[ref])
        except:
            results = {"rougeL": 0}
        return results["rougeL"]

    return [
        _process_single_example(pred, ref) for pred, ref in zip(predictions, references)
    ]


def get_bertscore_score(predictions, references, sources=None, lang="en"):
    results = model_bertscore.compute(
        predictions=predictions, references=references, lang=lang
    )
    return results["f1"]


def get_comet_score(predictions, references, sources, lang=None):
    results = model_comet.compute(
        predictions=predictions, references=references, sources=sources, gpus=1
    )
    return results["scores"]


models = [
    # {"id": "BLEU-1", "model": get_bleu_1_score},
    # {"id": "BLEU-2", "model": get_bleu_2_score},
    # {"id": "BLEU-3", "model": get_bleu_3_score},
    # {"id": "BLEU-4", "model": get_bleu_4_score},
    # {"id": "METEOR", "model": get_meteor_score},
    # {"id": "ROUGE-L", "model": get_rougel_score},
    # {"id": "BERTScore", "model": get_bertscore_score},
    {"id": "COMET", "model": get_comet_score},
]

# =====================================
# Prepare the Dataset Helper Functions
# =====================================


def calculate_fluency(dataset):
    if dataset == "wmt18":
        Dataset = WMT18
    else:
        raise Exception("Unsupported dataset!")

    for lang_pair in tqdm(Dataset.supported_languages, desc="language pairs"):
        # load the datasets
        dataloader = (
            Dataset(lang_pair, batch_size=1).setup(stage="fluency").fluency_dataloader()
        )
        language = lang_pair.split("-")[1]
        # calculate the scores
        fluency = []
        for data in tqdm(dataloader, desc=f"Processing lang pair {lang_pair}"):
            fluency_scores = {}
            for model in models:
                reference_refs = model["model"](
                    predictions=data["o_reference"],
                    references=data["o_reference"],
                    sources=data["o_source"],
                    lang=language,
                )
                reference_tests = [
                    model["model"](
                        predictions=[
                            list(sentence)[0] for sentence in data["j_reference"]
                        ],
                        references=[
                            data["o_reference"][0]
                            for _ in range(len(data["j_reference"]))
                        ],
                        sources=[
                            data["o_source"][0] for _ in range(len(data["j_reference"]))
                        ],
                        lang=language,
                    )
                ]
                fluency_scores[model["id"]] = {
                    "test": reference_tests,
                    "ref": reference_refs,
                }
            fluency.append(fluency_scores)
        # save the metric scores
        rel_path = os.path.join("results", "baselines", dataset, "fluency")
        file_path = f"fluency.{lang_pair}.json"
        save_scores_to_file(rel_path, file_path, fluency)


# =====================================
# Calculate the Scores on the Datasets
# =====================================

if datasets == None or "wmt18" in datasets:
    print("WMT18 dataset: Start evaluation")
    calculate_fluency("wmt18")
