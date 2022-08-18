import os
import torch

from src.data.STS import STS
from src.data.WMT18 import WMT18

from src.utils.fs import save_scores_to_file

from tqdm import tqdm


def calculate_scores_sts(model):
    # load the datasets
    dataset = STS(batch_size=1)
    dataloader = dataset.setup().test_dataloader()

    # calculate the scores
    scores = []
    for data in tqdm(dataloader, desc="datasets"):
        with torch.no_grad():
            distances, _, _ = model(data["sentence1"], data["sentence2"])
        scores.append(
            {
                "system_score": distances[0].item(),
                "human_score": data["score"][0].item(),
            }
        )
    # save the metric scores
    rel_path = "results/sts"
    file_path = "scores.json"
    save_scores_to_file(rel_path, file_path, scores)


def calculate_scores_wmt18(model):
    # prepare the targetted language pairs
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

    for lang_pair in tqdm(language_pairs, desc="language pairs"):
        # load the datasets
        dataset = WMT18(lang_pair, batch_size=1)
        dataloader = dataset.setup().test_dataloader()
        # calculate the scores
        scores = []
        for data in tqdm(dataloader, desc="datasets"):
            model_ids = data["model_id"]
            with torch.no_grad():
                distances, _, _ = model(data["system"], data["reference"])
            scores.append(
                {
                    "model_id": model_ids[0],
                    "system_score": distances[0].item(),
                    "human_score": data["score"][0].item(),
                }
            )
        # save the metric scores
        rel_path = "results/wmt18"
        file_path = f"scores.{lang_pair}.json"
        save_scores_to_file(rel_path, file_path, scores)
