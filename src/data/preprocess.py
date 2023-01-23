import os
import torch
from src.data.Datasets import WMT17, WMT18, WMT20
from src.utils.fs import save_scores_to_file
from tqdm import tqdm


def calculate_scores(model, dataset):
    if dataset == "wmt17":
        Dataset = WMT17
    elif dataset == "wmt18":
        Dataset = WMT18
    elif dataset == "wmt20":
        Dataset = WMT20
    else:
        raise Exception("Unsupported dataset!")

    # prepare the targetted language pairs
    language_pairs = list(
        filter(
            lambda x: x.split("-")[1] == model.hparams.lang, Dataset.supported_languages
        )
    )

    for lang_pair in tqdm(language_pairs, desc="language pairs"):
        # load the datasets
        dataloader = Dataset(lang_pair, batch_size=1).setup().test_dataloader()
        # calculate the scores
        scores = []
        count = 0
        for data in tqdm(dataloader, desc="datasets"):
            with torch.no_grad():
                distances, _, _ = model(data["system"], data["reference"])
            for dist, score, raw_score in zip(
                distances, data["score"], data["raw_score"]
            ):
                scores.append(
                    {
                        "system_score": dist.item(),
                        "score": score.item(),
                        "raw_score": raw_score.item(),
                    }
                )
            count += 1
            if count > 5:
                break
        # save the metric scores
        rel_path = os.path.join("results", dataset, "correlations")
        file_path = f"scores.{lang_pair}.json"
        save_scores_to_file(rel_path, file_path, scores)


def calculate_fluency(model, dataset):
    if dataset == "wmt18":
        Dataset = WMT18
    else:
        raise Exception("Unsupported dataset!")

    # prepare the targetted language pairs
    language_pairs = list(
        filter(
            lambda x: x.split("-")[1] == model.hparams.lang,
            Dataset.supported_languages,
        )
    )

    for lang_pair in tqdm(language_pairs, desc="language pairs"):
        # load the datasets
        dataloader = (
            Dataset(lang_pair, batch_size=1).setup(stage="fluency").fluency_dataloader()
        )
        # calculate the scores
        fluency = []
        for data in tqdm(dataloader, desc="datasets"):
            with torch.no_grad():
                refs_dist, _, _ = model(data["o_reference"], data["o_reference"])
                test_dists, _, _ = model(data["o_reference"], data["j_reference"])
                test_dist = [td.item() for td in test_dists]
            fluency.append(
                {
                    "ref": refs_dist[0].item(),
                    "test": test_dist,
                }
            )
        # save the metric scores
        rel_path = os.path.join("results", dataset, "fluency")
        file_path = f"fluency.{lang_pair}.json"
        save_scores_to_file(rel_path, file_path, fluency)
