import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# python types
from typing import Optional


DIRNAME = os.path.dirname(__file__)

# =====================================
# Helper Functions
# =====================================


def construct_source_path(lang_pair):
    source_lang = lang_pair.split("-")[0]
    lang_pair_id = lang_pair.replace("-", "")
    rel_path = (
        f"../../data/raw/wmt18/sources/newstest2018-{lang_pair_id}-src.{source_lang}"
    )
    source_path = os.path.join(DIRNAME, rel_path)
    return source_path


def construct_reference_path(lang_pair):
    source_lang = lang_pair.split("-")[1]
    lang_pair_id = lang_pair.replace("-", "")
    rel_path = (
        f"../../data/raw/wmt18/references/newstest2018-{lang_pair_id}-ref.{source_lang}"
    )
    reference_path = os.path.join(DIRNAME, rel_path)
    return reference_path


def get_model_name(file):
    return ".".join(file.split(".")[1:-1])


def get_system_paths(lang_pair):
    [source_lang, target_lang] = lang_pair.split("-")
    rel_path = (
        f"../../data/raw/wmt18/system-outputs/newstest2018/{source_lang}-{target_lang}"
    )
    system_folder_path = os.path.join(DIRNAME, rel_path)

    files = []
    for _, _, files in os.walk(system_folder_path):
        files = [
            {
                "path": os.path.join(DIRNAME, rel_path, file),
                "model": get_model_name(file),
            }
            for file in files
            if "hybrid" not in file
        ]
    # return all files in the system path
    return files


def get_source_sentences(lang_pair):
    with open(construct_source_path(lang_pair), "r", encoding="utf8") as f:
        return [line.strip() for line in f.readlines()]


def get_reference_sentences(lang_pair):
    with open(construct_reference_path(lang_pair), "r", encoding="utf8") as f:
        return [line.strip() for line in f.readlines()]


def get_system_sentences(lang_pair):
    system_file_names = get_system_paths(lang_pair)

    system_values = {}
    for file in system_file_names:
        with open(file["path"], "r", encoding="utf8") as f:
            system_values[file["model"]] = [line.strip() for line in f.readlines()]

    return system_values


def get_humaneval_values(lang_pair):
    human_eval = {}

    # rel_path = f"../../data/raw/wmt18/humaneval/ad-seg-scores-{lang_pair}.csv"
    # file_path = os.path.join(DIRNAME, rel_path)
    # with open(file_path, "r", encoding="utf8") as f:
    #     for line in f.readlines()[1:]:
    #         if line != "":
    #             [SYSTEM, SID, RAW, Z, N] = line.strip().split()
    #             if SYSTEM not in human_eval:
    #                 human_eval[SYSTEM] = {}
    #             human_eval[SYSTEM][int(SID)] = float(RAW)

    rel_path = f"../../data/raw/wmt18/humaneval/ad-good-raw.csv"
    file_path = os.path.join(DIRNAME, rel_path)
    with open(file_path, "r", encoding="utf8") as f:
        for line in f.readlines()[1:]:
            if line != "":
                [
                    hit_id,
                    Worker_id,
                    input_src,
                    input_trg,
                    input_item,
                    hit,
                    sys_id,
                    rid,
                    type,
                    sid,
                    score,
                    time,
                ] = line.strip().split()

                if f"{input_src}-{input_trg}" != lang_pair or type != "SYSTEM":
                    continue

                if sys_id not in human_eval:
                    human_eval[sys_id] = {}
                human_eval[sys_id][int(sid)] = float(score)

    return human_eval


# =====================================
# Define WMT18 Dataset
# =====================================


class WMT18Dataset(Dataset):
    def __init__(self, language: str = None):
        self.language = language

        # get the data values
        sources = get_source_sentences(self.language)
        references = get_reference_sentences(self.language)
        system = get_system_sentences(self.language)
        human_eval = get_humaneval_values(self.language)

        self.data = [
            {
                "model_id": key,
                "source": sources[idx],
                "reference": references[idx],
                "system": system,
                "score": human_eval[key][idx + 1],
            }
            for key, value in system.items()
            for idx, system in enumerate(value)
            if idx + 1 in human_eval[key]
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# =====================================
# Define WMT18 DataLoader
# =====================================


class WMT18(pl.LightningDataModule):
    def __init__(self, language: str = None, batch_size: int = 32):
        # Please pick one among the available configs:
        #   ['cs-en', 'de-en', 'et-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en', 'en-cs', 'en-de', 'en-et', 'en-fi', 'en-ru', 'en-tr', 'en-zh']
        super().__init__()

        if not language:
            raise Exception(
                """Language not specified! Please pick one among the available configs:
                ['cs-en', 'de-en', 'et-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en', 'en-cs', 'en-de', 'en-et', 'en-fi', 'en-ru', 'en-tr', 'en-zh']
                """
            )

        self.language = language
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = None
            self.data_val = None
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = WMT18Dataset(self.language)
        if stage == "predict" or stage is None:
            self.data_predict = None

        return self

    def train_dataloader(self):
        return DataLoader(self.data_train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, self.batch_size)
