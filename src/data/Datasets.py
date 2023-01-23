import os
import pandas
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from src.utils.fs import read_json_file

# python types
from typing import Optional


DIRNAME = os.path.dirname(__file__)

# =====================================
# Helper Functions
# =====================================


def get_dataset_fluency(lang_pair):
    return read_json_file("data/processed/wmt18", f"permutations.{lang_pair}.json")


def get_dataset(dataset, language):

    if dataset == "wmt17":
        data_path = f"data/raw/wmt17/2017-da.csv"
    elif dataset == "wmt18":
        data_path = f"data/raw/wmt18/2018-da.csv"
    elif dataset == "wmt20":
        data_path = f"data/raw/wmt20/2020-da.csv"
    else:
        raise Exception(f"Unsupported dataset: {dataset}")

    df = pandas.read_csv(data_path)
    df = df[df["lp"] == language]
    return [
        {
            "source": record["src"],
            "reference": record["ref"],
            "system": record["mt"],
            "score": record["score"],
            "raw_score": record["raw_score"],
        }
        for record in df.to_dict("records")
    ]


# =====================================
# Define WMT18 Dataset
# =====================================


class DatasetContainer(Dataset):
    def __init__(self, dataset: str, language: str = None):
        self.language = language
        self.data = get_dataset(dataset, language)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# =====================================
# Define WMT18 DataLoader
# =====================================


class WMT17(pl.LightningDataModule):

    supported_languages = [
        "cs-en",
        "de-en",
        "fi-en",
        "ru-en",
        "tr-en",
        "zh-en",
        "en-cs",
        "en-de",
        "en-fi",
        "en-lv",
        "en-ru",
        "en-tr",
        "en-zh",
    ]

    def __init__(self, language: str = None, batch_size: int = 32):
        # Please pick one among the available configs:
        #   ['cs-en', 'de-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en', 'en-cs', 'en-de', 'en-fi', 'en-lv', 'en-ru', 'en-tr', 'en-zh']
        super().__init__()

        if language not in self.supported_languages:
            raise Exception(
                """Unsupported language! Please pick one among the available configs:
                ['cs-en', 'de-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en', 'en-cs', 'en-de', 'en-fi', 'en-lv', 'en-ru', 'en-tr', 'en-zh']
                """
            )

        self.language = language
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = DatasetContainer("wmt17", self.language)
        return self

    def train_dataloader(self):
        return DataLoader(self.data_train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, self.batch_size)


class WMT18(pl.LightningDataModule):

    supported_languages = [
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

    def __init__(self, language: str = None, batch_size: int = 32):
        # Please pick one among the available configs:
        #   ['cs-en', 'de-en', 'et-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en', 'en-cs', 'en-de', 'en-et', 'en-fi', 'en-ru', 'en-tr', 'en-zh']
        super().__init__()

        if language not in self.supported_languages:
            raise Exception(
                """Unsupported language! Please pick one among the available configs:
                ['cs-en', 'de-en', 'et-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en', 'en-cs', 'en-de', 'en-et', 'en-fi', 'en-ru', 'en-tr', 'en-zh']
                """
            )

        self.language = language
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = DatasetContainer("wmt18", self.language)
        if stage == "fluency" or stage is None:
            self.data_fluency = get_dataset_fluency(self.language)

        return self

    def test_dataloader(self):
        return DataLoader(self.data_test, self.batch_size)

    def fluency_dataloader(self):
        return DataLoader(self.data_fluency, self.batch_size)


class WMT20(pl.LightningDataModule):

    supported_languages = [
        "cs-en",
        "de-en",
        "iu-en",
        "ja-en",
        "km-en",
        "pl-en",
        "ps-en",
        "ru-en",
        "ta-en",
        "zh-en",
        "en-cs",
        "en-de",
        "en-iu",
        "en-ja",
        "en-km",
        "en-pl",
        "en-ps",
        "en-ru",
        "en-ta",
        "en-zh",
        "de-fr",
        "fr-de",
    ]

    def __init__(self, language: str = None, batch_size: int = 32):
        # Please pick one among the available configs:
        #   ['cs-en', 'de-en', 'iu-en', 'ja-en', 'km-en', 'pl-en', 'ps-en', 'ru-en', 'ta-en', 'zh-en', 'en-cs', 'en-de', 'en-iu', 'en-ja', 'en-km', 'en-pl', 'en-ps', 'en-ru', 'en-ta', 'en-zh', 'de-fr', 'fr-de']
        super().__init__()

        if language not in self.supported_languages:
            raise Exception(
                """Unsupported language! Please pick one among the available configs:
                ['cs-en', 'de-en', 'iu-en', 'ja-en', 'km-en', 'pl-en', 'ps-en', 'ru-en', 'ta-en', 'zh-en', 'en-cs', 'en-de', 'en-iu', 'en-ja', 'en-km', 'en-pl', 'en-ps', 'en-ru', 'en-ta', 'en-zh', 'de-fr', 'fr-de']
                """
            )

        self.language = language
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = DatasetContainer("wmt20", self.language)
        return self

    def test_dataloader(self):
        return DataLoader(self.data_test, self.batch_size)
