import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from src.data.preprocess import (
    get_source_sentences,
    get_reference_sentences,
    get_system_sentences,
    get_humaneval_values,
)

# python types
from typing import Optional


class WMT18Dataset(Dataset):
    def __init__(self, language: str = None, only_hybrid: bool = True):
        self.language = language
        self.only_hybrid = only_hybrid

        # get the data values
        sources = get_source_sentences(self.language)
        references = get_reference_sentences(self.language)
        system = get_system_sentences(self.language, self.only_hybrid)
        # human_eval = get_humaneval_values(self.language, self.only_hybrid)

        self.data = [
            {
                "model_id": key,
                "source": sources[idx],
                "reference": references[idx],
                "system": system,
            }
            for key, value in system.items()
            for idx, system in enumerate(value)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class WMT18(pl.LightningDataModule):
    def __init__(
        self, language: str = None, only_hybrid: bool = True, batch_size: int = 32
    ):
        # Please pick one among the available configs:
        #   ['cs-en', 'de-en', 'et-en', 'fi-en', 'kk-en', 'ru-en', 'tr-en', 'zh-en']
        super().__init__()

        if not language:
            raise Exception(
                """Language not specified! Please pick one among the available configs:
                ['cs-en', 'de-en', 'et-en', 'fi-en', 'kk-en', 'ru-en', 'tr-en', 'zh-en']
                """
            )

        self.language = language
        self.only_hybrid = only_hybrid
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = None
            self.data_val = None
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_text = WMT18Dataset(self.language, self.only_hybrid)
        if stage == "predict" or stage is None:
            self.data_predict = None

    def train_dataloader(self):
        return DataLoader(self.data_train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_text, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, self.batch_size)
