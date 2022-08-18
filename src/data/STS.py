import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# python types
from typing import Optional


DIRNAME = os.path.dirname(__file__)

# =====================================
# Helper Functions
# =====================================


def construct_dataset_path(type: str):
    rel_path = f"../../data/raw/sts/sts-{type}.csv"
    source_path = os.path.join(DIRNAME, rel_path)
    return source_path


def get_dataset(type: str):
    with open(construct_dataset_path(type), "r", encoding="utf8") as f:
        return [line.strip().split("\t") for line in f.readlines()]


# =====================================
# Define STS Dataset
# =====================================


class STSDataset(Dataset):
    def __init__(self, type):
        self.type = type

        # get the data values
        lines = get_dataset(self.type)
        self.data = [
            {
                "genre": values[0],
                "filename": values[1],
                "year": values[2],
                "score": float(values[4 if len(values) >= 7 else 3]),
                "sentence1": values[5 if len(values) >= 7 else 4],
                "sentence2": values[6 if len(values) >= 7 else 5],
            }
            for values in lines
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# =====================================
# Define STS DataLoader
# =====================================


class STS(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = STSDataset("train")
            self.data_val = STSDataset("dev")
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = STSDataset("test")
        if stage == "predict" or stage is None:
            self.data_predict = STSDataset("test")

        return self

    def train_dataloader(self):
        return DataLoader(self.data_train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, self.batch_size)
