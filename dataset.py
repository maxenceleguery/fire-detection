import os
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch import Generator
from typing import List

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def download():
    if not os.path.exists("./data/test/wildfire/-59.03238,51.85132.jpg"):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files("abdelghaniaaba/wildfire-prediction-dataset", "./data", unzip=True)


# Define paths to train, test, and validation datasets
train_path = "./data/train"
test_path = "./data/test"
valid_path = "./data/valid"


class TrainDataset(ImageFolder):
    def __getitem__(self, index: int):
        sample, _ = super().__getitem__(index)
        return sample, -1


# Create an ImageFolder dataset
def make_datasets(paths: List[str], resize: int=350):
    t = T.Compose([ T.Resize(resize), T.ToTensor() ])
    return [
        (TrainDataset if "train" in path else ImageFolder)(root=path, transform=t) 
        for path in paths
    ]


def get_dataloaders(batch_size: int=256, split: float=0.7, num_workers: int=4, resize: int=350, seed: int=777):
    """Load train dataset as a subset of the validation set. Returns (train, test, val) loaders."""

    test_dataset, valid_dataset = make_datasets([test_path, valid_path], resize)
    new_train_dataset, new_valid_dataset = random_split(valid_dataset, (split, (1-split)), Generator().manual_seed(seed))

    train_load = DataLoader(new_train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_load = DataLoader(test_dataset, batch_size, num_workers=num_workers, pin_memory=True)
    val_load = DataLoader(new_valid_dataset, batch_size, num_workers=num_workers, pin_memory=True)

    return train_load, test_load, val_load


def get_unsupervised_train(batch_size: int=256, num_workers: int=4, resize: int=350):
    """Get the training dataset while removing the labels."""
    
    train_dataset = make_datasets([train_path], resize)
    assert type(train_dataset) is TrainDataset, "The training dataset should not use any labels."
    return DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    download()
    train_dataset, test_dataset, valid_dataset = make_datasets([train_path, test_path, valid_path])
    print("Images in training data :", len(train_dataset))
    print("Images in test data :", len(test_dataset))
    print("Images in validation data :", len(valid_dataset))
    print("Classes:", train_dataset.classes)
