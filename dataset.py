import os

def download():
    if not os.path.exists("./data/test/wildfire/-59.03238,51.85132.jpg"):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files("abdelghaniaaba/wildfire-prediction-dataset", "./data", unzip=True)

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch import Generator

# Define paths to train, test, and validation datasets
train_path = "./data/train"
test_path = "./data/test"
valid_path = "./data/valid"

# Create an ImageFolder dataset
def make_datasets(paths, resize: int = 350):
    loaders = []
    for path in paths:
        loader = ImageFolder(
            root=path,
            transform = transforms.Compose(
                [
                transforms.Resize((resize,resize)),
                transforms.ToTensor()
                ]
            )
        )
        loaders.append(loader)
    return loaders

def get_dataloaders(batch_size: int = 256, split: float = 0.7, resize: int = 350):
    train_dataset, test_dataset, valid_dataset = make_datasets([train_path, test_path, valid_path], resize=resize)

    new_train_dataset, new_valid_dataset = random_split(valid_dataset, [int(len(valid_dataset)*split), int(len(valid_dataset)*(1-split))], Generator().manual_seed(777))

    train_load = DataLoader(new_train_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    test_load = DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_load = DataLoader(new_valid_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    return train_load, test_load, val_load

if __name__ == "__main__":
    download()
    train_dataset, test_dataset, valid_dataset = make_datasets([train_path, test_path, valid_path])
    print(f"Images in training data : {len(train_dataset)}")
    print(f"Images in test data : {len(test_dataset)}")
    print(f"Images in validation data : {len(valid_dataset)}")
    print("Classes: \n",train_dataset.classes)