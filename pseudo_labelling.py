import uuid
from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd

from dataset import get_unsupervised_train
from models import CNN, Resnet50, DeepEmsemble

def train_parser():
    parser = ArgumentParser()
    parser.add_argument("--bs", type=int, help="batch size", default=128)
    parser.add_argument("--resize", type=int, help="Resize", default=350)
    parser.add_argument("--model", type=str, default="CNN", choices=["CNN", "resnet50"], help="Model")
    parser.add_argument("--DE_size", type=int, default=1, help="Size of the ensemble")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Load model checkpoint.")
    return parser

def main(kwargs: Namespace):
    trainloader = get_unsupervised_train(batch_size=kwargs.bs, resize=kwargs.resize)

    if kwargs.DE_size > 1:
        if kwargs.model == "CNN":
            model = DeepEmsemble(CNN, {}, kwargs.DE_size).cuda()
        elif kwargs.model == "resnet50":
            model = DeepEmsemble(Resnet50, {}, kwargs.DE_size).cuda()
    else:
        if kwargs.model == "CNN":
            model = CNN().cuda()
        elif kwargs.model == "resnet50":
            model = Resnet50().cuda()

    if kwargs.checkpoint is None:
        raise ValueError("A checkpoint is need for pseudo labelization.")
    
    print("loading checkpoint...")
    state_dict = torch.load(kwargs.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict)

    with torch.no_grad():
        model.eval()
        for images, _, indexes in tqdm(trainloader, desc="Pseudo labelling..."):
            images = images.cuda()
            out = model(images)

            predictions = torch.argmax(out, dim=1)
            for index, label in zip(indexes, predictions):
                trainloader.dataset.set_pseudo_label(index, label)

    trainloader.dataset.save_pseudo_labels()

if __name__ == "__main__":
    kwargs = train_parser().parse_args()
    main(kwargs)