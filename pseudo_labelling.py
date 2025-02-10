from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torchvision.datasets import ImageFolder

from dataset import get_unsupervised_train
from models import CNN, Resnet50, DeepEmsemble


def train_parser():
    parser = ArgumentParser()
    parser.add_argument("--th", type=float, default=.5, help="Threshold for selecting pseudolabel")
    parser.add_argument("--bs", type=int, help="batch size", default=128)
    parser.add_argument("--resize", type=int, help="Resize", default=350)
    parser.add_argument("--model", type=str, default="CNN", choices=["CNN", "resnet50"], help="Model")
    parser.add_argument("--DE_size", type=int, default=1, help="Size of the ensemble")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Load model checkpoint.")
    return parser


def main(kwargs: Namespace):
    trainloader = get_unsupervised_train(batch_size=kwargs.bs, resize=kwargs.resize)
    labels = np.array(ImageFolder(root="./data/train").targets, dtype=np.int8)

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

            for index, conf, label in zip(indexes, *torch.max(out, dim=1)):
                if conf > kwargs.th:
                    trainloader.dataset.set_pseudo_label(index, label)

    trainloader.dataset.save_pseudo_labels()
    pseudo_labels = trainloader.dataset.pseudo_labels
    corrects = labels[pseudo_labels > -1] == pseudo_labels[pseudo_labels > -1]
    print(f"Pseudo labelling accuracy: {corrects.mean() * 100:.2f}% over {len(corrects)}/{len(labels)} samples")

if __name__ == "__main__":
    kwargs = train_parser().parse_args()
    main(kwargs)