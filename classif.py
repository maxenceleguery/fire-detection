import uuid
from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd

from dataset import get_dataloaders
from models import CNN


def train_parser():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, help="batch size", default=256)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=True, help="Remove tqdm")
    return parser


def main(kwargs: Namespace) -> None:
    train_load, _, val_load = get_dataloaders(batch_size=kwargs.bs)
    model = CNN().cuda()
    ctx = Namespace(
        num_epochs=kwargs.epochs,
        verbose=kwargs.verbose,
        optimizer=torch.optim.Adam(model.parameters(), kwargs.lr),
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_load, val_loader=val_load ,
        train_losses=[],
    )

    for epoch in range(kwargs.epochs):
        train(epoch, model, ctx)
        test(epoch, model, ctx)

    id_ = uuid.uuid4().fields[0]
    (workspace := Path("training")).mkdir(exist_ok=True)
    print(f"Saving training data to '{workspace}' folder with id {id_}...")
    pd.Series(ctx.train_losses).to_csv(workspace / f"train_loss-{id_}.csv")


def train(epoch: int, model: nn.Module, ctx: Namespace):
    total = wrong = 0
    model.train()
    pbar = tqdm(ctx.train_loader, disable=not ctx.verbose)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.cuda(), labels.cuda()
        out = model(images)

        predictions = torch.argmax(out, dim=1)
        incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

        total += predictions.shape[0]
        wrong += incorrect_indices.shape[0]

        ctx.optimizer.zero_grad()
        loss = ctx.criterion(out, labels)
        loss.backward()
        ctx.optimizer.step()

        ctx.train_losses.append(loss.item())
        if ctx.verbose:
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        elif i%5 == 0:
            print(f"Epoch [{epoch+1}/{ctx.num_epochs}] Iter [{i+1}/] Train loss : {loss.item():.3f}")
    print(f"Epoch [{epoch+1}/{ctx.num_epochs}] Train acc : {100 - 100.*wrong/total:.2f}")


@torch.no_grad()
def test(epoch: int, model: nn.Module, ctx: Namespace):
    total = wrong = 0
    for images, labels in tqdm(ctx.val_loader, disable=not ctx.verbose):
        images, labels = images.cuda(), labels.cuda()
        out = model(images)

        predictions = torch.argmax(out, dim=1)
        incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

        total += predictions.shape[0]
        wrong += incorrect_indices.shape[0]
    print(f"Epoch [{epoch+1}/{ctx.num_epochs}] Val acc : {100 - 100.*wrong/total:.2f}")


if __name__ == "__main__":
    kwargs = train_parser().parse_args()
    main(kwargs)
