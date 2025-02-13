import uuid
from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd

from dataset import get_dataloaders, get_unsupervised_train
from models import load_model, MODELS, DeepEmsemble
import fixmatch


def train_parser():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, help="batch size", default=128)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--resize", type=int, help="Resize", default=350)
    parser.add_argument("-m", "--model", type=str, default="CNN", choices=MODELS, help="Model")
    parser.add_argument("--DE_size", type=int, default=3, help="Size of the ensemble when a DE model is selected")
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=True, help="Remove tqdm")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Load model checkpoint.")
    parser.add_argument("--hugging", type=str, default=None, choices=["Maxenceleguery/de-3-resnet-50"], help="Load model from HuggingFace")
    parser.add_argument("--fixmatch", action="store_true", help="Train on pseudo labels")
    return parser


def main(kwargs: Namespace) -> float:
    train_load, test_load, val_load = get_dataloaders(batch_size=kwargs.bs, resize=kwargs.resize)

    if kwargs.fixmatch:
        train_load = get_unsupervised_train(batch_size=kwargs.bs, resize=kwargs.resize)
        assert Path("pseudo_labels.csv").exists()
        train_load.dataset.load_pseudo_labels()
        train_load.dataset.toggle_pseudo_labels()
        train_load.dataset.transforms = fixmatch.get_transform(kwargs.resize)
    

    model = load_model(kwargs.model, DE_size=kwargs.DE_size)
    
    if kwargs.hugging:
        model.from_pretrained(kwargs.hugging)
    else:
        if kwargs.checkpoint:
            print("loading checkpoint...")
            state_dict = torch.load(kwargs.checkpoint, map_location="cuda", weights_only=True)
            model.load_state_dict(state_dict)

    if isinstance(model, DeepEmsemble):
        optim = model.get_optimizers(kwargs.lr)
    else:
        optim = torch.optim.Adam(model.parameters(), kwargs.lr)

    ctx = Namespace(
        num_epochs=kwargs.epochs,
        verbose=kwargs.verbose,
        optimizer=optim,
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_load, test_loader=test_load, val_loader=val_load,
        train_losses=[],
    )

    best_model, best_loss = None, float('inf')
    for epoch in range(kwargs.epochs):
        train(epoch, model, ctx)
        loss = test(epoch, model, ctx)

        # save best model
        if loss < best_loss:
            best_model = model.state_dict().copy()
            best_loss = loss

    # only perform test if no epochs
    if kwargs.epochs > 0:
        id_ = uuid.uuid4().fields[0]
        (workspace := Path("training")).mkdir(exist_ok=True)
        print(f"Saving training data to '{workspace}' folder with id {id_}...")

        # save loss
        pd.Series(ctx.train_losses).to_csv(workspace / f"train_loss-{id_}.csv")
        # save model
        if best_model is not None:
            torch.save(best_model, workspace / f"classifier-{id_}.pt")
            print("load best model...")
            model.load_state_dict(best_model)

    ctx.num_epochs=1
    return test(0, model, ctx, val=False)


def train(epoch: int, model: nn.Module, ctx: Namespace) -> None:
    """Training loop."""

    model.train()
    corrects = 0
    pbar = tqdm(ctx.train_loader, disable=not ctx.verbose, desc="Train")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.cuda(), labels.cuda()
        out = model(images)

        if isinstance(model, DeepEmsemble):
            loss = 0
            for optim, o in zip(ctx.optimizer, out):
                optim.zero_grad()
                loss_ = ctx.criterion(o, labels)
                loss_.backward()
                optim.step()
                loss += loss_
            loss /= model.emsemble_size
            predictions = model.apply_reduction(torch.stack(out, dim=0)).argmax(dim=1)
        else:
            # default training step
            predictions = out.argmax(dim=1)

            ctx.optimizer.zero_grad()
            loss = ctx.criterion(out, labels)
            loss.backward()
            ctx.optimizer.step()

        corrects += (predictions == labels).sum().item()
        ctx.train_losses.append(loss.item())
        if ctx.verbose:
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        elif i%5 == 0:
            print(f"Epoch [{epoch+1}/{ctx.num_epochs}] Iter [{i+1}/] Train loss : {loss.item():.3f}")
    acc = corrects / len(ctx.train_loader.dataset)
    print(f"Train epoch [{epoch+1}/{ctx.num_epochs}] acc={acc * 100:.2f}")


@torch.no_grad()
def test(epoch: int, model: nn.Module, ctx: Namespace, val: bool=True) -> float:
    """Evaluation loop."""

    model.eval()
    corrects = loss = 0
    loader = ctx.val_loader if val else ctx.test_loader
    val_ = "Val" if val else "Test"
    for images, labels in tqdm(loader, disable=not ctx.verbose, desc=val_):
        images, labels = images.cuda(), labels.cuda()
        out = model(images)
        loss += ctx.criterion(out, labels).item() * images.shape[0]

        predictions = torch.argmax(out, dim=1)
        corrects += (predictions == labels).sum().item()

    total = len(loader.dataset)
    loss /= total
    acc = 100 * corrects / total
    print(val_, f"epoch [{epoch+1}/{ctx.num_epochs}], acc={acc:.2f}, {loss=:.4f}")
    return loss


if __name__ == "__main__":
    kwargs = train_parser().parse_args()
    main(kwargs)
