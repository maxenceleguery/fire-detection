import uuid
from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd

from dataset import get_dataloaders, get_unsupervised_train
from models import CNN, Resnet50, DeepEmsemble
from simmim.vit import ViT


def train_parser():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, help="batch size", default=128)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--resize", type=int, help="Resize", default=350)
    parser.add_argument("--model", type=str, default="CNN", choices=["CNN", "resnet50", "vit"], help="Model")
    parser.add_argument("--DE_size", type=int, default=3, help="Size of the ensemble")
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=True, help="Remove tqdm")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Load model checkpoint.")
    parser.add_argument("--train_fixmatch", action="store_true", help="Train on pseudo labels")
    return parser


def main(kwargs: Namespace) -> float:
    train_load, test_load, val_load = get_dataloaders(batch_size=kwargs.bs, resize=kwargs.resize)
    if kwargs.train_fixmatch:
        train_load = get_unsupervised_train(batch_size=kwargs.bs, resize=kwargs.resize)
        train_load.dataset.load_pseudo_labels()
        train_load.dataset.toggle_pseudo_labels()

    if kwargs.model == "CNN":
        model = DeepEmsemble(CNN, {}, kwargs.DE_size).cuda()
    elif kwargs.model == "resnet50":
        model = DeepEmsemble(Resnet50, {}, kwargs.DE_size).cuda()
    elif kwargs.model == "vit":
        model = DeepEmsemble(
            ViT,
            {
                "image_size": 256,
                "patch_size": 16,
                "num_classes": 2,
                "dim": 256,
                "depth": 18,
                "heads": 12,
                "mlp_dim": 512,
            },
            kwargs.DE_size
        ).cuda()
        
    if kwargs.checkpoint:
        print("loading checkpoint...")
        state_dict = torch.load(kwargs.checkpoint, map_location="cuda", weights_only=True)
        model.load_state_dict(state_dict)
    
    ctx = Namespace(
        num_epochs=kwargs.epochs,
        verbose=kwargs.verbose,
        optimizers=model.get_optimizers(kwargs.lr),
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
    
    return test(1, model, ctx, val=False)  # FIXME: #1 error while doing test


def train(epoch: int, model: nn.Module, ctx: Namespace) -> None:
    """Training loop."""

    model.train()
    total = wrong = 0
    pbar = tqdm(ctx.train_loader, disable=not ctx.verbose, desc="Train")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.cuda(), labels.cuda()
        outs = model(images)

        predictions = torch.argmax(model.apply_reduction(torch.stack(outs, dim=0)), dim=1)
        incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

        total += predictions.shape[0]
        if len(incorrect_indices.shape) > 0:
            wrong += incorrect_indices.shape[0]

        final_loss = 0
        for optimizer, out in zip(ctx.optimizers, outs):
            optimizer.zero_grad()
            loss = ctx.criterion(out, labels)
            loss.backward()
            optimizer.step()
            final_loss += loss

        ctx.train_losses.append(final_loss.item()/model.emsemble_size)
        if ctx.verbose:
            pbar.set_postfix_str(f"loss={final_loss.item():.4f}")
        elif i%5 == 0:
            print(f"Epoch [{epoch+1}/{ctx.num_epochs}] Iter [{i+1}/] Train loss : {final_loss.item():.3f}")
    print(f"Train epoch [{epoch+1}/{ctx.num_epochs}] acc={100 - 100.*wrong/total:.2f}")


@torch.no_grad()
def test(epoch: int, model: nn.Module, ctx: Namespace, val: bool=True) -> float:
    """Evaluation loop."""

    model.eval()
    total = wrong = loss = 0
    loader = ctx.val_loader if val else ctx.test_loader
    val_ = "Val" if val else "Test"
    for images, labels in tqdm(loader, disable=not ctx.verbose, desc=val_):
        images, labels = images.cuda(), labels.cuda()
        out = model(images)
        loss += ctx.criterion(out, labels).item() * images.shape[0]

        predictions = torch.argmax(out, dim=1)
        incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

        total += predictions.shape[0]
        if len(incorrect_indices.shape) > 0:
            wrong += incorrect_indices.shape[0]
    loss /= total
    acc = 100 - 100.*wrong/total
    print(val_, f"epoch [{epoch+1}/{ctx.num_epochs}], acc={acc:.2f}, {loss=:.4f}")
    return loss


if __name__ == "__main__":
    kwargs = train_parser().parse_args()
    main(kwargs)
