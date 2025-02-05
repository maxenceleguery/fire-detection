import uuid
from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd

from dataset import get_dataloaders, get_unsupervised_train
from models import AutoEncoder, MLP


def train_parser():
    parser = ArgumentParser()
    parser.add_argument("--lr_ae", type=float, default=1e-3)
    parser.add_argument("--lr_mlp", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, help="batch size", default=256)
    parser.add_argument("--epochs_ae", type=int, help="Number of epochs AE", default=10)
    parser.add_argument("--epochs_mlp", type=int, help="Number of epochs MLP", default=15)
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=True, help="Remove tqdm")
    parser.add_argument("--checkpoint_ae", type=Path, default=None, help="Load autoencoder checkpoint.")
    parser.add_argument("--checkpoint_mlp", type=Path, default=None, help="Load mlp checkpoint.")
    return parser


def main(kwargs: Namespace) -> float:
    unsupervised_load = get_unsupervised_train(kwargs.bs, 8)
    train_load, test_load, val_load = get_dataloaders(batch_size=kwargs.bs)
    ae_model = AutoEncoder().cuda()
    mlp_model = MLP().cuda()
    if kwargs.checkpoint_ae and kwargs.checkpoint_mlp:
        print("loading checkpoint...")
        state_dict_ae = torch.load(kwargs.checkpoint_ae, map_location="cuda", weights_only=True)
        ae_model.load_state_dict(state_dict_ae)
        state_dict_mlp = torch.load(kwargs.checkpoint_mlp, map_location="cuda", weights_only=True)
        mlp_model.load_state_dict(state_dict_mlp)

    ctx = Namespace(
        num_epochs_ae=kwargs.epochs_ae,
        num_epochs_mlp=kwargs.epochs_mlp,
        verbose=kwargs.verbose,
        optimizer_ae=torch.optim.Adam(ae_model.parameters(), kwargs.lr_ae),
        optimizer_mlp=torch.optim.Adam(mlp_model.parameters(), kwargs.lr_mlp),
        criterion_ae=nn.MSELoss(),
        criterion_mlp=nn.CrossEntropyLoss(),
        unsupervised_loader=unsupervised_load,
        train_loader=train_load, test_loader=test_load, val_loader=val_load,
        train_losses_ae=[], train_losses_mlp=[]
    )

    # Train the AE
    best_ae_model, best_ae_loss = None, float('inf')
    for epoch in range(kwargs.epochs_ae):
        loss = train_ae(epoch, ae_model, ctx)

        # save best model
        if loss < best_ae_loss:
            best_ae_model = ae_model.state_dict().copy()
            best_ae_loss = loss

    # Train the MLP
    best_mlp_model, best_mlp_loss = None, float('inf')
    for epoch in range(kwargs.epochs_mlp):
        train_mlp(epoch, ae_model, mlp_model, ctx)
        loss = test(epoch, ae_model, mlp_model, ctx)

        # save best model
        if loss < best_mlp_loss:
            best_mlp_model = mlp_model.state_dict().copy()
            best_mlp_loss = loss

    # only perform test if no epochs
    if kwargs.epochs_mlp > 0:
        id_ = uuid.uuid4().fields[0]
        (workspace := Path("training")).mkdir(exist_ok=True)
        print(f"Saving training data to '{workspace}' folder with id {id_}...")

        # save loss
        pd.Series(ctx.train_losses_ae).to_csv(workspace / f"train_loss_ae-{id_}.csv")
        pd.Series(ctx.train_losses_mlp).to_csv(workspace / f"train_loss_mlp-{id_}.csv")
        # save model
        if best_ae_model is not None:
            torch.save(best_ae_model, workspace / f"autoencoder-{id_}.pt")
            print("load best model...")
            ae_model.load_state_dict(best_ae_model)
        if best_mlp_model is not None:
            torch.save(best_mlp_model, workspace / f"mlp-{id_}.pt")
            print("load best model...")
            mlp_model.load_state_dict(best_mlp_model)

    return test(1, ae_model, mlp_model, ctx, val=False)  # FIXME: #1 error while doing test


def train_ae(epoch: int, ae_model: nn.Module, ctx: Namespace) -> None:
    """Training loop for the Auto Encoder."""

    ae_model.train()
    pbar = tqdm(ctx.unsupervised_loader, disable=not ctx.verbose, desc="Train AE")
    losses = []
    for i, (images, _) in enumerate(pbar):
        images = images.cuda()
        images_rec = ae_model(images)

        ctx.optimizer_ae.zero_grad()
        loss = ctx.criterion_ae(images_rec, images)
        loss.backward()
        ctx.optimizer_ae.step()

        losses.append(loss.item())
        ctx.train_losses_ae.append(loss.item())
        if ctx.verbose:
            pbar.set_postfix_str(f"loss_ae={loss.item():.4f}")
        elif i%5 == 0:
            print(f"Epoch [{epoch+1}/{ctx.num_epochs}] Iter [{i+1}/] Train loss : {loss.item():.3f}")
    return sum(losses)/len(losses)

def train_mlp(epoch: int, ae_model: nn.Module, mlp_model: nn.Module, ctx: Namespace) -> None:
    """Training loop for the MLP part."""
    ae_model.train()
    mlp_model.train()

    # Freeze AE model
    for param in ae_model.parameters():
        param.requires_grad = False

    total = wrong = 0
    pbar = tqdm(ctx.train_loader, disable=not ctx.verbose, desc="Train MLP")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.cuda(), labels.cuda()
        latent_reps = ae_model.encoder(images)
        out = mlp_model(latent_reps)

        predictions = torch.argmax(out, dim=1)
        incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

        total += predictions.shape[0]
        if len(incorrect_indices.shape) > 0:
            wrong += incorrect_indices.shape[0]

        ctx.optimizer_mlp.zero_grad()
        loss = ctx.criterion_mlp(out, labels)
        loss.backward()
        ctx.optimizer_mlp.step()

        ctx.train_losses_mlp.append(loss.item())
        if ctx.verbose:
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        elif i%5 == 0:
            print(f"Epoch [{epoch+1}/{ctx.num_epochs_mlp}] Iter [{i+1}/] Train loss MLP : {loss.item():.3f}")
    print(f"Train epoch [{epoch+1}/{ctx.num_epochs_mlp}] acc={100 - 100.*wrong/total:.2f}")


@torch.no_grad()
def test(epoch: int, ae_model: nn.Module, mlp_model: nn.Module, ctx: Namespace, val: bool=True) -> float:
    """Evaluation loop."""

    ae_model.eval()
    mlp_model.eval()

    total = wrong = loss = 0
    loader = ctx.val_loader if val else ctx.test_loader
    val_ = "Val" if val else "Test"
    for images, labels in tqdm(loader, disable=not ctx.verbose, desc=val_):
        images, labels = images.cuda(), labels.cuda()
        out = mlp_model(ae_model.encoder(images))
        loss += ctx.criterion_mlp(out, labels).item() * images.shape[0]

        predictions = torch.argmax(out, dim=1)
        incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

        total += predictions.shape[0]
        if len(incorrect_indices.shape) > 0:
            wrong += incorrect_indices.shape[0]
    loss /= total
    acc = 100 - 100.*wrong/total
    print(val_, f"epoch [{epoch+1}/{ctx.num_epochs_mlp}], acc={acc:.2f}, {loss=:.4f}")
    return loss


if __name__ == "__main__":
    kwargs = train_parser().parse_args()
    main(kwargs)
