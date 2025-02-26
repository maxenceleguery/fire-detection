import uuid
from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
from copy import deepcopy

from dataset import get_dataloaders, get_unsupervised_train
from models import AutoEncoder, EncoderMLP
from metrics.scoring import Scoring


def train_parser():
    parser = ArgumentParser()
    parser.add_argument("--lr_ae", type=float, default=1e-3)
    parser.add_argument("--lr_emlp", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, help="batch size", default=256)
    parser.add_argument("--epochs_ae", type=int, help="Number of epochs AE", default=5)
    parser.add_argument("--epochs_emlp", type=int, help="Number of epochs emlp", default=20)
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=True, help="Remove tqdm")
    parser.add_argument("--checkpoint_ae", type=Path, default=None, help="Load autoencoder checkpoint.")
    parser.add_argument("--checkpoint_emlp", type=Path, default=None, help="Load emlp checkpoint.")
    parser.add_argument("--resnet", action="store_true", default=False, help="Set the encoder to a ResNet50")
    return parser


def main(kwargs: Namespace) -> float:
    unsupervised_load = get_unsupervised_train(kwargs.bs, 8)
    train_load, test_load, val_load = get_dataloaders(batch_size=kwargs.bs)
    ae_model = AutoEncoder(resnet = kwargs.resnet).cuda()
    emlp_model = EncoderMLP(resnet = kwargs.resnet).cuda()
    if kwargs.checkpoint_ae:
        print("loading checkpoint...")
        state_dict_ae = torch.load(kwargs.checkpoint_ae, map_location="cuda", weights_only=True)
        ae_model.load_state_dict(state_dict_ae)

    if kwargs.checkpoint_emlp:
        print("loading checkpoint...")
        state_dict_emlp = torch.load(kwargs.checkpoint_emlp, map_location="cuda", weights_only=True)
        emlp_model.load_state_dict(state_dict_emlp)

    ctx = Namespace(
        num_epochs_ae=kwargs.epochs_ae,
        num_epochs_emlp=kwargs.epochs_emlp,
        verbose=kwargs.verbose,
        optimizer_ae=torch.optim.Adam(ae_model.parameters(), kwargs.lr_ae),
        optimizer_emlp=torch.optim.Adam(emlp_model.parameters(), kwargs.lr_emlp),
        criterion_ae=nn.MSELoss(),
        criterion_emlp=nn.CrossEntropyLoss(),
        unsupervised_loader=unsupervised_load,
        train_loader=train_load, test_loader=test_load, val_loader=val_load,
        train_losses_ae=[], train_losses_emlp=[]
    )

    # Train the AE
    best_encoder_model, best_ae_loss = None, float('inf')
    for epoch in range(kwargs.epochs_ae):
        loss = train_ae(epoch, ae_model, ctx)

        # save best encoder model
        if loss < best_ae_loss:
            best_encoder_model = deepcopy(ae_model.encoder.state_dict())
            best_ae_loss = loss

    # Load the trained encoder into the Encodermlp architecture
    if best_encoder_model is not None:
        emlp_model.load_encoder_from_state_dict(best_encoder_model)

    # Train the Encodermlp
    best_emlp_model, best_emlp_loss = None, float('inf')
    for epoch in range(kwargs.epochs_emlp):
        train_emlp(epoch, emlp_model, ctx)
        loss = test(epoch, emlp_model, ctx)

        # save best model
        if loss < best_emlp_loss:
            best_emlp_model = emlp_model.state_dict().copy()
            best_emlp_loss = loss

    # only perform test if no epochs
    if kwargs.epochs_emlp > 0:
        id_ = uuid.uuid4().fields[0]
        (workspace := Path("training")).mkdir(exist_ok=True)
        print(f"Saving training data to '{workspace}' folder with id {id_}...")

        # save loss
        pd.Series(ctx.train_losses_ae).to_csv(workspace / f"train_loss_ae-{id_}.csv")
        pd.Series(ctx.train_losses_emlp).to_csv(workspace / f"train_loss_emlp-{id_}.csv")
        # save model
        if best_emlp_model is not None:
            torch.save(best_emlp_model, workspace / f"encoder_emlp-{id_}.pt")
            print("load best model...")
            emlp_model.load_state_dict(best_emlp_model)

    return test(1, emlp_model, ctx, val=False)  # FIXME: #1 error while doing test


def train_ae(epoch: int, ae_model: nn.Module, ctx: Namespace) -> None:
    """Training loop for the Auto Encoder."""

    ae_model.train()
    pbar = tqdm(ctx.unsupervised_loader, disable=not ctx.verbose, desc="Train AE")
    losses = []
    for i, (images, _, _) in enumerate(pbar):
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

def train_emlp(epoch: int, emlp_model: nn.Module, ctx: Namespace) -> None:
    """Training loop for the Encoderemlp part."""
    emlp_model.train()

    total = wrong = 0
    pbar = tqdm(ctx.train_loader, disable=not ctx.verbose, desc="Train EncoderMLP")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.cuda(), labels.cuda()
        out = emlp_model(images)

        predictions = torch.argmax(out, dim=1)
        incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

        total += predictions.shape[0]
        if len(incorrect_indices.shape) > 0:
            wrong += incorrect_indices.shape[0]

        ctx.optimizer_emlp.zero_grad()
        loss = ctx.criterion_emlp(out, labels)
        loss.backward()
        ctx.optimizer_emlp.step()

        ctx.train_losses_emlp.append(loss.item())
        if ctx.verbose:
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        elif i%5 == 0:
            print(f"Epoch [{epoch+1}/{ctx.num_epochs_emlp}] Iter [{i+1}/] Train loss Encoderemlp : {loss.item():.3f}")
    print(f"Train epoch [{epoch+1}/{ctx.num_epochs_emlp}] acc={100 - 100.*wrong/total:.2f}")


@torch.no_grad()
def test(epoch: int, emlp_model: nn.Module, ctx: Namespace, val: bool=True) -> float:
    """Evaluation loop."""

    emlp_model.eval()

    total = wrong = loss = 0
    preds_list = []
    labels_list = []

    loader = ctx.val_loader if val else ctx.test_loader
    val_ = "Val" if val else "Test"
    for images, labels in tqdm(loader, disable=not ctx.verbose, desc=val_):
        images, labels = images.cuda(), labels.cuda()
        out = emlp_model(images)
        loss += ctx.criterion_emlp(out, labels).item() * images.shape[0]

        predictions = torch.argmax(out, dim=1)
        incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

        preds_list.extend(torch.nn.functional.softmax(out, dim=-1)[:, 1].tolist())
        labels_list.extend(labels.tolist())

        total += predictions.shape[0]
        if len(incorrect_indices.shape) > 0:
            wrong += incorrect_indices.shape[0]
    loss /= total
    acc = 100 - 100.*wrong/total
    print(val_, f"epoch [{epoch+1}/{ctx.num_epochs_emlp}], acc={acc:.2f}, {loss=:.4f}")

    if not val:
        scoring = Scoring(preds_list, labels_list)
        print(scoring)
    return loss


if __name__ == "__main__":
    kwargs = train_parser().parse_args()
    main(kwargs)
