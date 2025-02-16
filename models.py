import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights
from einops.layers.torch import Rearrange

from typing import Literal

MODELS = ["CNN", "EncoderMLP", "vit", "resnet50", "CNN-DE", "resnet50-DE", "vit-DE"]
VIT_CONFIG = dict( image_size=256, num_classes=2, patch_size=16, dim=256, depth=18, heads=12, mlp_dim=512 )

def load_model(name: str, **kwargs):
    match name:
        case "CNN":  model = CNN()
        case "EncoderMLP": model = EncoderMLP()
        case "resnet50":  model = Resnet50()
        case "resnet50_AE": model = EncoderMLP(resnet=True)
        case "vit":
            from simmim.vit import ViT
            model = ViT(**VIT_CONFIG)
        case "CNN-DE":  model = DeepEmsemble(CNN, {}, kwargs["DE_size"])
        case "resnet50-DE":  model = DeepEmsemble(Resnet50, {}, kwargs["DE_size"])
        case "vit-DE":  model = DeepEmsemble(ViT, VIT_CONFIG, kwargs["DE_size"])
        case _:  raise ValueError(f"Unknown model {name}")
    return model.cuda()



class CNN(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=2, stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(in_features=61952, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.network(x)

class Resnet50(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()

        self.base = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)

        self.base.fc = nn.Sequential(
            nn.Linear(in_features=self.base.fc.in_features, out_features=2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.base(x)

class ResNetBlock(nn.Module):
    """A class to represent ResNet Blocks"""
    def __init__(self, in_channels, out_channels, n_hidden):
        super().__init__()
        self.resnet = nn.Sequential(
        nn.ReLU(),
        # We want to keep the same size, so we add padding
        nn.Conv2d(in_channels, n_hidden, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(n_hidden, out_channels, kernel_size=1, bias=False)
        )
    def forward(self, x):
        return self.resnet(x) + x

class Encoder(nn.Module):
    """A class for the Encoder"""
    def __init__(self, resnet:bool=False):
        super().__init__()
        if resnet:
            # Only keep the encoder part of the ResNet
            self.encoder = Resnet50()
            self.encoder.base.fc = Rearrange("b e -> b e 1 1")
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size=2, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=2, stride=1, padding=1), 
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=2, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.3),
            )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    """A class for the Decoder"""
    def __init__(self, resnet:bool=False):
        super().__init__()
        if resnet:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(2048, 512, kernel_size=5, stride=4, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=5, stride=4, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=5, stride=4, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=1, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=2, stride=2),
                nn.BatchNorm2d(3),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, resnet:bool=False):
        super().__init__()
        self.encoder = Encoder(resnet)
        self.decoder = Decoder(resnet)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class EncoderMLP(nn.Module, PyTorchModelHubMixin):
    """A class to represent an encoder followed by a MLP"""
    def __init__(self, resnet:bool=False):
        super().__init__()
        self.encoder = Encoder(resnet)
        input_features = 61952
        if resnet:
            input_features = 2048
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=2),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.mlp(self.encoder(x))
    
    def load_encoder_from_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict)


class DeepEmsemble(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_class: nn.Module | list[nn.Module] = Resnet50, model_kwargs: dict | list[dict] = None, emsemble_size: int = None, reduction: Literal["mean", "sum", "vote"] = "mean"):
        super().__init__()

        if model_kwargs is None:
            model_kwargs = {}

        self.emsemble_size = emsemble_size
        self.reduction = reduction

        if isinstance(model_class, list):
            self.models = nn.ModuleList([model(**kwargs) for model, kwargs in zip(model_class, model_kwargs, strict=True)])

        else:
            assert isinstance(model_kwargs, dict)
            assert emsemble_size is not None
            self.models = nn.ModuleList([model_class(**model_kwargs) for _ in range(emsemble_size)])


    def get_optimizers(self, lr: float | list[float]):
        if isinstance(lr, float):
            return [torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4) for model in self.models]
        return [torch.optim.Adam(model.parameters(), lr_, weight_decay=5e-4) for model, lr_ in zip(self.models, lr)]

    def apply_reduction(self, x):
        if self.reduction == "mean":  return x.mean(dim=0)
        elif self.reduction == "sum":  return x.sum(dim=0)
        elif self.reduction == "vote":  raise NotImplementedError()

    def forward(self, x):
        outs = [model(x) for model in self.models]
        if self.training:
            return outs
        return self.apply_reduction(torch.stack(outs, dim=0))
