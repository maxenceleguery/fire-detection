import torch
from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights

from typing import Literal

class CNN(nn.Module):
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
    
class Resnet50(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
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
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    """A class for the Decoder"""
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class EncoderMLP(nn.Module):
    """A class to represent an encoder followed by a MLP"""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=61952, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=2),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.mlp(self.encoder(x))
    
    def load_encoder_from_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict)


class DeepEmsemble(nn.Module):
    def __init__(self, model_class: nn.Module | list[nn.Module], model_kwargs: dict | list[dict], emsemble_size: int = None, reduction: Literal["mean", "sum", "vote"] = "mean"):
        super().__init__()

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
            return [torch.optim.Adam(model.parameters(), lr) for model in self.models]
        else:
            return [torch.optim.Adam(model.parameters(), single_lr) for model, single_lr in zip(self.models, lr)]

    def apply_reduction(self, x):
        if self.reduction == "mean":
            return x.mean(dim=0)
        elif self.reduction == "sum":
            return x.sum(dim=0)
        elif self.reduction == "vote":
            raise NotImplementedError()

    def forward(self, x):
        if self.training:
            return [model(x) for model in self.models]
        else:
            return self.apply_reduction(
                torch.stack([model(x) for model in self.models], dim=0)
            )
    
