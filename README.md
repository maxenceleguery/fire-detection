# Fire detection

Install into an environment using `pip install -r requirements.txt`, you also need to have `pytorch` and `torchvision` installed.
To create a job on a node use this command:

```bash
srun --pty --time=02:00:00 --partition=ENSTA-h100 --gpus=1 bash
python3 classif.py --model resnet50 --resize 256 --bs 128
python3 classif_DE.py --model resnet50 --resize 256 --bs 64
python3 pseudo_labelling.py --bs 64 --resize 256 --model resnet50 --DE_size 3 --checkpoint ./training/de-98.pt
python3 ae_mlp.py --lr_ae 1e-3 --lr_emlp 1e-4 --bs 256 --epochs_emlp 20
```

To install the dataset in the `data` folder, run the script `python dataset.py`. You need a kaggle API key to download the dataset.

## Structure

The loading of the data is performed in the `dataset.py` file.
The different models used are stored in `models.py`. 

The script `train.py` trains a CNN classifier as default using the labels of the validation dataset.
- To load a model to be trained, use the `--checkpoint` argument to provide the path to the model weights. To only test a model, one can use the argument `--epochs 0`.
- This script also supports deep ensemble models that are load with a `-DE` adding to the model's name and `--DE_size` specifying the ensemble size (defaults to 3).
- The scripts also support training using fixmatch with the `--fixmatch` option. The training dataset is replaced by the pseudolabelised one.

The script `ae_mlp.py` trains first an Auto Encoder using the training set (without the labels), and then trains a MLP on the labels of the validation dataset using the latent representation from the trained encoder. To load a model to be trained, use the `--checkpoint_ae` and `--checkpoint_mlp` argument to provide the path to the models (AE and MLP) weights. To only test a model, one can use the argument `--epochs 0`.

## Trainings

A training using fixmatch can be performed this way:
- train a model using `train.py -m [model] ...`
- perform pseudo_labelling with it using `pseudo_labelling.py -m [model] --checkpoint [previous checkpoint] ...`
- train the first model or a new model with fixmatch using `train.py -m [model] --checkpoint [] --fixmatch ...`
Note: I (clément) can't get good results using fixmatch on the default CNN as doing it decrease the performances (I do it on 2 epochs). The pseudolabels have over 90% of correctness if I look at the training labels (I used a threhold of 0.99 to create them).

## Weights

Available on HuggingFace :
- Maxenceleguery/de-3-resnet-50
- Maxenceleguery/cnn-ae-pretrained
- Maxenceleguery/vit-simmim

```python
from models import load_model

model = load_model("resnet50-DE", DE_size=3)
model.from_pretrained("Maxenceleguery/de-3-resnet-50")
```

### Testing models

```bash
python3 train.py --epochs 0 --bs 128 --resize 256 -m resnet50-DE --DE_size 3 --hugging Maxenceleguery/de-3-resnet-50
python3 train.py --epochs 0 --bs 128 --resize 350 -m EncoderMLP --hugging Maxenceleguery/cnn-ae-pretrained
python3 train.py --epochs 0 --bs 128 --resize 256 -m vit --hugging Maxenceleguery/vit-simmim
```

- de-3-resnet-50 (fixmatch) 99.16 %
- cnn-ae-pretrained 93.32 %
- vit-simmim 91.86 %