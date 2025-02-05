# Fire detection

Install into an environment using `pip install -r requirements.txt`, you also need to have `pytorch` and `torchvision` installed.
To create a job on a node use this command:

```bash
srun --pty --time=02:00:00 --partition=ENSTA-h100 --gpus=1 bash
```

To install the dataset in the `data` folder, run the script `python dataset.py`. You need a kaggle API key to download the dataset.

## Structure

The loading of the data is performed in the `dataset.py` file.
The different models used are stored in `models.py`. 

The script `classif.py` trains a classifier using the labels of the validation dataset. To load a model to be trained, use the `--checkpoint` argument to provide the path to the model weights. To only test a model, one can use the argument `--epochs 0`. 

The script `ae_mlp.py` trains first an Auto Encoder using the training set (without the labels), and then trains a MLP on the labels of the validation dataset using the latent representation from the trained encoder. To load a model to be trained, use the `--checkpoint_ae` and `--checkpoint_mlp` argument to provide the path to the models (AE and MLP) weights. To only test a model, one can use the argument `--epochs 0`. 