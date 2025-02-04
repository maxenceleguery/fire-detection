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

The script `classifiy.py` trains a classifier using the labels of the validation dataset.