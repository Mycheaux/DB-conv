from src.read_config import read_config

import numpy as np

import torch
# import torchmetrics
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.loggers import MLFlowLogger
# from pytorch_lightning.loggers import NeptuneLogger
# import wandb
# from pytorch_lightning.loggers import WandbLogger

# from pytorch_lightning.plugins import CheckpointIO
# from pytorch_lightning.strategies import SingleDeviceStrategy
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import torch.cuda as cuda
from torch.utils.data import TensorDataset, Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
def load_data(file_path):
    """Load data from .npy or .csv files into NumPy array"""
    if str(file_path).endswith('.npy'):
        return np.load(file_path)
    elif str(file_path).endswith('.csv'):
        return np.loadtxt(file_path, delimiter=',', skiprows=1)
    else:
        raise ValueError("Unsupported input file extension. Use .npy or .csv")

def load_test_data(x_test_path, y_test_path, batch_size):

    x_test = load_data(str(x_test_path))


    y_test = load_data(str(y_test_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # x_train_tt = torch.tensor(x_train,dtype=torch.float)#.unsqueeze(1)
    # x_val_tt = torch.tensor(x_val, dtype=torch.float)#.unsqueeze(1)
    x_test_tt = torch.tensor(x_test, dtype=torch.float)#.unsqueeze(1)
    # y_train_tt = torch.tensor(np.array(y_train),dtype=torch.float)#.unsqueeze(1)
    # y_val_tt = torch.tensor(np.array(y_val), dtype=torch.float)#.unsqueeze(1)
    y_test_tt = torch.tensor(np.array(y_test), dtype=torch.float)#.unsqueeze(1)
    # train_dataset = TensorDataset(x_train_tt, y_train_tt)
    # val_dataset = TensorDataset(x_val_tt, y_val_tt)
    test_dataset = TensorDataset(x_test_tt,y_test_tt)
    # training_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle= True)
    # validation_loader = DataLoader(dataset= val_dataset, batch_size=batch_size, shuffle= False)
    test_loader = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle= False)
    return x_test_tt, y_test_tt