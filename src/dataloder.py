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

data_path_yaml = read_config('config/data_path.yaml')
data_path =  data_path_yaml.get('data_path')
x_train_path = data_path + '/'+ data_path_yaml.get('x_train','za_train.npy')
x_val_path = data_path + '/'+ data_path_yaml.get('x_val','za_val.npy')
x_test_path = data_path + '/'+ data_path_yaml.get('x_test','za_test.npy')
y_train_path = data_path + '/'+ data_path_yaml.get('y_train','zb_train.npy')
y_val_path = data_path + '/'+ data_path_yaml.get('y_val','zb_val.npy')
y_test_path = data_path + '/'+ data_path_yaml.get('y_test','zb_test.npy')

config = read_config('config/config.yaml')
batch_size = config.get('batch_size', 24)

def load_data(file_path):
    """Load data from .npy or .csv files into NumPy array"""
    if str(file_path).endswith('.npy'):
        return np.load(file_path)
    elif str(file_path).endswith('.csv'):
        return np.loadtxt(file_path, delimiter=',', skiprows=1)
    else:
        raise ValueError("Unsupported input file extension. Use .npy or .csv")



x_train = load_data(str(x_train_path))
x_val = load_data(str(x_val_path))
x_test = load_data(str(x_test_path))

y_train= load_data(str(y_train_path))
y_val = load_data(str(y_val_path))
y_test = load_data(str(y_test_path))

# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_train_tt = torch.tensor(x_train,dtype=torch.float)#.unsqueeze(1)
x_val_tt = torch.tensor(x_val, dtype=torch.float)#.unsqueeze(1)
x_test_tt = torch.tensor(x_test, dtype=torch.float)#.unsqueeze(1)
y_train_tt = torch.tensor(np.array(y_train),dtype=torch.float)#.unsqueeze(1)
y_val_tt = torch.tensor(np.array(y_val), dtype=torch.float)#.unsqueeze(1)
y_test_tt = torch.tensor(np.array(y_test), dtype=torch.float)#.unsqueeze(1)
train_dataset = TensorDataset(x_train_tt, y_train_tt)
val_dataset = TensorDataset(x_val_tt, y_val_tt)
test_dataset = TensorDataset(x_test_tt,y_test_tt)
training_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle= True)
validation_loader = DataLoader(dataset= val_dataset, batch_size=batch_size, shuffle= False)
test_loader = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle= False)