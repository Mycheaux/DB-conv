from read_config import read_config

import torch
# import torchmetrics
import pytorch_lightning as pl
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
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.cuda as cuda
# from torch.utils.data import TensorDataset, Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

arch_config = read_config('config/architecture.yaml')
Mapper_input_size = arch_config.get('Mapper_input_size', 3)
Mapper_output_size = arch_config.get('Mapper_output_size', 3)
Mapper_learning_rate = arch_config.get('Mapper_learning_rate', None)
Inverter_input_size = arch_config.get('Inverter_input_size', 3)
Inverter_output_size = arch_config.get('Inverter_input_size', 3)
Inverter_learning_rate = arch_config.get('Inverter_input_size', None)

# Mapper_input_size,Mapper_output_size,Mapper_learning_rate,Inverter_input_size,Inverter_output_size,Inverter_learning_rate = read_config('config/advanced_config.yaml')

config = read_config('config/config.yaml')
dropout_rate = config.get('dropout_rate', 0.0)

class Mapper (pl.LightningModule):
    def __init__(self, Mapper_input_size,Mapper_output_size,Mapper_learning_rate ):
        super(Mapper, self).__init__()
        self.Mapper_input_size = Mapper_input_size
        self.Mapper_output_size = Mapper_output_size

        # self.latent_size = latent_size
        self.Mapper_learning_rate = Mapper_learning_rate
        self.fc1 = nn.Linear(3, 56)
        self.fc2 = nn.Linear(56, 128)
        # self.fc3 = nn.Linear(256, 512)
        # self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(128, 56)
        self.fc6= nn.Linear(56,3)
        # self.fc4= nn.Linear(64,9)
        # self.fc3 = nn.Linear(128, 32)
        # self.fc4 = nn.Linear(32, 4)
        self.drop = nn.Dropout(p=0.3)

        self.bn1 = nn.BatchNorm1d(num_features=56, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        self.bn2 = nn.BatchNorm1d(num_features=128, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        # self.bn3 = nn.BatchNorm1d(num_features=512, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        # self.bn4 = nn.BatchNorm1d(num_features=256, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        self.bn5 = nn.BatchNorm1d(num_features=56, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        # self.bn6 = nn.BatchNorm1d(num_features=3, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )

        self.prelu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.prelu2 = nn.PReLU(num_parameters=1, init=0.25)
        self.prelu3 = nn.PReLU(num_parameters=1, init=0.25)
        self.prelu4 = nn.PReLU(num_parameters=1, init=0.25)
        self.prelu5 = nn.PReLU(num_parameters=1, init=0.25)
        self.prelu6 = nn.PReLU(num_parameters=1, init=0.25)


        # self.softmax = nn.Softmax(dim=-1)
        # self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=1)


    def forward(self,x):
        x= torch.flatten(x, start_dim=1)
        # print (x.shape)
        x = self.fc1(x)
        x = self.prelu1(x)
        # x= self.drop(x)
        x= self.bn1(x)

        x = self.fc2(x)
        x = self.prelu2(x)
        # x= self.drop(x)
        x= self.bn2(x)

        # x = self.fc3(x)
        # x = self.prelu3(x)
        # # x= self.drop(x)
        # x= self.bn3(x)

        # x = self.fc4(x)
        # x = self.prelu4(x)
        # # x= self.drop(x)
        # x= self.bn4(x)

        x = self.fc5(x)
        x = self.prelu5(x)
        # x= self.drop(x)
        x= self.bn5(x)

        x = self.fc6(x)
        x = self.prelu6(x)
        # x= self.drop(x)
        # x= self.bn6(x)


        # x = self.fc4(x)
        # x = self.softmax(x)
        # print(x.shape)

        # x = self.tanh(x)

        return (x)
    
    
class Inverter (pl.LightningModule):
    def __init__(self, Inverter_input_size,Inverter_output_size,Inverter_learning_rate ):
        super(Inverter, self).__init__()
        self.Inverter_input_size = Inverter_input_size
        self.Inverter_output_size = Inverter_output_size

        # self.latent_size = latent_size
        self.Inverter_learning_rate = Inverter_learning_rate
        self.fc1 = nn.Linear(3, 56)
        self.fc2 = nn.Linear(56, 128)
        # self.fc3 = nn.Linear(256, 512)
        # self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(128, 56)
        self.fc6= nn.Linear(56,3)
        # self.fc4= nn.Linear(64,9)
        # self.fc3 = nn.Linear(128, 32)
        # self.fc4 = nn.Linear(32, 4)
        self.drop = nn.Dropout(p=dropout_rate)

        self.bn1 = nn.BatchNorm1d(num_features=56, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        self.bn2 = nn.BatchNorm1d(num_features=128, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        # self.bn3 = nn.BatchNorm1d(num_features=512, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        # self.bn4 = nn.BatchNorm1d(num_features=256, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        self.bn5 = nn.BatchNorm1d(num_features=56, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )
        # self.bn6 = nn.BatchNorm1d(num_features=3, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True )

        self.prelu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.prelu2 = nn.PReLU(num_parameters=1, init=0.25)
        # self.prelu3 = nn.PReLU(num_parameters=1, init=0.25)
        # self.prelu4 = nn.PReLU(num_parameters=1, init=0.25)
        self.prelu5 = nn.PReLU(num_parameters=1, init=0.25)
        self.prelu6 = nn.PReLU(num_parameters=1, init=0.25)


        # self.softmax = nn.Softmax(dim=-1)
        # self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=1)


    def forward(self,x):
        x= torch.flatten(x, start_dim=1)
        # print (x.shape)
        x = self.fc1(x)
        x = self.prelu1(x)
        x= self.drop(x)
        x= self.bn1(x)

        x = self.fc2(x)
        x = self.prelu2(x)
        x= self.drop(x)
        x= self.bn2(x)

        # x = self.fc3(x)
        # x = self.prelu3(x)
        # x= self.drop(x)
        # x= self.bn3(x)

        # x = self.fc4(x)
        # x = self.prelu4(x)
        # x= self.drop(x)
        # x= self.bn4(x)

        x = self.fc5(x)
        x = self.prelu5(x)
        x= self.drop(x)
        x= self.bn5(x)

        x = self.fc6(x)
        x = self.prelu6(x)
        # x= self.drop(x)
        # x= self.bn6(x)


        # x = self.fc4(x)
        # x = self.softmax(x)
        # print(x.shape)

        # x = self.tanh(x)

        return (x)
    
    
    
mapper = Mapper(Mapper_input_size,Mapper_output_size,Mapper_learning_rate)
inverter = Inverter(Inverter_input_size,Inverter_output_size,Inverter_learning_rate)