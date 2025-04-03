from src.read_config import read_config
# from src.model import Mapper, Inverter

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
# import torch.cuda as cuda
# from torch.utils.data import TensorDataset, Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary




# Mapper Loading
def load_mapper(checkpoint_path, Mapper, Mapper_input_size, Mapper_output_size, Mapper_learning_rate):
    # Define checkpoint kwargs
    ckpt_kwargs = {
        'Mapper_input_size': Mapper_input_size,
        'Mapper_output_size': Mapper_output_size,
        'Mapper_learning_rate': Mapper_learning_rate,
    }

    # Load the checkpoint state_dict
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint["state_dict"]

    # Create a new state_dict with modified keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("mapper."):
            new_state_dict[k[len("mapper."):]] = v  # Remove the "mapper." prefix
        else:
            pass  # Handle other keys if needed

    # Load the modified state_dict into the model
    m = Mapper(**ckpt_kwargs)  # Assuming Mapper is your model class
    m.load_state_dict(new_state_dict, strict=False)  # strict=False handles missing or extra keys

    return m


# Inverter Loading
def load_inverter(checkpoint_path, Inverter, Inverter_input_size, Inverter_output_size, Inverter_learning_rate):
    # Define checkpoint kwargs
    ckpt_kwargs = {
        'Inverter_input_size': Inverter_input_size,
        'Inverter_output_size': Inverter_output_size,
        'Inverter_learning_rate': Inverter_learning_rate,
    }

    # Load the checkpoint state_dict
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint["state_dict"]

    # Create a new state_dict with modified keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("inverter."):
            new_state_dict[k[len("inverter."):]] = v  # Remove the "inverter." prefix
        else:
            pass  # Handle other keys if needed

    # Load the modified state_dict into the model
    i = Inverter(**ckpt_kwargs)  # Assuming Inverter is your model class
    i.load_state_dict(new_state_dict, strict=False)  # strict=False handles missing or extra keys

    return i
