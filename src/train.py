from src.read_config import read_config
from src.db_converter import db_converter
from src.utils import get_gpu_info
from src.dataloder import training_loader, validation_loader

import torch
# import torchmetrics
import pytorch_lightning as pl
# from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.loggers import MLFlowLogger
# from pytorch_lightning.loggers import NeptuneLogger
# import wandb
from pytorch_lightning.loggers import WandbLogger

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



config = read_config('config/config.yaml')
grad_clip = config.get('grad_clip',0.0)
num_epochs = config.get('num_epochs', 1000)

model_name = config.get('model_name')
project_name = config.get('project_name')
output_model_path = config.get('output_model_path','output/models')
output_plots_path = config.get('output_plots_path','output/plots')
every_n_epochs = config.get('every_n_epochs', 100)

# wandb_logger = WandbLogger(name = model_name, project = project_name)
# wandb_logger.watch(model, log="all", log_graph=True)

checkpoint_callback = ModelCheckpoint(dirpath=str(output_model_path+'/'+project_name+'/'+model_name),
                                      filename='{epoch}-{training_loss:.2f}-{val_loss:.2f}',
                                      # monitor="valid/val_loss", mode = "min",
                                    #   every_n_train_steps=every_n_train_steps,
                                    #   every_n_epochs= 25
                                      verbose=True,
                                      # save_on_train_epoch_end=True,
                                      save_last = True,
                                      every_n_epochs= every_n_epochs,
                                      save_top_k=-1
                                      )





def train_model():
    model = db_converter()
    
    #wandb logger
    wandb_logger = WandbLogger(name = model_name, project = project_name)
    wandb_logger.watch(model, log="all", log_graph=True)
    
    # GPU availability
    if torch.cuda.is_available():
        accelerator = "gpu"
        print("GPU is available. Using GPU for training.")
        gpu_info = get_gpu_info()
        print("GPU Information:")
        print(gpu_info)
    else:
        accelerator = "cpu"
        print("GPU is not available. Using CPU for training.")
    
    #trainer
    trainer = pl.Trainer(max_epochs=num_epochs,
                     callbacks=[ checkpoint_callback],
                     gradient_clip_val=grad_clip,
                     accelerator=accelerator, #amp_backend="native")
                     val_check_interval=2,
                     logger=wandb_logger)
    
    #training the model
    trainer.fit(model, train_dataloaders=training_loader, val_dataloaders= validation_loader)

