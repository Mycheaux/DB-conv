from src.read_config import read_config
from src.model import Mapper, Inverter
from src.utils import havrda_charvat_entropy, k_twin

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
# import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torch.cuda as cuda
# from torch.utils.data import TensorDataset, Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

adv_config = read_config('config/advanced_config.yaml')
lr_m = adv_config.get('lr_m', 0.0001)
b1_m = adv_config.get('b1_m', 0.9)
b2_m = adv_config.get('b2_m', 0.999)
lr_i = adv_config.get('lr_i', 0.0001)
b1_i = adv_config.get('b1_i', 0.9)
b2_i = adv_config.get('b2_i', 0.999)
k_in_ktwin_for_loss = adv_config.get('k_in_ktwin_for_loss', 5)

config = read_config('config/config.yaml')
batch_size = config.get('batch_size', 24)
num_epochs = config.get('num_epochs', 1000)
dropout_rate = config.get('dropout_rate', 0.0)
l1_lamb = config.get('l1_lamb',1)
l2_lamb = config.get('l2_lamb', 1)
pt_lamb = config.get('pt_lamb',0)
hce_lamb = config.get('hce_lamb',0.001)
tji_lamb = config.get('tje_lam',0.1)
mapper_repeat = config.get('mapper_repeat',1)
inverter_repeat = config.get('inverter_repeat',1)



arch_config = read_config('config/architecture.yaml')
Mapper_input_size = arch_config.get('Mapper_input_size', 3)
Mapper_output_size = arch_config.get('Mapper_output_size', 3)
Mapper_learning_rate = arch_config.get('Mapper_learning_rate', None)
Inverter_input_size = arch_config.get('Inverter_input_size', 3)
Inverter_output_size = arch_config.get('Inverter_input_size', 3)
Inverter_learning_rate = arch_config.get('Inverter_input_size', None)





class db_converter(pl.LightningModule):

    def __init__(
        self,
        lr_m: float = lr_m,
        b1_m: float = b1_m,
        b2_m: float = b2_m,
        lr_i: float = lr_i,
        b1_i: float = b1_i,
        b2_i: float = b2_i,

        batch_size: int = batch_size,
        num_epochs: int = num_epochs,
        **kwargs,):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False


        self.mapper = Mapper(Mapper_input_size, Mapper_output_size,  Mapper_learning_rate)
        self.inverter = Inverter(Inverter_input_size, Inverter_output_size,  Inverter_learning_rate)

    def forward(self, data):
      return self.mapper(data)

    def mapper_loss(self,hat,true):
        norm = torch.sqrt(torch.sum(hat ** 2, -1, keepdim=True))
        normalized_emb = hat / norm
        similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
        batch_size = hat.size(0)
        pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1)) # cos similarity
        hce_loss = havrda_charvat_entropy(hat, true, parameter_a=1.3)
        l1_loss = F.l1_loss(hat, true)
        l2_loss = F.mse_loss(hat, true)
        tji = 1- k_twin(hat.detach().cpu().numpy(), true.detach().cpu().numpy(), 5, "L2")
        loss = l1_lamb * l1_loss + l2_lamb * l2_loss + pt_lamb *pt + hce_lamb * hce_loss + tji_lamb * tji
        self.log("train/mapping_loss", loss, on_step=False, on_epoch=True)
        self.log("train/mapping_l1_loss", l1_loss, on_step=False, on_epoch=True)
        self.log("train/mapping_l2_loss", l2_loss, on_step=False, on_epoch=True)
        self.log("train/mapping_pt_loss", pt, on_step=False, on_epoch=True)
        self.log("train/mapping_hce_loss", hce_loss, on_step=False, on_epoch=True)
        self.log("train/mapping_tji_loss", tji, on_step=False, on_epoch=True)
        return loss

    def inverter_loss(self,hat,true):
        norm = torch.sqrt(torch.sum(hat ** 2, -1, keepdim=True))
        normalized_emb = hat / norm
        similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
        batch_size = hat.size(0)
        pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1)) # cos similarity
        hce_loss = havrda_charvat_entropy(hat, true, parameter_a=1.3)
        l1_loss = F.l1_loss(hat, true)
        l2_loss = F.mse_loss(hat, true)
        tji = 1 - k_twin(hat.detach().cpu().numpy(), true.detach().cpu().numpy(), k_in_ktwin_for_loss, "L2")
        loss = l1_lamb * l1_loss + l2_lamb * l2_loss + pt_lamb *pt + hce_lamb * hce_loss + tji_lamb * tji
        self.log("train/inverting_loss", loss, on_step=False, on_epoch=True)
        self.log("train/inverting_l1_loss", l1_loss, on_step=False, on_epoch=True)
        self.log("train/inverting_l2_loss", l2_loss, on_step=False, on_epoch=True)
        self.log("train/inverting_pt_loss", pt, on_step=False, on_epoch=True)
        self.log("train/inverting_hce_loss", hce_loss, on_step=False, on_epoch=True)
        self.log("train/inverting_tji_loss", tji, on_step=False, on_epoch=True)
        return loss


    def training_step(self, batch, batch_idx):
        optimizer_m, optimizer_i = self.optimizers()

        for i in range (0,mapper_repeat):
            #mapper
            self.toggle_optimizer(optimizer_m)
            data, label = batch
            # y_hat = self.mapper(data)
            # y_hat_hat = self.inverter(y_hat)
            # m_loss = self.mapper_loss(y_hat_hat, data)
            x_hat = self.inverter(label)
            x_hat_hat = self.mapper(x_hat)
            i_loss = self.inverter_loss(x_hat_hat, label)
            self.manual_backward(i_loss)
            optimizer_m.step()
            optimizer_m.zero_grad()
            self.untoggle_optimizer(optimizer_m)
        for i in range (0,inverter_repeat):
            #inverter
            self.toggle_optimizer(optimizer_i)
            data, label = batch
            # x_hat = self.inverter(label)
            # x_hat_hat = self.mapper(x_hat)
            # i_loss = self.inverter_loss(x_hat_hat, label)
            y_hat = self.mapper(data)
            y_hat_hat = self.inverter(y_hat)
            m_loss = self.mapper_loss(y_hat_hat, data)
            self.manual_backward(m_loss)
            optimizer_i.step()
            optimizer_i.zero_grad()
            self.untoggle_optimizer(optimizer_i)

        # self.log("train/m_loss", m_loss, on_step=False, on_epoch=True)
        # self.log("train/i_loss", i_loss, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        data, label = batch
        y_hat = self.mapper(data)
        y_hat_hat = self.inverter(y_hat)
        # m_loss = self.mapper_loss(y_hat_hat, data) #dont demethylate the loss definations are defined for training, they log train loss
        m_l1_loss = F.l1_loss(y_hat_hat, data)
        m_l2_loss = F.mse_loss(y_hat_hat, data)

        x_hat = self.inverter(label)
        x_hat_hat = self.mapper(x_hat)
        # i_loss = self.inverter_loss(x_hat_hat, label)
        i_l1_loss = F.l1_loss(x_hat_hat, label)
        i_l2_loss = F.mse_loss(x_hat_hat, label)

        self.log("val/mapping_l1_loss", m_l1_loss, on_step=False, on_epoch=True)
        self.log("val/mapping_l2_loss", m_l2_loss, on_step=False, on_epoch=True)
        # self.log("val/mapping_loss", m_loss, on_step=False, on_epoch=True)

        self.log("val/inverting_l1_loss", i_l1_loss, on_step=False, on_epoch=True)
        self.log("val/inverting_l2_loss", i_l2_loss, on_step=False, on_epoch=True)
        # self.log("val/inverting_loss", i_loss, on_step=False, on_epoch=True)

        # pca = PCA(n_components=2)
        # pca_mox = pca.fit_transform(y_hat.cpu().numpy())
        # pca_iomox = pca.fit_transform(y_hat_hat.cpu().numpy())
        # pca_ioy = pca.fit_transform(x_hat.cpu().numpy())
        # pca_moxioy = pca.fit_transform(x_hat_hat.cpu().numpy())

        # plt.close()

        # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        # axs[0, 0].scatter(pca_mox[:, 0], pca_mox[:, 1],s=1)
        # axs[0, 0].set_title('m(X)')
        # axs[0, 1].scatter(pca_iomox[:, 0], pca_mox[:, 1],s=1)
        # axs[0, 1].set_title('i(m(X))')
        # axs[1, 0].scatter(pca_mox[:, 0], pca_mox[:, 1],s=1)
        # axs[1, 0].set_title('i(Y)')
        # axs[1, 1].scatter(pca_mox[:, 0], pca_mox[:, 1],s=1)
        # axs[1, 1].set_title('m(i(Y))')


    def configure_optimizers(self):
        lr_m = self.hparams.lr_m
        b1_m = self.hparams.b1_m
        b2_m = self.hparams.b2_m
        lr_i = self.hparams.lr_i
        b1_i = self.hparams.b1_i
        b2_i = self.hparams.b2_i

        opt_m = torch.optim.Adam(self.mapper.parameters(), lr=lr_m, betas=(b1_m, b2_m))
        opt_i = torch.optim.Adam(self.inverter.parameters(), lr=lr_i, betas=(b1_i, b2_i))
        return [opt_m, opt_i], []

    # def on_validation_epoch_end(self):
    #     # data, label = batch
    #     # y_hat = self.mapper(data)
    #     # y_hat_hat = self.inverter(y_hat)
    #     # m_loss = self.mapper_loss(y_hat_hat, data)
    #     # x_hat = self.inverter(label)
    #     # x_hat_hat = self.mapper(x_hat)

    #     # pca = PCA(n_components=2)
    #     # pca_mox = pca.fit_transform(y_hat.cpu().numpy())
    #     # pca_iomox = pca.fit_transform(y_hat_hat.cpu().numpy())
    #     # pca_ioy = pca.fit_transform(x_hat.cpu().numpy())
    #     # pca_moxioy = pca.fit_transform(x_hat_hat.cpu().numpy())

    #     # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    #     # axs[0, 0].scatter(pca_mox[:, 0], pca_mox[:, 1])
    #     # axs[0, 0].set_title('m(X)')
    #     # axs[0, 0].scatter(pca_iomox[:, 0], pca_mox[:, 1])
    #     # axs[0, 0].set_title('i(m(X))')
    #     # axs[0, 0].scatter(pca_mox[:, 0], pca_mox[:, 1])
    #     # axs[0, 0].set_title('i(Y)')
    #     # axs[0, 0].scatter(pca_mox[:, 0], pca_mox[:, 1])
    #     # axs[0, 0].set_title('m(i(Y))')
    #     print("...")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["hyperparameters"] = ( num_epochs, batch_size, Mapper_input_size, Mapper_output_size,  Mapper_learning_rate, Inverter_input_size, Inverter_output_size,  Inverter_learning_rate)

    def on_load_checkpoint(self, checkpoint):
        num_epochs, batch_size, Mapper_input_size, Mapper_output_size,  Mapper_learning_rate, Inverter_input_size, Inverter_output_size,  Inverter_learning_rate = checkpoint["hyperparameters"]