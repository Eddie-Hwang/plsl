import pytorch_lightning as pl
import torch
import torch.nn as nn
from modules.helpers import *


# Create your own model
class DummyModel(pl.LightningModule):
    def __init__(self, lr, monitor, **kwargs):
        super().__init__()

        self.lr = lr
        self.monitor = monitor
        self.model = nn.Identity()

    def forward(self, *args, **kwargs):
        return 
    
    def training_step(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        return opt
    