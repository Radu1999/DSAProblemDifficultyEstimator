from supervised_contrastive import SupervisedContrastive
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from utils import collate_fn
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import torch
torch.set_float32_matmul_precision('medium')

BATCH_SIZE = 16
dataset = load_dataset("cosmadrian/rocode")

# Dataloaders
train_loader = DataLoader(dataset['train'], collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset['validation'], collate_fn=collate_fn, batch_size=BATCH_SIZE)

model_name = 'dumitrescustefan/bert-base-romanian-cased-v1'
model = SupervisedContrastive(encoder_name=model_name)

# Logger
wandb.login(key='')
wandb_logger = WandbLogger(log_model="all")
trainer = pl.Trainer(logger=wandb_logger)

trainer.fit(model, train_dataloaders=train_loader)

