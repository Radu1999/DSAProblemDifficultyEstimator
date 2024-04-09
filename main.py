from supervised_contrastive import SupervisedContrastive
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from utils import collate_fn
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import os
import torch
torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

BATCH_SIZE = 40
LR = 1e-6
dataset = load_dataset("cosmadrian/rocode")

# Dataloaders
train_loader = DataLoader(dataset['train'], collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
val_loader = DataLoader(dataset['validation'], collate_fn=collate_fn, num_workers=16, batch_size=BATCH_SIZE)

model_name = 'dumitrescustefan/bert-base-romanian-cased-v1'
model = SupervisedContrastive(encoder_name=model_name, lr=LR)

# Logger
wandb.login(key='93443c480bfbaa0b19be76d24f2efeb6be3319fd')
wandb_logger = WandbLogger(log_model="all")
trainer = pl.Trainer(logger=wandb_logger, max_epochs=300, enable_checkpointing=False)

trainer.fit(model, train_dataloaders=train_loader)

