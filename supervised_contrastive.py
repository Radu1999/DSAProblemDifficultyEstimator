import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
from itertools import combinations
import wandb

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    https://github.com/adambielski/siamese-triplet/tree/0c719f9e8f59fa386e8c59d10b2ddde9fac46276
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class FunctionNegativeTripletSelector:
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets, anchor_positive, anchor_negative, negative_indices = [], [], [], []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

class AllTripletSelector:
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

class SupervisedContrastive(pl.LightningModule):
    def __init__(self, encoder_name, margin=70, lr=1e-7):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=encoder_name)
        triplet_selector = FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=random_hard_negative)
        #triplet_selector = AllTripletSelector()
        self.loss = OnlineTripletLoss(triplet_selector=triplet_selector, margin=margin)
        self.lr = lr

    def encode(self, texts):
        input_ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True,
                                   max_length=512)[
            'input_ids'].to(self.encoder.device)
        outputs = self.encoder(input_ids)
        return torch.mean(outputs.last_hidden_state, dim=1)

    def compute_loss(self, x, labels):
        return self.loss(x, labels)

    def training_step(self, batch):
        inp, labels = batch
        labels = torch.tensor(labels).to(self.encoder.device)
        x = self.encode(inp)
        return self.compute_loss(x, labels)

    def training_step(self, batch):
        inp, labels = batch
        labels = torch.tensor(labels).to(self.encoder.device)
        x = self.encode(inp)
        loss = self.compute_loss(x, labels)[0]
        self.log("train_loss", loss.item(), on_step=True, logger=True)
        return loss

    def validation_step(self, batch):
        inp, labels = batch
        labels = torch.tensor(labels).to(self.encoder.device)
        x = self.encode(inp)
        loss = self.compute_loss(x, labels)[0]
        self.log("validation_loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
