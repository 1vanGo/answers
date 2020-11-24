"""Train simple GhostNet model."""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from adabelief_pytorch import AdaBelief
from albumentations.pytorch import ToTensor
from albumentations import Compose, Resize
from utils import sigmoid, get_device, mono_to_color, mixup


class TrainDataset(Dataset):
    """Dataset for train speakers."""
    def __init__(self, root_dir, speakers=[], transform=None):
        self.root_dir = root_dir
        self.speakers = speakers
        self.transform = transform

        # get all filenames, left only "speakers"
        self.files = list(glob.iglob(f'{root_dir}/**/*.npy', recursive=True))
        if len(speakers):
            self.files = [filename for filename in self.files if self.check_speaker(filename, self.speakers)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        class_map = {'clean': 0, 'noisy': 1}

        img = np.load(self.files[idx]).astype(np.float32)
        img = mono_to_color(img)
        label = class_map[self.files[idx].split('/')[1]]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        return img, label

    @staticmethod
    def check_speaker(filename, speakers):
        """Check speaker's id."""
        return filename.split('/')[2] in set(speakers)


class TestDataset(Dataset):
    """Dataset for valid/test speakers."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [filename for filename in glob.iglob(f'{root_dir}/**/*.npy', recursive=True)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx]).astype(np.float32)
        img = mono_to_color(img)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img


def get_valid_speakers():
    """Separate part of train speakers to validation sample."""
    np.random.seed(0)
    speakers = []

    for _, sub, _ in os.walk('train'):
        speakers.extend(sub)
    speakers = sorted(list(set(speakers) - set(['noisy', 'clean'])))
    valid_speakers = np.random.choice(speakers, 200, replace=False)
    train_speakers = sorted(list(set(speakers) - set(valid_speakers)))
    return train_speakers, valid_speakers


def get_valid_targets(valid_dataset):
    """Get valid targets from dataset."""
    yvalid = pd.Series(valid_dataset.files).apply(lambda s: s.split('/')[1])
    return yvalid.map({'clean': 0, 'noisy': 1})


def main():
    """Model training."""
    train_speakers, valid_speakers = get_valid_speakers()

    # define transforms for train & validation samples
    train_transform = Compose([Resize(760, 80), ToTensor()])

    # define datasets & loaders
    train_dataset = TrainDataset('train', train_speakers, transform=train_transform)
    valid_dataset = TrainDataset('train', valid_speakers, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    device = get_device()
    print(f'Selected device: {device}')

    model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    model.classifier = nn.Linear(in_features=1280, out_features=1, bias=True)

    net = model
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdaBelief(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, eps=1e-4, verbose=True)

    # prepare valid target
    yvalid = get_valid_targets(valid_dataset)

    # training loop
    for epoch in range(10):
        loss_log = {'train': [], 'valid': []}
        train_loss = []

        net.train()
        for x, y in tqdm(train_loader):
            x, y = mixup(x, y, alpha=0.2)
            x, y = x.to(device), y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = net(x)

            loss = criterion(outputs, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # save loss
            train_loss.append(loss.item())

        # evaluate
        net.eval()
        valid_pred = torch.Tensor([]).to(device)

        for x, y in valid_loader:
            with torch.no_grad():
                x, y = x.to(device), y.to(device, dtype=torch.float32)
                ypred = net(x)
                valid_pred = torch.cat([valid_pred, ypred], 0)

        valid_pred = sigmoid(valid_pred.cpu().numpy())
        val_loss = log_loss(yvalid, valid_pred, eps=1e-7)
        val_acc = (yvalid == (valid_pred > 0.5).astype(int).flatten()).mean()
        tqdm.write(f'Epoch {epoch} train_loss={np.mean(train_loss):.4f}; val_loss={val_loss:.4f}; val_acc={val_acc:.4f}')

        loss_log['train'].append(np.mean(train_loss))
        loss_log['valid'].append(val_loss)
        scheduler.step(loss_log['valid'][-1])

    torch.save(net.state_dict(), 'ghostnet_model.pt')
    print('Training is complete.')


if __name__ == '__main__':
    main()
