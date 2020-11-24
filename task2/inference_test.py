"""Test the model."""
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensor
from albumentations import Compose, Resize
from utils import sigmoid, get_device, mono_to_color


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


def get_valid_targets(valid_dataset):
    """Get valid targets from dataset."""
    yvalid = pd.Series(valid_dataset.files).apply(lambda s: s.split('/')[1])
    return yvalid.map({'clean': 0, 'noisy': 1})


def load_net(net, path):
    """Load pytorch model."""
    device = get_device()
    net.load_state_dict(torch.load(path, map_location=device))
    return net


def main():
    """Model testing."""

    test_transform = Compose([Resize(760, 80), ToTensor()])
    test_dataset = TrainDataset('val', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = get_device()
    print(f'Selected device: {device}')

    # load net
    model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    model.classifier = nn.Linear(in_features=1280, out_features=1, bias=True)
    net = load_net(model, 'ghostnet_model.pt')
    net.to(device)
    net.eval()

    ytest = get_valid_targets(test_dataset)
    test_pred = torch.Tensor([]).to(device)

    for x, _ in tqdm(test_loader):
        with torch.no_grad():
            x = x.to(device)
            ypred = net(x)
            test_pred = torch.cat([test_pred, ypred], 0)

    test_pred = sigmoid(test_pred.cpu().numpy())
    test_acc = (ytest == (test_pred > 0.5).astype(int).flatten()).mean()
    print('Testing is complete.')
    print(f'Test accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    main()
