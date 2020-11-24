"""Test the model."""
import argparse
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


def load_net(net, path):
    """Load pytorch model."""
    device = get_device()
    net.load_state_dict(torch.load(path, map_location=device))
    return net


def main():
    """Model testing on single point."""

    # Parse arguments
    parser = argparse.ArgumentParser(description='File')
    parser.add_argument('--filename', action='store', dest='filename')
    args = parser.parse_args()

    data = np.load(args.filename).astype(np.float32)
    data = mono_to_color(data)

    test_transform = Compose([Resize(760, 80), ToTensor()])
    data = test_transform(**{'image': data})['image']

    device = get_device()
    print(f'Selected device: {device}')

    # load net
    model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    model.classifier = nn.Linear(in_features=1280, out_features=1, bias=True)
    net = load_net(model, 'ghostnet_model.pt')
    net.to(device)
    net.eval()

    with torch.no_grad():
        x = data.to(device)
        x = x.unsqueeze(0)
        pred = net(x)

    pred = sigmoid(pred.cpu().numpy())
    print(f'Is noisy with probability: {pred[0][0]:.4f}')


if __name__ == '__main__':
    main()
