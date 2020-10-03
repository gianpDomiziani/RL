import torch
import torch.nn as nn
import numpy as np

from tensorboardX import SummaryWriter

import argparse


LEARNING_RATE = 0.003
class FCNNModel(nn.Module):

    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super(FCNNModel, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipe(x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda computation")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")


    netBasic = FCNNModel(num_inputs=10, num_classes=3)
    optimizer = torch.optim.Adam(params=netBasic.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    objective = nn.BCELoss()

    writer = SummaryWriter()

    losses = []

