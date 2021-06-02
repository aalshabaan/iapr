import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_float

from sklearn.preprocessing import LabelEncoder

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

class all_classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Conv2d(1, 32, 5, padding=2),
                            nn.MaxPool2d(2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),

                            nn.Dropout(0.3),
                            nn.Conv2d(32, 64, 5, padding=2),
                            nn.MaxPool2d(2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),

                            nn.Dropout(0.3),
                            nn.Conv2d(64, 128, 5, padding=2),
                            nn.MaxPool2d(2),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),

                            nn.Dropout(0.3),
                            nn.Conv2d(128, 256, 5, padding=2),
                            nn.MaxPool2d(2),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                            nn.Flatten())

        self.symbol_classifier = nn.Sequential(
            nn.Linear(6 * 6 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 13)
        )

        self.suit_classifier = nn.Sequential(
            nn.Linear(6 * 6 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        f = self.features(x)
        #print(f.shape)
        symbol = self.symbol_classifier(f)
        suit = self.suit_classifier(f)
        #print(f'Symbol {symbol.shape}, Suit {suit.shape}')
        return symbol, suit





def train_new_classifier(folder, verbose=False, save=False):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if verbose:
        print(f'Current device: {device}')

    model = all_classifier().to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-3)
    suit_loss = nn.CrossEntropyLoss()
    symbol_loss = nn.CrossEntropyLoss()
    labels = produce_labels()

    if verbose:
        print('Creating Tensors')

    images, symbols, suits, symbol_encoder, suit_encoder = get_training_tensors(folder, labels)
    images = images.to(device)
    symbols = symbols.to(device)
    suits = suits.to(device)
    data = DataLoader(TensorDataset(images, symbols, suits), batch_size=13)

    if verbose:
        print('Starting training')

    for epoch in range(100):
        optimizer.zero_grad()
        sym_acc = 0
        suit_acc = 0
        for x, y, z in data:

            symbol_out, suit_out = model(x)
            loss = symbol_loss(symbol_out, y) + suit_loss(suit_out, z)
            sym_acc += (symbol_out.argmax(1) == y).sum().item()
            suit_acc += (suit_out.argmax(1) == z).sum().item()
            loss.backward()
            optimizer.step()

        sym_acc /= symbols.shape[0]
        suit_acc /= suits.shape[0]
        if verbose:
            print(f'Epoch {epoch}, loss = {loss.item()}, symbol_acc = {sym_acc}, suit_acc = {suit_acc}')

    model.eval()
    if save:
        torch.save(model, 'all_classifier.pkl')
    return model






def get_training_tensors(folder, labels):
    files = [f'{folder}/{x}' for x in os.listdir(folder) if 'p' in x]
    images = []
    symbols = []
    suits = []
    for f in files:
        player_id = f'P{f.split("p")[1][0]}'
        game_id = int(f.split('game')[2][0])
        round_id = int(f.split('_')[3])
        row_id = np.logical_and(labels['round'] == round_id, labels['game'] == game_id)
        images.append(resize(img_as_float(imread(f)), output_shape=(100, 100)))
        label = labels.loc[row_id, player_id].iloc[0]
        #print(label)
        symbols.append(label[0])
        suits.append(label[1])

    images = torch.Tensor(images).unsqueeze(1)
    symbol_encoder = LabelEncoder()
    symbols = torch.from_numpy(symbol_encoder.fit_transform(symbols))
    suit_encoder = LabelEncoder()
    suits = torch.from_numpy(suit_encoder.fit_transform(suits))

    return images, symbols, suits, symbol_encoder, suit_encoder



def produce_labels():
    label_root = './train_games'
    labels = pd.read_csv(f'{label_root}/game1/game1.csv')
    labels['game'] = 1

    for game in range(2, 8):
        temp = pd.read_csv(f'{label_root}/game{game}/game{game}.csv')
        temp['game'] = game
        labels = labels.append(temp, ignore_index=True)
    return labels