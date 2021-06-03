import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST

import pickle
import numpy as np
import os
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_float

from sklearn.preprocessing import LabelEncoder


class Classifier(nn.Module):

    def __init__(self):
        """
        A card image classifier that predicts both the digit and the suit using PyTorch
        """
        super().__init__()
        self.features = nn.Sequential(
                            #nn.Dropout(0.3),
                            nn.Conv2d(1, 32, 5, padding=2),
                            nn.MaxPool2d(2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),

                            #nn.Dropout(0.2),
                            nn.Conv2d(32, 64, 5, padding=2),
                            nn.MaxPool2d(2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),

                            #nn.Dropout(0.2),
                            nn.Conv2d(64, 128, 5, padding=2),
                            nn.MaxPool2d(2),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),

                            #nn.Dropout(0.2),
                            nn.Conv2d(128, 256, 5, padding=2),
                            nn.MaxPool2d(2),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                            nn.Flatten())

        self.symbol_classifier = nn.Sequential(
            nn.Linear(6 * 6 * 256, 128),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 13)
        )

        self.suit_classifier = nn.Sequential(
            nn.Linear(6 * 6 * 256, 128),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            #nn.Dropout(0.3),
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


def train_new_classifier(folder:str, verbose=False, save=False, validate=True, validation_game=7, mnist_samples=None):
    """
    Trains a model of the class Classifier defined above, optionally adding some MNIST digits returns it with its training history and label encoders. Can be used to save the model or validate it as well
    :param folder: Path to the folder containing the training images, those are binary masks of individual cards
    :param verbose: Whether to print the training progress to the console
    :param save: Whether to save the model's state_dict to a file
    :param validate: Whether to validate the model, if True, game number validation_game will be excluded from training and used for evaluation
    :param validation_game: Number of the game used for validation, has no effect if validate=False or if is not in range(1,8)
    :param mnist_samples: The number of MNIST digits to add. None means none will be added
    :return: model:Classifier: The trained classifier
             hist:dict: The model's training history, has lists of losses and accuracies
             encoders:dict A dictionary of LabelEncoders used to train this model, used to reverse the numerical labels to ones similar to what's in the training labels

    """

    validate = validate and validation_game in range(1,8)  # Check if the validation game index is correct

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if verbose:
        print(f'Current device: {device}')
        print(f'Validating? {validate}')

    if mnist_samples:
        if verbose:
            print(f'Adding {mnist_samples} samples from the MNIST dataset')
        mnist = MNIST('data/MNIST', train=True, download=True)
        idx = torch.randint(0, mnist.data.shape[0], (mnist_samples,))  # A random sample of MNIST images
        mnist_images = torch.Tensor([resize(img_as_float(x), (100,100)) for x in mnist.data[idx]]).to(device)  # Resizing the images and moving them to GPU if available
        mnist_images = mnist_images.unsqueeze(1)
        mnist_symbols = mnist.targets[idx].to(device).long()
        mnist_suits = torch.zeros_like(mnist_symbols).to(device).long()  # Dummy label, causes problems if too many MNIST images

    model = Classifier().to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-3)
    suit_loss = nn.CrossEntropyLoss()
    symbol_loss = nn.CrossEntropyLoss()

    labels = produce_labels() #  Load the label Dataframe
    hist = {'loss':[], 'sym_acc':[], 'suit_acc':[], 'val_loss':[], 'val_sym_acc':[], 'val_suit_acc':[]}

    if verbose:
        print('Creating Tensors')

    if validate:
        images, symbols, suits, images_val, symbols_val, suits_val, symbol_encoder, suit_encoder = get_training_tensors(folder, labels, validation_game)
        images = images.to(device)
        symbols = symbols.to(device)
        suits = suits.to(device)
        images_val = images_val.to(device)
        symbols_val = symbols_val.to(device)
        suits_val = suits_val.to(device)

        if mnist_samples:
            #  Add MNIST images to the dataset
            images = torch.cat((images, mnist_images), dim=0)
            symbols = torch.cat((symbols, mnist_symbols), dim=0)
            suits = torch.cat((suits, mnist_suits), dim=0)
        train_dataset = TensorDataset(images, symbols, suits)
        validation_dataset = TensorDataset(images_val, symbols_val, suits_val)
        validation_data = DataLoader(validation_dataset, batch_size=13)

    else:
        images, symbols, suits, symbol_encoder, suit_encoder = get_training_tensors(folder, labels, validation_game=None)
        images = images.to(device)
        symbols = symbols.to(device)
        suits = suits.to(device)
        train_dataset = TensorDataset(images, symbols, suits)

    data = DataLoader(train_dataset, batch_size=13, shuffle=True)
    encoders = {'sym':symbol_encoder, 'suit':suit_encoder}

    if verbose:
        print('Starting training')

    for epoch in range(75):
        model.train()
        optimizer.zero_grad()
        sym_acc = 0
        suit_acc = 0
        epoch_loss = 0
        for x, y, z in data:

            symbol_out, suit_out = model(x)
            loss = 13/17 * symbol_loss(symbol_out, y) + 4/17 * suit_loss(suit_out, z)
            sym_acc += (symbol_out.argmax(1) == y).sum().item()
            suit_acc += (suit_out.argmax(1) == z).sum().item()
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()


        sym_acc /= len(train_dataset)
        suit_acc /= len(train_dataset)
        epoch_loss /= len(train_dataset)

        hist['sym_acc'].append(sym_acc)
        hist['suit_acc'].append(suit_acc)
        hist['loss'].append(epoch_loss)

        if validate:
            model.eval()
            sym_acc_val = 0
            suit_acc_val = 0
            epoch_loss_val = 0
            for x, y, z in validation_data:

                symbol_out, suit_out = model(x)
                loss_val = 13/17 * symbol_loss(symbol_out, y) + 4/17 * suit_loss(suit_out, z)
                sym_acc_val += (symbol_out.argmax(1) == y).sum().item()
                suit_acc_val += (suit_out.argmax(1) == z).sum().item()
                epoch_loss_val += loss_val.item()

            sym_acc_val /= len(validation_dataset)
            suit_acc_val /= len(validation_dataset)
            epoch_loss_val /= len(validation_dataset)

            hist['val_sym_acc'].append(sym_acc_val)
            hist['val_suit_acc'].append(suit_acc_val)
            hist['val_loss'].append(epoch_loss_val)

        if verbose:
            if validate:
                print(f'Epoch {epoch}, loss = {epoch_loss:.2f}, symbol train accuracy = {100*sym_acc:.2f}%, suit train accuracy = {100*suit_acc:.2f}%'
                      f' validation loss = {epoch_loss_val:.2f}, symbol val accuracy = {100*sym_acc_val:.2f}%, suit val accuracy = {100*suit_acc_val:.2f}%')
            else:
                print(f'Epoch {epoch:}, loss = {epoch_loss:.2f}, symbol train accuracy = {100*sym_acc:.2f}%, suit train accuracy = {100*suit_acc:.2f}%')

    model.eval()
    if save:
        print(f'Saving Model to {os.getcwd()+os.path.sep}classifier.torch')
        torch.save(model.state_dict(), 'classifier.torch')
        print(f'Saving encoders to {os.getcwd()+os.path.sep}encoders.pkl')
        with open('./encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
    return model, hist, encoders


def get_training_tensors(folder:str, labels:pd.DataFrame, validation_game=7):
    """
    Creates PyTorch Tensors from the images present in the folder `folder` and the label DataFrame, can return one set of tensors for training or a training and validation set
    :param folder: Path to the folder containing the training images
    :param labels: DataFrame containing the compiled labels
    :param validation_game: Index of the validation game, has to be between 1 and 7, otherwise all games are used for training
    :return: Tensors in the order (images, symbols, suits, images_val, symbols_val, suits_val) and 2 encoders (symbol_encoder, label_encoder)
    """
    files = [f'{folder}/{x}' for x in os.listdir(folder) if 'p' in x]
    images = []
    symbols = []
    suits = []

    images_val = []
    symbols_val = []
    suits_val = []

    for f in files:
        player_id = f'P{f.split("p")[1][0]}'
        game_id = int(f.split('game')[2][0])
        round_id = int(f.split('_')[3])
        row_id = np.logical_and(labels['round'] == round_id, labels['game'] == game_id)

        if game_id == validation_game:
            images_val.append(resize(img_as_float(imread(f)), output_shape=(100, 100)))
            label = labels.loc[row_id, player_id].iloc[0]  # loc returns a series because we are indexing rows by a boolean mask, so iloc[0] returns the value inside
            # print(label)
            symbols_val.append(label[0])
            suits_val.append(label[1])

        else:
            images.append(resize(img_as_float(imread(f)), output_shape=(100, 100)))
            label = labels.loc[row_id, player_id].iloc[0]
            #print(label)
            symbols.append(label[0])
            suits.append(label[1])

    images = torch.Tensor(images).unsqueeze(1)
    # Transform labels from characters ('0', 'Q', etc.) to numbers
    symbol_encoder = LabelEncoder()
    symbols = torch.from_numpy(symbol_encoder.fit_transform(symbols)).long()
    suit_encoder = LabelEncoder()
    suits = torch.from_numpy(suit_encoder.fit_transform(suits)).long()

    if validation_game in range(1,8):
        images_val = torch.Tensor(images_val).unsqueeze(1)
        symbols_val = torch.from_numpy(symbol_encoder.transform(symbols_val)).long()
        suits_val = torch.from_numpy(suit_encoder.transform(suits_val)).long()
        return images, symbols, suits, images_val, symbols_val, suits_val, symbol_encoder, suit_encoder
    else:
        return images, symbols, suits, symbol_encoder, suit_encoder


def produce_labels():
    """
    Concatenate all label DataFrames into one
    :return: The compiled label DataFrame
    """
    label_root = './train_games'
    labels = pd.read_csv(f'{label_root}/game1/game1.csv')
    labels['game'] = 1

    for game in range(2, 8):
        temp = pd.read_csv(f'{label_root}/game{game}/game{game}.csv')
        temp['game'] = game
        labels = labels.append(temp, ignore_index=True)
    return labels


def load_trained_model():
    """
    Loads a pretrained model from the file 'classifier.torch'
    :return: The loaded model
    """

    model = Classifier()
    model.load_state_dict(torch.load('./classifier.torch'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    with open('./encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders


def predict_round(model:Classifier, encoders:dict, images:np.ndarray):
    """
    Predicts the values of the 4 cards present in this round
    :param model: The classifier model to do the predictions
    :param encoders: The label encoders that'll be used to inverse transform the labels
    :param images: list of 4 card masks, input to the model
    :return: list of strings ['QH', '6S', etc..]
    """
    device = next(model.parameters()).device
    input_ = torch.Tensor([resize(img_as_float(im), (100,100)) for im in images]).to(device).unsqueeze(1)
    symbol_out, suit_out = model(input_)
    symbol_pred = symbol_out.argmax(1)
    suit_pred = suit_out.argmax(1)

    symbol_decoded = encoders['sym'].inverse_transform(symbol_pred.cpu().numpy())
    suit_decoded = encoders['suit'].inverse_transform(suit_pred.cpu().numpy())

    out = [f'{x}{y}' for x,y in zip(symbol_decoded, suit_decoded)]
    return out

