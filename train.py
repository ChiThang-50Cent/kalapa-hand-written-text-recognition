import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

import models.utils as utils
import models.datasets as datasets
import models.model as models

parser = argparse.ArgumentParser()

parser.add_argument('--path', required=True, help='path to root folder')
parser.add_argument('--text_path', default='annotations', help='path to dataset')
parser.add_argument('--img_path', default='binary', help='path to dataset')
parser.add_argument('--backbone', default='vgg16', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=768, help='the width of the input image to network')
parser.add_argument('--hidden_dim', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--model_save_path', default='/content', help='reproduce experiemnt')

opt = parser.parse_args()

def train(path=opt.path, text_path=opt.text_path, img_path=opt.img_path, 
          backbone=opt.backbone, n_epochs=opt.n_epoch, batch_size=opt.batch_size, 
          rnn_hidden_dim=opt.hidden_dim, lr=opt.lr, imgH=opt.imgH, imgW=opt.imgW,
          model_save_path=opt.model_save_path, seed = opt.manualSeed):
    
    utils.all_seed(seed)

    #Read dataset
    dataset = datasets.ImageFolders(path, img_path, text_path=text_path, 
                                    transform=datasets.compose)
    
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size

    train_set, valid_set = random_split(dataset, [train_size, valid_size])
    print(f'Train size: {train_size}, valid size: {valid_size}')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    all_chars = utils.get_all_char(
        utils.get_img_name_and_labels(f'{path}{text_path}')[:,-1])
    
    char2idx = utils.char2idx(all_chars)
    idx2char = utils.idx2char(all_chars)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Run on: {device}')

    weight_decay = 1e-3
    clip_norm = 1

    model = models.Model((imgH, imgW), backbone, 1, 
                         rnn_hidden_dim, numChars=len(char2idx))
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=lr, 
                           weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    loss_fn = nn.CTCLoss()
    model.to(device)

    model_save_name = f"{backbone}_{rnn_hidden_dim}"
    PATH = os.path.join(model_save_path, f"{model_save_name}.pth")

    if os.path.isfile(PATH):
        answer = input("Would you like to load previous model - Y/n: ")
        if answer[0].lower() == 'y':
            model.load_state_dict(torch.load(PATH))
    
    min_val_loss = 100_000
    
    print('Start Training....')

    for epoch in range(n_epochs):

        total_train_loss = 0
        model.train()

        print(f'Epoch {epoch + 1}/{n_epochs}')
        
        for i, (img, text) in enumerate(train_loader):
            
            if i % 10 == 0: print(f'{i}', end=', ')
            optimizer.zero_grad()

            logits = model(img.to(device))
            loss = utils.compute_loss(loss_fn, text, logits, device, char2idx)            
            loss.backward()
            total_train_loss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        print(f'Avg train loss: {total_train_loss/len(train_loader):.5f}', end=' ')

        model.eval()
        total_valid_loss = 0

        with torch.no_grad():
            for img, text in valid_loader:
                logits = model(img.to(device))
                loss = utils.compute_loss(loss_fn, text, logits, device, char2idx)
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        print(f'Avg valid loss: {avg_valid_loss:.5f}')

        scheduler.step()

        if min_val_loss > avg_valid_loss:
            min_val_loss = avg_valid_loss
            torch.save(model.state_dict(), PATH)
        

if __name__ == '__main__':
    train()