import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

import models.utils as utils
import models.datasets as datasets
import models.model as models

def train(backbone, n_epochs, batch_size, rnn_hidden_dim, lr,
          model_save_path):
    
    utils.all_seed(1234)

    #Read dataset
    text_path = 'annotations'
    img_path = 'images'
    path = './OCR/training_data/'

    dataset = datasets.ImageFolders(path, text_path, img_path, 
                                    transform=datasets.compose)
    
    train_size = int(0.9*len(dataset))
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

    model = models.Model((64, 768), backbone, 1, rnn_hidden_dim, numChars=len(char2idx))
    optimizer = optim.Adam(model.parameters(), 
                           lr=lr, 
                           weight_decay=weight_decay)
    
    loss_fn = nn.CTCLoss()
    model.to(device)

    model_save_name = f"{backbone}_{rnn_hidden_dim}"
    PATH = os.path.join(model_save_path, f"{model_save_name}.pth")

    if os.path.isfile(PATH):
        answer = input("Would you like to load previous model - Y/n: ")
        if answer[0].lower() == 'y':
            model.load_state_dict(torch.load(PATH))
    

    min_val_loss = None
    
    for epoch in range(n_epochs):

        total_train_loss = 0
        model.train()

        for img, text in train_loader:
            
            print(f'Epoch {epoch}/{n_epochs + 1}', end=' ')

            optimizer.zero_grad()

            logits = model(img.to(device))
            loss = utils.compute_loss(loss_fn, text, logits, device, char2idx)            
            loss.backward()
            total_train_loss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        print(f'Avg train loss: {total_train_loss/len(train_loader)}', end=' ')

        model.eval()

        total_valid_loss = 0

        with torch.no_grad():
            for img, text in valid_loader:
                logits = model(img.to(device))
                loss = utils.compute_loss(loss_fn, text, logits, device, char2idx)
                total_valid_loss += loss.item()

            avg_valid_loss = total_valid_loss / len(valid_loader)
            print(f'Avg train loss: {total_train_loss/len(train_loader)}', end=' ')

            if min_val_loss < avg_valid_loss:
                min_val_loss = avg_valid_loss
                torch.save(model.state_dict(), PATH)


