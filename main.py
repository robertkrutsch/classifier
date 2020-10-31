# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from PIL import Image
from generate_dataset import GenData
from data_loader import Dataset
from network import Network
from testing import NetworkTool
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def plot_stats(v, t):
    fig, ax = plt.subplots()
    ax.plot(v, 'r')
    ax.plot(t, 'b')
    ax.set(xlabel='Epochs', ylabel='r- validation loss/b - training loss',
           title='Loss function.')
    ax.grid()
    # fig.savefig("loss.png")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train_dataset = Dataset(csv_file='/content/dataset/train.csv',
                            root_dir='/content/dataset/train')

    valid_dataset = Dataset(csv_file='/content/dataset/valid.csv',
                            root_dir='/content/dataset/valid')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=2)

    net = Network()
    tool = NetworkTool(path='/content/dataset/networks/')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    '''
    #betas - this are the values provided in the Adam paper
    #eps - 1e-4 to 1e-8 is suggested in the paper
    #weight decay - it cannot be too much as then we prioratize small weights to the goal, fastai puts 0.01
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loss = []
    valid_loss = []
    # Loop over epochs
    for epoch in range(30):
        # Training
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            local_labels = data['labels']
            local_batch = data['image'].float()
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            net.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(local_batch)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 0:
                print('TRAINING: epoch: %d loss: %.3f' % (epoch + 1, running_loss / (i + 1)))

        train_loss.append(running_loss / float(i + 1))
        if epoch % 2 == 0:
            tool.save_checkpoint(net, epoch, loss, optimizer)

        # validation
        running_loss = 0.0
        for i, data in enumerate(valid_loader, 0):
            local_labels = data['labels']
            local_batch = data['image'].float()
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            net.to(device)

            outputs = net(local_batch)
            loss = criterion(outputs, local_labels)
            running_loss += loss.item()
        valid_loss.append(running_loss / float(i + 1))
        print('VALIDATION: epoch: %d loss: %.3f' % (epoch + 1, running_loss / (i + 1)))

    plot_stats(valid_loss, train_loss)

    #tool.run_checkpoint(0)

'''
TBD :
Done: - optimizer setup, what optimizer to use , wieght decay, lr decay 
Done: - how to initialize with something 
Done: - switch to gpu
Done: - save and load a network 
Done: - write some testing script
- Comment the code and put it on github
- a way to iterate through networks , maybe something like in NAS where ya generate the network from a config 
'''
