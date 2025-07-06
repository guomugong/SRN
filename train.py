from model.unet_model import UNet
from utils.dataset import FundusSeg_Loader
from torch import optim
import torch.nn as nn
import random
import torch
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="0"

dataset_name=sys.argv[1]
run_num=int(sys.argv[2])

if dataset_name == "drive":
    train_data_path = "/data/95/dataset/drive/train/"
    valid_data_path = "/data/95/dataset/drive/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.4969, 0.2702, 0.1620]
    dataset_std=[0.3479,0.1896,0.1075]
    early_epoch = 400

if dataset_name == "stare":
    train_data_path = "/data/95/dataset/stare/train/"
    valid_data_path = "/data/95/dataset/stare/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.5889, 0.3272, 0.1074]
    dataset_std=[0.3458,0.1844,0.1104]
    early_epoch = 400

if dataset_name == "chase":
    train_data_path = "/data/95/dataset/chase_db1/train/"
    valid_data_path = "/data/95/dataset/chase_db1/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.4416, 0.1606, 0.0277]
    dataset_std=[0.3530,0.1407,0.0366]
    early_epoch = 400

if dataset_name == "rimone":
    train_data_path = "/data/95/dataset/oc/rimone/train/"
    valid_data_path = "/data/95/dataset/oc/rimone/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.0001
    batch_size = 8
    test_epoch = 2 
    dataset_mean = [0.3383, 0.1164, 0.0465] # In use
    dataset_std = [0.1849, 0.0913, 0.0441]
    early_epoch = 400

if dataset_name == "refuge":
    train_data_path = "/data/95/dataset/oc/refuge/train/"
    valid_data_path = "/data/95/dataset/oc/refuge/train_valid/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.0001
    batch_size = 8
    test_epoch = 2 
    dataset_mean = [0.4237, 0.2414, 0.1182] # In Use
    dataset_std  = [0.1996, 0.1206, 0.0712]
    early_epoch = 400

if dataset_name == "refuge2":
    train_data_path = "/data/95/dataset/oc/refuge/valid_train/"
    valid_data_path = "/data/95/dataset/oc/refuge/valid_test/"
    N_epochs = 2500
    lr_decay_step = [2000]
    lr_init = 0.0001
    batch_size = 8
    test_epoch = 2 
    dataset_mean = [0.5984, 0.4048, 0.3161] # In Use
    dataset_std  = [0.2416, 0.1871, 0.1442]
    early_epoch = 400

def train_net(net, device, run_num, epochs=N_epochs, batch_size=batch_size, lr=lr_init):
    train_dataset = FundusSeg_Loader(train_data_path, 1, dataset_name, dataset_mean, dataset_std)
    valid_dataset = FundusSeg_Loader(valid_data_path, 0, dataset_name, dataset_mean, dataset_std)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=6, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    print('Train images: %s' % len(train_loader.dataset))
    print('Valid images: %s' % len(valid_loader.dataset))

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=lr_decay_step,gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    best_epoch = 10
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        net.train()
        train_loss = 0
        for i, (image, label, filename, raw_height, raw_width) in enumerate(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

        # Validation
        # epoch != test_epoch
        if ((epoch+1) % test_epoch == 0):
            net.eval()
            val_loss = 0
            for i, (image, label, filename, raw_height, raw_width) in enumerate(valid_loader):
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                val_loss = val_loss + loss.item()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), './snapshot/'+dataset_name+'_b'+str(run_num)+'.pth')
                print('saving model............................................')
                best_epoch = epoch
            if (epoch - best_epoch) > early_epoch:
                print('Early Stopping ............................................')
                exit()
        
            print('Loss/valid', val_loss / i)
            sys.stdout.flush()

        scheduler.step()

if __name__ == "__main__":
    random.seed(run_num) 
    np.random.seed(run_num)
    torch.manual_seed(run_num)
    torch.cuda.manual_seed(run_num)
    torch.cuda.manual_seed_all(run_num)
    device = torch.device('cuda')
    net = UNet(n_channels=3, n_classes=1)
    net.to(device=device)
    train_net(net, device, run_num)
