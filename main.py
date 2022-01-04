import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os

from model import UNET

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        for _, _, filenames in os.walk(f'{root_dir}/pointclouds'):
            self.filenames = filenames
        
    def __getitem__(self, index):
        pointcloud = np.load(f'{self.root_dir}/pointclouds/{self.filenames[index]}')
        mask = np.load(f'{self.root_dir}/masks/{self.filenames[index]}') 
        
        noise = np.random.randint(0, pointcloud.shape[0], (np.random.randint(0, 1000), 3))
        pointcloud[noise[:,0], noise[:,1], noise[:,2]] = 1

        return (
            torch.tensor(pointcloud, dtype=torch.float)[None,:],
            torch.tensor(mask, dtype=torch.float)[None,:],
        )


    def __len__(self):
        return len(self.filenames)


# train_dataset = MaskDataset('data/train')
# print(train_dataset[0])
# exit()

# hyperparameters
num_epochs = 1000
batch_size = 16
learning_rate = 0.0001

model = UNET(1, 1, features=[32, 64, 128, 256, 512]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

try:
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
except:
    pass

# dataset
train_dataset = MaskDataset('./data/train')
# test_dataset = MaskDataset('./data/test')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

loss_history = []
rolling_loss = 0

# training loop
for epoch in range(num_epochs):
    for i, (pointclouds, masks) in enumerate(train_loader):
        pointclouds = pointclouds.to(device)
        masks = masks.to(device)

        outputs = model(pointclouds)
        loss = criterion(outputs, masks)
        
        rolling_loss = 0.975*rolling_loss + 0.025*loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(rolling_loss)

        if (i+1)%1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {rolling_loss:.4f}')
    
    if epoch % 20 == 0:
        if rolling_loss == min(loss_history):
            torch.save(model, f'model{epoch}.pth')
        
        for i in range(masks.shape[-1]):
            plt.imsave(f"vis/mask{i}_o.png", masks[0,:,:,i].cpu(), cmap=plt.cm.gray)
            plt.imsave(f"vis/mask{i}_p.png", outputs[0,:,:,i].detach().cpu(), cmap=plt.cm.gray)  

        # loss_history = loss_history[-5000:]
        plt.plot(loss_history)
        plt.show()
                
            
