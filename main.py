import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import numpy as np
import os

from model import UNET

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

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

        patch_size = 96
        pcd_patches = patchify(pointcloud, (patch_size,patch_size,patch_size), step=patch_size)
        mask_patches = patchify(mask, (patch_size,patch_size,patch_size), step=patch_size)

        patch_orientation = np.array(pcd_patches.shape[:3])
        num_patches = patch_orientation.cumprod()[-1]

        return \
        [torch.tensor(pcd_patches[np.unravel_index(index, patch_orientation)], dtype=torch.float)[None,None,:] for index in range(num_patches)],\
        [torch.tensor(mask_patches[np.unravel_index(index, patch_orientation)], dtype=torch.float)[None,None,:] for index in range(num_patches)]

    def __len__(self):
        return len(self.filenames)

def data_collator(batch):
    pcd_patches = torch.cat(batch[0][0],dim=0)
    mask_patches = torch.cat(batch[0][1],dim=0)

    for i in range(1, len(batch)):
        pcd_patches = torch.cat([pcd_patches, *batch[i][0]],dim=0)
        mask_patches = torch.cat([mask_patches, *batch[i][1]],dim=0)

    return pcd_patches, mask_patches

# train_dataset = MaskDataset('data/train')
# print(train_dataset[0])
# exit()

# hyperparameters
num_epochs = 1000
batch_size = 2
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

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
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
            plt.imsave(f"vis2/mask{i}_o.png", masks[0,:,:,i].cpu(), cmap=plt.cm.gray)
            plt.imsave(f"vis2/mask{i}_p.png", outputs[0,:,:,i].detach().cpu(), cmap=plt.cm.gray)  

        # loss_history = loss_history[-5000:]
        plt.plot(loss_history)
        plt.show()
                
            
