import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import numpy as np
import os
import random

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
        self.unpatchify_shape = pointcloud.shape

        patch_size = 64
        pcd_patches = patchify(pointcloud, (patch_size,patch_size,patch_size), step=patch_size)
        mask_patches = patchify(mask, (patch_size,patch_size,patch_size), step=patch_size)

        self.patchify_shape = pcd_patches.shape
        patch_orientation = np.array(pcd_patches.shape[:3])
        num_patches = patch_orientation.cumprod()[-1]
        
        if 'train' in self.root_dir:
            n_patches = 4
            n_patch_indices = []
            range_patches = list(range(num_patches))
            while len(n_patch_indices) < n_patches:
                random_patch = random.sample(range_patches, 1)[0]
                if (pcd_patches[np.unravel_index(random_patch, patch_orientation)] == 0).all():
                    range_patches.remove(random_patch)
                else:
                    noise = np.random.randint(0, patch_size, (np.random.randint(0, 1000//num_patches), 3))
                    pcd_patches[np.unravel_index(random_patch, patch_orientation)][noise[:,0], noise[:,1], noise[:,2]] = 1

                    n_patch_indices.append(random_patch)
        else:
            n_patch_indices = list(range(num_patches))

        return \
        [torch.tensor(pcd_patches[np.unravel_index(index, patch_orientation)], dtype=torch.float)[None,None,:] for index in n_patch_indices],\
        [torch.tensor(mask_patches[np.unravel_index(index, patch_orientation)], dtype=torch.float)[None,None,:] for index in n_patch_indices]

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
test_dataset = MaskDataset('./data/test')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

loss_history = []
rolling_loss = 0
old_loss = 9999

with torch.no_grad():
    test_pointcloud_patches, test_mask_patches = next(iter(test_loader))
    test_pointcloud_patches = test_pointcloud_patches.to(device)
    test_mask_patches = test_mask_patches.to(device)

    predicted_mask_patches = model(test_pointcloud_patches)

    predicted_mask = predicted_mask_patches.reshape(test_dataset.patchify_shape)
    original_mask = test_mask_patches.reshape(test_dataset.patchify_shape)

    predicted_mask = unpatchify(predicted_mask, test_dataset.unpatchify_shape)
    original_mask = unpatchify(original_mask, test_dataset.unpatchify_shape)

    for i in range(predicted_mask.shape[-1]):
        plt.imsave(f"vis2/mask{i}_o.png", original_mask[:,:,i].cpu(), cmap=plt.cm.gray)
        plt.imsave(f"vis2/mask{i}_p.png", predicted_mask[:,:,i].cpu(), cmap=plt.cm.gray)  


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
    
    if epoch % 2 == 0:
        if loss.item() < old_loss:
            torch.save(model.state_dict(), f'model.pth')
            old_loss = rolling_loss

        with torch.no_grad():
            test_pointcloud_patches, test_mask_patches = next(iter(test_loader))
            test_pointcloud_patches = test_pointcloud_patches.to(device)
            test_mask_patches = test_mask_patches.to(device)

            predicted_mask_patches = model(test_pointcloud_patches)

            predicted_mask = predicted_mask_patches.reshape(test_dataset.patchify_shape)
            original_mask = test_mask_patches.reshape(test_dataset.patchify_shape)

            predicted_mask = unpatchify(predicted_mask, test_dataset.unpatchify_shape)
            original_mask = unpatchify(original_mask, test_dataset.unpatchify_shape)

            for i in range(predicted_mask.shape[-1]):
                plt.imsave(f"vis2/mask{i}_o.png", original_mask[:,:,i].cpu(), cmap=plt.cm.gray)
                plt.imsave(f"vis2/mask{i}_p.png", predicted_mask[:,:,i].cpu(), cmap=plt.cm.gray)  

        # loss_history = loss_history[-5000:] 
        plt.plot(loss_history)
        plt.savefig('loss.png')
        # plt.show()
                
            
