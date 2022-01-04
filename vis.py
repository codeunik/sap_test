import matplotlib.pyplot as plt
import numpy as np

pcd = np.load('data/train/pointclouds/0.npy')
mask = np.load('data/train/masks/0.npy')

for i in range(pcd.shape[2]):
    plt.imsave(f"vis/pointcloud{i}.png", pcd[:,:,i], cmap=plt.cm.gray)
    plt.imsave(f"vis/mask{i}.png", mask[:,:,i], cmap=plt.cm.gray)
