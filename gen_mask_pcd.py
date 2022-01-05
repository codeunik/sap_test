import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import copy

voxel_grid_size = 192

os.system('mkdir -p data/train/pointclouds')
os.system('mkdir -p data/train/masks')

os.system('mkdir -p data/test/pointclouds')
os.system('mkdir -p data/test/masks')

os.system('mkdir -p vis')
os.system('mkdir -p vis2')

for _, _, f in os.walk(f'data/meshes'):
    mesh_files = f

purpose = 'train'
count = 0

for mesh_file in mesh_files:
    mesh = o3d.io.read_triangle_mesh(f"data/meshes/{mesh_file}")
    
    for _ in range(10):
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.random.random(3) * np.pi), mesh_copy.get_center())
        max_coord = max(mesh_copy.get_max_bound())
        min_coord = min(mesh_copy.get_min_bound())
        center = (max_coord + min_coord)/2
        center = np.array([center, center, center])
        mesh_copy.translate(-center)
        mesh_copy.scale(0.9*voxel_grid_size/((max_coord-min_coord)), (0,0,0))
        mesh_copy.translate(np.array([voxel_grid_size, voxel_grid_size, voxel_grid_size])/2 - center)
        
        pointcloud = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size))
        mask = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size))

        pcd = mesh_copy.sample_points_uniformly(number_of_points=10000)
        pcd = np.asarray(pcd.points, dtype=int)
        pointcloud[pcd[:,0], pcd[:,1], pcd[:,2]] = 1

        pcd = mesh_copy.sample_points_uniformly(number_of_points=100000)
        pcd = np.asarray(pcd.points, dtype=int)
        for x,y,z in pcd:
            mask[x-2:x+3, y-2:y+3, z-2:z+3] = 1

        np.save(f"data/{purpose}/pointclouds/{count}.npy", pointcloud)
        np.save(f"data/{purpose}/masks/{count}.npy", mask)
        count += 1
        