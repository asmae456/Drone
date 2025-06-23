import os
from shutil import copyfile
import argparse

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import namedtuple

from tools import RunManager, get_num_correct, clear, str2bool, renameBestModel

from models import ResNet_cifar, VGG


# Training settings
parser = argparse.ArgumentParser(description='QNN PyTorch implementation (CIFAR-10 dataset)')

parser.add_argument('-n', '--network', default='ResNet',
					help='Neural network model to use: Choose ResNet or VGG')
parser.add_argument('-l', '--layers', type=int, default=20,
					help='Number of layers on the neural network model')

parser.add_argument('-o', '--optimizer', default='Adam',
					help='Optimizer to update weights')
parser.add_argument('-m', '--momentum', type=float, default=0.9,
					help='Momentum parameter for the SGD optimizer')

parser.add_argument('-wb', '--wbits', type=int, default=1,
					help='Bit width to represent weights')
parser.add_argument('-ab', '--abits', type=int, default=1,
					help='Bit width to represent input activations')

parser.add_argument('-bn', '--batch_norm', type=str2bool, nargs='?', const=True, default=True,
					help='If True, Batch Normalization is used (applicable for VGG; default=True)')

parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
					help='Initial Learning Rate')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4,
					help='Weight decay parameter for optimizer')

parser.add_argument('-bs', '--batch_size', type=int, default=128,
					help='Batch Size for training and validation')
parser.add_argument('-e', '--epochs', type=int, default=200,
					help='Number of epochs to train')

parser.add_argument('-nw', '--number_workers', type=int, default=1,
					help='Number of workers on data loader')

parser.add_argument('-lc', '--load_checkpoint', type=str2bool, nargs='?', const=True, default=False,
					help='To resume training, set to True')

class TrajectoryDataset(Dataset):
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, j = self.indices[index]        
        X = torch.tensor([
            self.dataset['dx'][i, j],
            self.dataset['dy'][i, j],
            self.dataset['dz'][i, j],
            self.dataset['vx'][i, j],
            self.dataset['vy'][i, j],
            self.dataset['vz'][i, j],
            self.dataset['phi'][i, j],
            self.dataset['theta'][i, j],
            self.dataset['psi'][i, j],
            self.dataset['p'][i, j],
            self.dataset['q'][i, j],
            self.dataset['r'][i, j],
            self.dataset['omega'][i, j, 0],
            self.dataset['omega'][i, j, 1],
            self.dataset['omega'][i, j, 2],
            self.dataset['omega'][i, j, 3],
#             self.dataset['Mx_ext'][i],
#             self.dataset['My_ext'][i],
#             self.dataset['Mz_ext'][i]
        ], dtype=torch.float32)
        
        U = torch.tensor([
            self.dataset['u'][i, j, 0],
            self.dataset['u'][i, j, 1],
            self.dataset['u'][i, j, 2],
            self.dataset['u'][i, j, 3]
        ], dtype=torch.float32)
        
        return X, U
    
# trajectories containing 199 points
dataset_path = 'datasets/HOVER_TO_HOVER_NOMINAL.npz'

dataset = dict()
print('loading dataset...')
# See all keys and shapes
# with np.load(dataset_path) as data:
#     for key in data.files:
#         print(key, data[key].shape)

with np.load(dataset_path) as full_dataset:
    # total number of trajectories
    num = len(full_dataset['dx'])
    print(num, 'trajectories')
    dataset = {key: full_dataset[key] for key in [
        't', 'dx', 'dy', 'dz', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r','omega', 'u', 'omega_min','omega_max', 'k_omega', 'Mx_ext', 'My_ext', 'Mz_ext'
    ]}

# train/test split
batchsize_train = 256
batchsize_val = 256
train_trajectories = range(int(0.8*num))
test_trajectories = list(set(range(num)) - set(train_trajectories))

train_indices = [(i, j) for i in train_trajectories for j in range(199)]
train_set = TrajectoryDataset(dataset, train_indices)
train_loader = DataLoader(train_set, batch_size=batchsize_train, shuffle=True, num_workers=1)

test_indices = [(i, j) for i in test_trajectories for j in range(199)]
test_set = TrajectoryDataset(dataset, test_indices)
test_loader = DataLoader(test_set, batch_size=batchsize_val, shuffle=True, num_workers=1)

print('ready')

print('Amount of testing trajectories: ',len(test_trajectories),f'(Batchsize: {batchsize_val})')
print('Amount of training trajectories: ',len(train_trajectories),f'(Batchsize: {batchsize_train})')