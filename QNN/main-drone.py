import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from models import Drone
from tools import str2bool

# Set environment variables for deterministic behavior
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Argument parser
parser = argparse.ArgumentParser(description='Validation Only - Drone Model on Custom Dataset')
parser.add_argument('-wb', '--wbits', type=int, default=8, help='Bit width to represent weights')
parser.add_argument('-ab', '--abits', type=int, default=8, help='Bit width to represent input activations')
parser.add_argument('--checkpoint', type=str, default='/home/aelarrassi/drone/neural_networks/tmp_benchmark_quantized.pt', help='Path to quantized model checkpoint')
parser.add_argument('--dataset_path', type=str, default='/home/aelarrassi/drone/datasets/HOVER_TO_HOVER_NOMINAL.npz', help='Path to dataset')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for validation')
args = parser.parse_args()

# Dataset definition
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
            self.dataset['omega'][i, j, 3]
        ], dtype=torch.float32)

        U = torch.tensor([
            self.dataset['u'][i, j, 0],
            self.dataset['u'][i, j, 1],
            self.dataset['u'][i, j, 2],
            self.dataset['u'][i, j, 3]
        ], dtype=torch.float32)

        return X, U

# Load dataset
print('Loading dataset...')
with np.load(args.dataset_path) as full_dataset:
    num_trajectories = len(full_dataset['dx'])
    dataset = {key: full_dataset[key] for key in [
        'dx', 'dy', 'dz', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi',
        'p', 'q', 'r', 'omega', 'u'
    ]}

test_trajectories = range(int(0.8 * num_trajectories), num_trajectories)
test_indices = [(i, j) for i in test_trajectories for j in range(199)]
test_set = TrajectoryDataset(dataset, test_indices)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

from tqdm import tqdm
X_mean = torch.zeros(16)
X_std = torch.zeros(16)

N=10000

for i, data in tqdm(enumerate(test_set)):
    X = data[0]
    X_mean += X
    if i>=N:
        break
X_mean = X_mean/N

print('mean:')
print(X_mean)
    
for i, data in tqdm(enumerate(test_set)):
    X = data[0]
    X_std += (X-X_mean)**2
    if i>=N:
        break

X_std = torch.sqrt(X_std/N)
print('std:')
print(X_std)


print(f'Test set prepared: {len(test_trajectories)} trajectories, {len(test_set)} samples')

# Load model
print('Loading model...')
checkpoint = torch.load('neural_networks/HOVER_TO_HOVER_NOMINAL1.pt', weights_only=False)
model = Drone(wbits=args.wbits, abits=args.abits, X_mean=X_mean, X_std=X_std)

state_dict = checkpoint['network_state_dict']
new_state_dict = {}
key_map = {
    "1": "model.1",
    "3": "model.3",
    "5": "model.5",
    "7": "model.7"
}

for k, v in state_dict.items():
    layer_num, param_type = k.split(".")
    if layer_num in key_map:
        new_key = f"{key_map[layer_num]}.{param_type}"
        new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)


model.eval()

# # Validation
# criterion = torch.nn.MSELoss()
# total_loss = 0.0

# print('Running validation...')
# with torch.no_grad():
#     for inputs, targets in tqdm(test_loader, desc='Validating', leave=False):
#         inputs, targets = inputs.to('cpu'), targets.to('cpu')
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         total_loss += loss.item()

# avg_loss = total_loss / len(test_loader)
# print(f'\nValidation completed. Average loss: {avg_loss:.6f}')

# Validation
##############################################################
loader = test_loader
criterion = torch.nn.MSELoss()
# loop over the test dataset
loop = tqdm(enumerate(loader), total=len(loader), leave=False)
running_loss = 0

for i, (data, targets) in loop:
    outputs = model(data)
    loss = criterion(outputs, targets)
    
    running_loss += loss.item()
    
    # update progressbar
    loop.set_postfix(loss=loss.item())

loop.close()
print('average loss =', running_loss/len(loader))
###########Baseline#############
########### Baseline - Floating-Point Model Evaluation ###########
from models import DroneFP  # make sure DroneFP is implemented properly

print('Loading floating-point model...')
fp_model = DroneFP(X_mean=X_mean, X_std=X_std)

# Load checkpoint
checkpoint_fp = torch.load('neural_networks/HOVER_TO_HOVER_NOMINAL1.pt', weights_only=False)
state_dict_fp = checkpoint_fp['network_state_dict']
new_state_dict_fp = {}
key_map = {
    "1": "model.1",
    "3": "model.3",
    "5": "model.5",
    "7": "model.7"
}

for k, v in state_dict_fp.items():
    layer_num, param_type = k.split(".")
    if layer_num in key_map:
        new_key = f"{key_map[layer_num]}.{param_type}"
        new_state_dict_fp[new_key] = v

fp_model.load_state_dict(new_state_dict_fp)
fp_model.eval()
for name, param in fp_model.named_parameters():
    if 'weight' in name:
        # print(f"{name}:")
        # print(param.data)
        print("-" * 80)
# Evaluate FP model
criterion = torch.nn.MSELoss()
fp_loop = tqdm(enumerate(test_loader), total=len(test_loader), desc='FP Validation', leave=False)
fp_running_loss = 0

for i, (data, targets) in fp_loop:
    with torch.no_grad():
        outputs = fp_model(data)
        loss = criterion(outputs, targets)
        fp_running_loss += loss.item()
        fp_loop.set_postfix(fp_loss=loss.item())

fp_loop.close()
print(f'FP model average loss = {fp_running_loss / len(test_loader):.20f}')
