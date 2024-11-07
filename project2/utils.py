import numpy as np
import random
import torch

def read_data(file):
    return np.loadtxt(f'data/{file}.csv', delimiter=',', skiprows=1)

def write_policy(file, policy):
    n = len(policy) - 1
    out = [f'{x}\n' for x in policy[:-1]] + [str(policy[-1])]

    with open(f'output/{file}.policy', 'w') as f:
        f.writelines(out)

def train_test_split_tensors(states, actions, rewards, rewards_idx, next_states, device, train_ratio=0.9,
    dtypes = [torch.long, torch.long, torch.float, torch.long, torch.long],
    adjust_indexing = [False, True, False, False, False]):

    dataset = list(zip(states, actions, rewards, rewards_idx, next_states))
    random.shuffle(dataset)

    split_idx = int(train_ratio * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    def process_dataset(dataset, device):
        data = list(zip(*dataset)) 
        tensors = []

        for i, (dtype, adjust_idx) in enumerate(zip(dtypes, adjust_indexing)):
            np_array = np.array(data[i])
            tensor = torch.tensor(np_array, dtype=dtype).to(device)
            if adjust_idx:
                tensor = tensor - 1
            tensors.append(tensor)
        return tensors

    train_tensors = process_dataset(train_dataset, device)
    test_tensors = process_dataset(test_dataset, device)

    return (*train_tensors, *test_tensors)

def augment_data(data, duplication_factor=4):
    zero_reward_rows = data[data[:, 2] == 0]
    non_zero_reward_rows = data[data[:, 2] != 0]
    
    unique_rewards = np.unique(non_zero_reward_rows[:, 2])
    augmented_data_parts = [data]  # Start with the original data
    
    for reward in unique_rewards:
        reward_rows = non_zero_reward_rows[non_zero_reward_rows[:, 2] == reward]
        num_instances = reward_rows.shape[0]
        
        sampled_rows = reward_rows[np.random.choice(num_instances, size=duplication_factor * num_instances, replace=True)]
        augmented_data_parts.append(sampled_rows)
    
    augmented_data = np.concatenate(augmented_data_parts, axis=0)
    np.random.shuffle(augmented_data)
    
    return augmented_data

def get_mdp_tensors(data, device):
    tensors = []
    for i in range(data.shape[1]):
        col = data[:, i]
        tensor = torch.tensor(col).unsqueeze(1).to(device)
        tensors.append(tensor)
    
    return tuple(tensors)


