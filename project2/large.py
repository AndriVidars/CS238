import torch
import torch.nn as nn
import torch.optim as optim
import utils
import random
from collections import deque
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, dropout_prob=0.5):
        super(QModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def epsilon_greedy_action(q_model, state, explore_rate, action_space=range(9)):
    if random.random() < explore_rate:
        return int(random.choice(action_space))
    else:
        q_values = [q_model(torch.tensor([state, a], dtype=torch.float32).to(device)).item() for a in action_space]
        return torch.argmax(torch.tensor(q_values))

# TODO: tune lr
def train_q_model(states_tensor, actions_tensor, rewards_tensor, next_states_tensor,
                  batch_size=64, max_iters=10_000, lr=1e-3, gamma=0.95, explore_rate=0.1,
                  buffer_size=1000):

    q_model = QModel(input_dim=2).to(device)
    optimizer = optim.SGD(q_model.parameters(), lr=lr)
    
    # init buffer
    replay_buffer = deque(maxlen=buffer_size)
    
    for idx in range(buffer_size):
        experience = (states_tensor[idx].item(), actions_tensor[idx].item(),
                      rewards_tensor[idx].item(), next_states_tensor[idx].item())
        replay_buffer.append(experience)
    
    for i in tqdm(range(max_iters)):
        batch = random.sample(replay_buffer, min(batch_size, len(replay_buffer)))
        
        batch_states = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(device)
        batch_actions = torch.tensor([x[1] for x in batch], dtype=torch.int32).to(device)
        batch_rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(device)
        batch_next_states = torch.tensor([x[3] for x in batch], dtype=torch.float32).to(device)
        
        current_q_inputs = torch.stack([batch_states, batch_actions], dim=1).float().to(device)
        current_q_values = q_model(current_q_inputs).squeeze()
        
        with torch.no_grad():
            next_q_values = []
            for j in range(batch_size):
                next_state = batch_next_states[j].item()
                next_action = epsilon_greedy_action(q_model, next_state, explore_rate)
                next_q_input = torch.tensor([next_state, next_action], dtype=torch.float32).to(device)
                next_q_values.append(q_model(next_q_input).item())
            
            next_q_values = torch.tensor(next_q_values, dtype=torch.float32).to(device)
            target_q_values = batch_rewards + gamma * next_q_values
        
        #td_errors = target_q_values - current_q_values
        #loss = -(td_errors * current_q_values).mean()
        td_errors = target_q_values - current_q_values
        loss = td_errors.pow(2).mean()  # MSE
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if explore_rate > 0.1:
            explore_rate *= 0.9999
        
        idx = torch.randint(0, states_tensor.size(0), (1,)).item()
        experience = (states_tensor[idx].item(), actions_tensor[idx].item(),
                      rewards_tensor[idx].item(), next_states_tensor[idx].item())
        replay_buffer.append(experience)

        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}, Avg TD Error: {td_errors.mean().item()}")

    print("Training complete")
    return q_model

def extract_policy(q_model, num_states=302020, num_actions=9):
    s_vals = range(1, num_states + 1)
    a_vals = range(num_actions)
    
    state_arr = np.array([[s, a] for s in s_vals for a in a_vals])
    
    state_tensor = torch.tensor(state_arr, dtype=torch.float32).to(device)
    
    q_values = q_model(state_tensor).view(num_states, num_actions)
    policy = torch.argmax(q_values, dim=1) + 1
    
    return policy.cpu().numpy().astype(int)

if __name__ == '__main__':
    data = utils.read_data("large")
    data = utils.augment_data(data, duplication_factor=10) # upsample non zero rewards
    print(f"Data shape after augment: {data.shape}")
    
    states_tensor, actions_tensor, rewards_tensor, next_states_tensor = utils.get_mdp_tensors(data, device)
    actions_tensor = (actions_tensor - 1).to(torch.int32)
    
    # TODO: do Prioritized Experience Replay rather than data augment(upsampling rewards) # or mix of the two 
    # maybe use upsampling with duplication factor 4-6 and also prioritized experience replay
    q_model = train_q_model(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, max_iters=20_000)
    policy = extract_policy(q_model)
    utils.write_policy("large", policy)
    
    print("done")
