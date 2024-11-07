import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import large_data_prep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input dim = state_dim (5+10+10) + action_dim(9) = 34
class QModel(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=128, dropout_prob=0.5):
        super(QModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_q_model(states_tensor, actions_tensor, rewards_tensor, next_states_tensor,
                  batch_size=32, max_iters=10000, lr=5e-4, gamma=0.95,
                  explore_rate=1.0, min_explore_rate=0.1, explore_decay=0.9999,
                  buffer_size=50000, action_dim=9):

    input_dim = states_tensor.size(1) + action_dim
    q_model = QModel(input_dim=input_dim).to(device)
    target_q_model = QModel(input_dim=input_dim).to(device)
    target_q_model.load_state_dict(q_model.state_dict())
    optimizer = optim.Adam(q_model.parameters(), lr=lr)
    
    action_space = np.eye(action_dim)
    replay_buffer = deque(maxlen=buffer_size)
    priorities = deque(maxlen=buffer_size)

    # Initialize replay buffer and priorities
    for idx in range(states_tensor.size(0)):
        state = states_tensor[idx]
        action = actions_tensor[idx]
        reward = rewards_tensor[idx]
        next_state = next_states_tensor[idx]
        experience = (state, action, reward, next_state)
        replay_buffer.append(experience)
        priorities.append(1.0)  # Initial priority

    for i in tqdm(range(max_iters)):
        # Sample batch with prioritized experience replay
        priorities_np = np.array(priorities, dtype=np.float32)
        sampling_probs = priorities_np ** 0.6  # alpha parameter
        sampling_probs /= sampling_probs.sum()


        indices = np.random.choice(len(replay_buffer), batch_size, p=sampling_probs)
        batch = [replay_buffer[idx] for idx in indices]
        batch_importance = (1 / (len(replay_buffer) * sampling_probs[indices])) ** 0.4  # beta parameter
        batch_importance = torch.tensor(batch_importance, dtype=torch.float32).to(device)
        batch_importance /= batch_importance.max()


        batch_states = torch.stack([x[0] for x in batch]).to(device)
        batch_actions = torch.stack([x[1] for x in batch]).to(device)
        batch_rewards = torch.stack([x[2] for x in batch]).to(device)
        batch_next_states = torch.stack([x[3] for x in batch]).to(device)
        
        batch_state_actions = torch.cat([batch_states, batch_actions], dim=1)
        current_q_values = q_model(batch_state_actions).squeeze()

        with torch.no_grad():
            num_actions = action_space.shape[0]
            action_space_tensor = torch.tensor(action_space, dtype=torch.float32).to(device)
            
            batch_size_actual = batch_next_states.size(0)
            next_states_expanded = batch_next_states.unsqueeze(1).repeat(1, num_actions, 1)
            actions_expanded = action_space_tensor.unsqueeze(0).repeat(batch_size_actual, 1, 1)
            
            next_state_actions = torch.cat([next_states_expanded, actions_expanded], dim=2)
            next_state_actions_flat = next_state_actions.view(-1, next_state_actions.size(2))
            
            q_values = target_q_model(next_state_actions_flat).view(batch_size_actual, num_actions)
            max_q_values, _ = q_values.max(dim=1)
            
            target_q_values = batch_rewards + gamma * max_q_values

        td_errors = target_q_values - current_q_values
        loss = (batch_importance * td_errors.pow(2)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update priorities
        for idx, replay_idx in enumerate(indices):
            priorities[replay_idx] = abs(td_errors[idx].item()) + 1e-6  # Small constant to prevent zero priority

        # Update exploration rate
        if explore_rate > min_explore_rate:
            explore_rate *= explore_decay

        # Soft update of target network
        tau = 0.01  # Soft update parameter
        for target_param, param in zip(target_q_model.parameters(), q_model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        # Add new experience
        idx = torch.randint(0, states_tensor.size(0), (1,)).item()
        state = states_tensor[idx]
        action = actions_tensor[idx]
        reward = rewards_tensor[idx]
        next_state = next_states_tensor[idx]
        experience = (state, action, reward, next_state)
        replay_buffer.append(experience)
        priorities.append(max(priorities) if priorities else 1.0)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}, Avg TD Error: {td_errors.mean().item()}")

    print("Training complete")
    return q_model

if __name__ == '__main__':
    print(f'Running torch on device: {device}')
    
    states_encoded, next_states_encoded, actions_encoded, rewards,\
    all_one_hot_states, all_one_hot_actions, index_map, rev_index_map, \
    decode_state, decode_action = large_data_prep.prepare_data_np()
    
    states_tensor = torch.tensor(states_encoded, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions_encoded, dtype=torch.float32).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states_tensor = torch.tensor(next_states_encoded, dtype=torch.float32).to(device)
    
    q_model = train_q_model(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, max_iters=10000)
