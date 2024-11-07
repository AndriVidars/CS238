import torch
import torch.nn as nn
import utils
import random
from tqdm import tqdm

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

def epsilon_greedy_action(q_model, state, action_space, explore_rate):
    if random.random() < explore_rate:
        return random.choice(action_space)
    else:
        q_values = [q_model(torch.tensor([state, a], dtype=torch.float32).to(device)).item() for a in action_space]
        return action_space[torch.argmax(torch.tensor(q_values))]

def train_q_model(states_tensor, actions_tensor, rewards_tensor, next_states_tensor,
                  max_iters=1_000_000, lr=0.01, gamma=0.95, explore_rate=0.1):
    q_model = QModel(input_dim=2).to(device)
    
    action_space = torch.unique(actions_tensor).tolist()
    
    for i in tqdm(range(max_iters)):
        idx = torch.randint(0, states_tensor.size(0), (1,)).item()
        state = states_tensor[idx].item()
        action = actions_tensor[idx].item()
        reward = rewards_tensor[idx].item()
        next_state = next_states_tensor[idx].item()
        
        next_action = epsilon_greedy_action(q_model, next_state, action_space, explore_rate)
        
        current_q_input = torch.tensor([state, action], dtype=torch.float32).to(device)
        current_q_value = q_model(current_q_input)
        
        with torch.no_grad():
            next_q_input = torch.tensor([next_state, next_action], dtype=torch.float32).to(device)
            next_q_value = q_model(next_q_input)
            target_q_value = reward + gamma * next_q_value
        
        # Compute TD error
        td_error = target_q_value - current_q_value
        
        # Zero out any existing gradients
        q_model.zero_grad()
        current_q_value.backward(retain_graph=True)
        
        with torch.no_grad():
            for param in q_model.parameters():
                param += lr * td_error * param.grad 

        # Optional: decay exploration rate over time
        if explore_rate > 0.1:
            explore_rate *= 0.9999

        # Print progress occasionally
        if i % 1000 == 0:
            print(f"Iteration {i}, TD Error: {td_error.item()}")

    print("Training complete")

if __name__ == '__main__':
    print(f"Running torch on device: {device}")
    data = utils.read_data("large")
    states_tensor, actions_tensor, rewards_tensor, next_states_tensor = utils.get_mdp_tensors(data, device)
    train_q_model(states_tensor, actions_tensor, rewards_tensor, next_states_tensor)
    print("done")
