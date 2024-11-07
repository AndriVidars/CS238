import numpy as np
import utils
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model that learns the transition mechanics

# use one hot encoding of p, v as input to models?

class TransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim_p, output_dim_v, dropout_prob=0.5):
        super(TransitionModel, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc_p = nn.Linear(hidden_dim, output_dim_p)
        self.fc_v = nn.Linear(hidden_dim, output_dim_v)
        
    def forward(self, x):
        x = self.relu(self.fc(x.float()))
        x = self.dropout(x)
        p_out = self.fc_p(x)
        v_out = self.fc_v(x)
        return p_out, v_out

def train_transition_model(states_pv_tensor, actions_tensor, next_states_pv_tensor, n_epochs=50):
    print("Training transition model")
    inputs = torch.cat([states_pv_tensor, actions_tensor.unsqueeze(1)], dim=1)
    targets = next_states_pv_tensor

    transition_dataset = TensorDataset(inputs, targets)
    transition_loader = DataLoader(transition_dataset, batch_size=64, shuffle=True)

    transition_model = TransitionModel(input_dim=3, hidden_dim=128, output_dim_p=500, output_dim_v=100).to(device)
    criterion_transition_p = nn.CrossEntropyLoss()
    criterion_transition_v = nn.CrossEntropyLoss()
    optimizer_transition = torch.optim.Adam(transition_model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        transition_model.train() 
        total_loss = 0
        for batch_inputs, batch_targets in transition_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer_transition.zero_grad()
            outputs_p, outputs_v = transition_model(batch_inputs)
            loss_p = criterion_transition_p(outputs_p, batch_targets[:, 0])
            loss_v = criterion_transition_v(outputs_v, batch_targets[:, 1])
            loss = loss_p + loss_v
            loss.backward()
            optimizer_transition.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == n_epochs-1:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(transition_loader)}')
    
    return transition_model

def evaluate_transition_model(transition_model, states_pv_tensor, actions_tensor, next_states_pv_tensor):
    transition_model.eval()
    inputs = torch.cat([states_pv_tensor, actions_tensor.unsqueeze(1)], dim=1)
    targets = next_states_pv_tensor
    evaluation_dataset = TensorDataset(inputs, targets)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=256, shuffle=False)
    absolute_errors_p = []
    absolute_errors_v = []

    with torch.no_grad():
        for batch_inputs, batch_targets in evaluation_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            outputs_p, outputs_v = transition_model(batch_inputs)

            predicted_p_values = outputs_p.argmax(dim=1)
            predicted_v_values = outputs_v.argmax(dim=1)
            true_p_values = batch_targets[:, 0].long()
            true_v_values = batch_targets[:, 1].long()
            abs_errors_p = torch.abs(predicted_p_values - true_p_values)
            abs_errors_v = torch.abs(predicted_v_values - true_v_values)

            absolute_errors_p.extend(abs_errors_p.cpu().numpy())
            absolute_errors_v.extend(abs_errors_v.cpu().numpy())

    absolute_errors_p = np.array(absolute_errors_p)
    absolute_errors_v = np.array(absolute_errors_v)
    quantiles = [0.25, 0.5, 0.75, 0.95, 0.99]
    quantiles_p = np.quantile(absolute_errors_p, quantiles)
    quantiles_v = np.quantile(absolute_errors_v, quantiles)

    print("Quantiles of Absolute Errors for Position (p):")
    for q, value in zip(quantiles, quantiles_p):
        print(f"{int(q*100)}th percentile: {value:.4f}")

    print("\nQuantiles of Absolute Errors for Velocity (v):")
    for q, value in zip(quantiles, quantiles_v):
        print(f"{int(q*100)}th percentile: {value:.4f}")


class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.25):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x.float()))
        x = self.dropout(x)
        return self.fc_2(x)

def train_rewards_model(states_pv_tensor, actions_tensor, rewards_idx_tensor, n_epochs=50):
    print("Training Rewards model")
    inputs = torch.cat([states_pv_tensor, actions_tensor.unsqueeze(1)], dim=1)
    targets = rewards_idx_tensor

    rewards_dataset = TensorDataset(inputs, targets)
    rewards_loader = DataLoader(rewards_dataset, batch_size=64, shuffle=True)

    rewards_model = RewardModel(input_dim=3, hidden_dim=128, output_dim=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rewards_model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        rewards_model.train() 
        total_loss = 0
        for batch_inputs, batch_targets in rewards_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = rewards_model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == n_epochs-1:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(rewards_loader)}')
    
    return rewards_model

def evaluate_reward_model(rewards_model, states_pv_tensor, actions_tensor, rewards_idx_tensor, unique_rewards):
    rewards_model.eval()
    inputs = torch.cat([states_pv_tensor, actions_tensor.unsqueeze(1)], dim=1)
    targets = rewards_idx_tensor

    evaluation_dataset = TensorDataset(inputs, targets)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=256, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_inputs, batch_targets in evaluation_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            outputs = rewards_model(batch_inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    print(f'Rewards Model Accuracy on Test Data: {accuracy * 100:.2f}%\n')

def get_model_pred_arrays(transition_model, rewards_model, num_states=50000, num_actions=7):
    """
    Precompute predicted rewards and transitions from NN models.
    Adjust so that for p >= 466 the states are self-absorbing with reward 0.

    Additionally, provide diagnostics on how many transitions and rewards are overridden
    due to the constraints (p >= 466).
    """
    transition_model.eval()
    rewards_model.eval()

    state_indices = torch.arange(0, num_states, dtype=torch.long).to(device)  # 0-based indexing

    positions, velocities = zip(*[get_pv(idx, zero_index=True) for idx in state_indices.cpu().numpy()])
    positions = torch.tensor(positions, dtype=torch.float32, device=device)
    velocities = torch.tensor(velocities, dtype=torch.float32, device=device)
    states_pv_tensor = torch.stack([positions, velocities], dim=1)  # Shape: (num_states, 2)

    repeated_states_pv_tensor = states_pv_tensor.repeat_interleave(num_actions, dim=0)  
    actions_tensor = torch.arange(num_actions, device=device).repeat(num_states).unsqueeze(1) 

    inputs = torch.cat([repeated_states_pv_tensor, actions_tensor.float()], dim=1)

    with torch.no_grad():
        next_states_p, next_states_v = transition_model(inputs)
        prob_p = nn.Softmax(dim=1)(next_states_p)
        prob_v = nn.Softmax(dim=1)(next_states_v)
        next_p = torch.multinomial(prob_p, num_samples=1).squeeze(1)  
        next_v = torch.multinomial(prob_v, num_samples=1).squeeze(1) 

    next_state_indices = next_v * 500 + next_p 

    p_values = repeated_states_pv_tensor[:, 0] 
    absorbing_mask = p_values >= 466 

    state_indices_expanded = state_indices.repeat_interleave(num_actions) 
    next_state_indices_overridden = next_state_indices.clone() 
    next_state_indices_overridden[absorbing_mask] = state_indices_expanded[absorbing_mask]

    next_state_indices = next_state_indices_overridden
    transitions = next_state_indices.view(num_states, num_actions).cpu().numpy()

    # Predict rewards using the rewards model
    with torch.no_grad():
        rewards_output = rewards_model(inputs) 
        predicted_reward_indices = rewards_output.argmax(dim=1)

    predicted_rewards = torch.tensor([idx_to_reward[idx.item()] for idx in predicted_reward_indices], device=device)

    predicted_rewards[absorbing_mask] = 0.0 
    rewards = predicted_rewards.view(num_states, num_actions).cpu().numpy()

    assert transitions.max() < num_states, "Transition indices exceed the number of states!"
    assert transitions.min() >= 0, "Transition indices contain negative values!"

    return transitions, rewards

def value_iteration(transitions, rewards, num_states=50000, num_actions=7, discount_rate=0.999, theta=1e-4, max_iters=10000):
    V = np.zeros(num_states)

    for i in tqdm(range(max_iters), desc="Value Iteration"):
        V_prev = V.copy()
        action_values = rewards + discount_rate * V[transitions]
        V = np.max(action_values, axis=1)
        delta = np.max(np.abs(V - V_prev))

        if (i+1) % 1000 == 0 or i == 0:
            print(f"Iteration {i+1}: ΔV = {delta:.6f}")

        if delta < theta:
            print(f"Value Iteration converged after {i+1} iterations.")
            break

    policy = np.argmax(action_values, axis=1) + 1
    return V, policy

# Get position/velocity representation for states
def get_pv(idx, zero_index=False):
    idx = idx if zero_index else idx - 1
    v = idx // 500
    p = idx % 500
    return p, v

if __name__ == '__main__':
    print(f'Running torch on device: {device}')

    data = utils.read_data('medium')
    states = data[:, 0]
    actions = data[:, 1]
    rewards = data[:, 2]
    
    next_states = data[:, 3]
    states_pv = np.array([get_pv(s) for s in states])
    next_states_pv = np.array([get_pv(s) for s in next_states])
   
    unique_rewards = np.unique(rewards)
    reward_to_idx = {reward: idx for idx, reward in enumerate(unique_rewards)}
    idx_to_reward = {idx: reward for reward, idx in reward_to_idx.items()}
    reward_idx = np.array([reward_to_idx[r] for r in rewards])

    train_states_tensor, train_actions_tensor, train_rewards_tensor, train_rewards_idx_tensor, train_next_states_tensor, \
    test_states_tensor, test_actions_tensor, test_rewards_tensor, test_rewards_idx_tensor, test_next_states_tensor = \
        utils.train_test_split_tensors(states_pv, actions, rewards, reward_idx, next_states_pv, device, train_ratio=0.9)

    # train transition model
    transition_model = train_transition_model(train_states_tensor, train_actions_tensor, train_next_states_tensor, n_epochs=150) # n_epochs 50
    evaluate_transition_model(transition_model, test_states_tensor, test_actions_tensor, test_next_states_tensor)

    # train rewards model
    rewards_model = train_rewards_model(train_states_tensor, train_actions_tensor, train_rewards_idx_tensor, n_epochs=25) # n_epochs 20
    evaluate_reward_model(rewards_model, test_states_tensor, test_actions_tensor, test_rewards_idx_tensor, unique_rewards)

    transitions, rewards = get_model_pred_arrays(transition_model, rewards_model)
    V, policy = value_iteration(transitions, rewards)

    utils.write_policy('medium', policy)
    print("Done")
    