import numpy as np
import utils
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model that learns the transition mechanics
class TransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim_p, output_dim_v, dropout_prob=0.25):
        super(TransitionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc_p = nn.Linear(hidden_dim, output_dim_p)
        self.fc_v = nn.Linear(hidden_dim, output_dim_v)
        
    def forward(self, x):
        x = self.relu(self.fc1(x.float()))
        x = self.dropout(x)
        p_out = self.fc_p(x)
        v_out = self.fc_v(x)
        return p_out, v_out

def train_transition_model(states_pv_tensor, actions_tensor, next_states_pv_tensor, n_epochs=50):
    inputs = torch.cat([states_pv_tensor, actions_tensor.unsqueeze(1)], dim=1)
    targets = next_states_pv_tensor

    transition_dataset = TensorDataset(inputs, targets)
    transition_loader = DataLoader(transition_dataset, batch_size=64, shuffle=True)

    transition_model = TransitionModel(input_dim=3, hidden_dim=128, output_dim_p=500, output_dim_v=100).to(device)
    criterion_transition_p = nn.CrossEntropyLoss()
    criterion_transition_v = nn.CrossEntropyLoss()
    optimizer_transition = torch.optim.Adam(transition_model.parameters(), lr=0.001)

    # Training loop with loss printing
    for epoch in range(n_epochs):
        transition_model.train()  # Set model to training mode
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

class ValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.25):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x.float()))
        x = self.dropout(x)
        value = self.fc_out(x)
        return value

# Primary model
class PolicyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, dropout_prob=0.25):
        super(PolicyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc_out = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        x = self.relu(self.fc1(x.float()))
        x = self.dropout(x)
        action_scores = self.fc_out(x)
        return action_scores

# Simulate the environment using the Transition Model
def simulate_environment(state_pv_tensor, action, transition_model):
    input_tensor = torch.cat([state_pv_tensor, action.unsqueeze(0)], dim=0).unsqueeze(0).to(device)
    outputs_p, outputs_v = transition_model(input_tensor)
    prob_p = nn.Softmax(dim=1)(outputs_p)
    prob_v = nn.Softmax(dim=1)(outputs_v)
    
    # now use softmax probs to simulate next sate
    next_p = torch.multinomial(prob_p, num_samples=1)
    next_v = torch.multinomial(prob_v, num_samples=1)
    next_state_pv = torch.cat([next_p.squeeze(1), next_v.squeeze(1)], dim=0)
    return next_state_pv

def train_policy_model(states_pv_tensor, rewards_tensor, transition_model, num_iterations=10, num_epochs=10):
    policy_model = PolicyModel(input_dim=2, hidden_dim=128, num_actions=7).to(device)
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=0.001)

    value_function = ValueFunction(input_dim=2, hidden_dim=128).to(device)
    optimizer_value = torch.optim.Adam(value_function.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    gamma = 1.0  # Undiscounted
    for iteration in range(num_iterations):
        print(f'Policy Iteration {iteration+1}/{num_iterations}')
        # Policy Evaluation
        for epoch in range(num_epochs):
            total_value_loss = 0
            for i in range(len(states_pv_tensor)):
                state = states_pv_tensor[i].to(device)
                reward = rewards_tensor[i].to(device)
                action_logits = policy_model(state)
                action_prob = nn.Softmax(dim=0)(action_logits)
                expected_value = 0
                for a in range(7):
                    action = torch.tensor([a], dtype=torch.long).to(device)
                    next_state = simulate_environment(state, action, transition_model)
                    value_next = value_function(next_state)
                    expected_value += action_prob[a] * (reward + gamma * value_next)
                value_pred = value_function(state)
                loss_value = mse_loss(value_pred, expected_value.detach())
                optimizer_value.zero_grad()
                loss_value.backward()
                optimizer_value.step()
                total_value_loss += loss_value.item()
            print(f'Value Function Epoch {epoch+1}/{num_epochs}, Loss: {total_value_loss/len(states_pv_tensor)}')

        # Policy Improvement
        for epoch in range(num_epochs):
            total_policy_loss = 0
            for i in range(len(states_pv_tensor)):
                state = states_pv_tensor[i].to(device)
                reward = rewards_tensor[i].to(device)
                optimizer_policy.zero_grad()
                action_logits = policy_model(state)
                action_prob = nn.Softmax(dim=0)(action_logits)
                values = []
                for a in range(7):
                    action = torch.tensor([a], dtype=torch.long).to(device)
                    next_state = simulate_environment(state, action, transition_model)
                    value_next = value_function(next_state)
                    values.append((reward + gamma * value_next.item()))
                values = torch.tensor(values).to(device)
                loss_policy = -torch.sum(action_prob * values)
                loss_policy.backward()
                optimizer_policy.step()
                total_policy_loss += loss_policy.item()
            print(f'Policy Model Epoch {epoch+1}/{num_epochs}, Loss: {total_policy_loss/len(states_pv_tensor)}')

    return policy_model

# Get position/velocity representation for states
def get_pv(idx):
    idx = idx - 1
    v = idx // 500
    p = idx % 500
    return p, v

def evaluate_transition_model(transition_model, states_pv_tensor, actions_tensor, next_states_pv_tensor):
    # Set the model to evaluation mode
    transition_model.eval()
    
    # Prepare the inputs and targets
    inputs = torch.cat([states_pv_tensor, actions_tensor.unsqueeze(1)], dim=1)
    targets = next_states_pv_tensor

    # Create a DataLoader for the evaluation data
    evaluation_dataset = TensorDataset(inputs, targets)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=64, shuffle=False)
    
    # Initialize counters
    correct_p = 0
    total_p = 0
    correct_v = 0
    total_v = 0
    correct_both = 0
    total_samples = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in evaluation_loader:
            # Move data to the appropriate device
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Get model outputs
            outputs_p, outputs_v = transition_model(batch_inputs)
            
            # Predicted classes
            _, predicted_p = torch.max(outputs_p, dim=1)
            _, predicted_v = torch.max(outputs_v, dim=1)
            
            # Update counters for position accuracy
            correct_p += (predicted_p == batch_targets[:, 0]).sum().item()
            total_p += batch_targets.size(0)
            
            # Update counters for velocity accuracy
            correct_v += (predicted_v == batch_targets[:, 1]).sum().item()
            total_v += batch_targets.size(0)
            
            # Update counters for overall accuracy
            both_correct = (predicted_p == batch_targets[:, 0]) & (predicted_v == batch_targets[:, 1])
            correct_both += both_correct.sum().item()
            total_samples += batch_targets.size(0)
    
    # Calculate accuracy metrics
    p_accuracy = correct_p / total_p
    v_accuracy = correct_v / total_v
    overall_accuracy = correct_both / total_samples
    
    print(f'Position accuracy: {p_accuracy * 100:.2f}%')
    print(f'Velocity accuracy: {v_accuracy * 100:.2f}%')
    print(f'Overall accuracy (both position and velocity correct): {overall_accuracy * 100:.2f}%')

if __name__ == '__main__':
    print(f'Running torch on device: {device}')

    data = utils.read_data('medium')
    states = data[:, 0]
    actions = data[:, 1]
    rewards = data[:, 2]
    next_states = data[:, 3]
    states_pv = np.array([get_pv(s) for s in states])
    next_states_pv = np.array([get_pv(s) for s in next_states])

    states_pv_tensor = torch.tensor(states_pv, dtype=torch.long).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long) - 1  # Zero indexing
    actions_tensor = actions_tensor.to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
    next_states_pv_tensor = torch.tensor(next_states_pv, dtype=torch.long).to(device)

    # Train the Transition Model
    transition_model = train_transition_model(states_pv_tensor, actions_tensor, next_states_pv_tensor, n_epochs=100)
    evaluate_transition_model(transition_model, states_pv_tensor, actions_tensor, next_states_pv_tensor)

    # Train the Policy Model
    # TODO bugfix here
    #policy_model = train_policy_model(states_pv_tensor, rewards_tensor, transition_model, num_iterations=10, num_epochs=10)
