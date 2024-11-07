import utils
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm

class ValueIteration:
    def __init__(self, states, actions, rewards, next_states, num_states, num_actions):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.num_states = num_states
        self.num_actions = num_actions
            
    # MLE estimate for T
    def estimate_transition_probs(self):
        t_counts = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=int)
        np.add.at(t_counts, (self.states, self.actions, self.next_states), 1)

        T = np.zeros_like(t_counts, dtype=float)
        N_sa = np.sum(t_counts, axis=2, keepdims=True)
        N_sa_safe = np.where(N_sa == 0, 1, N_sa)
        T = t_counts / N_sa_safe
        T *= (N_sa != 0)
        self.t_counts = t_counts
        self.T = T
    
    # MLE estimate for R
    def estimate_rewards(self):
        r_sums = np.zeros((self.num_states, self.num_actions), dtype=float)
        np.add.at(r_sums, (self.states, self.actions), self.rewards)
        N_sa = np.sum(self.t_counts, axis=2)
        N_sa_safe = np.where(N_sa == 0, 1, N_sa)
        R = r_sums / N_sa_safe
        R[N_sa == 0] = 0
        self.R = R
    
    def value_iteration(self, gamma=0.95, max_iters=10_000, theta=1e-6):
        self.estimate_transition_probs()
        self.estimate_rewards()
        V = np.zeros(self.num_states)

        for i in tqdm(range(max_iters)):
            V_prev = V.copy()

            expected_V = np.sum(self.T * V[np.newaxis, np.newaxis, :], axis=2)  
            action_values = self.R + gamma * expected_V 
            V = np.max(action_values, axis=1)

            delta = np.max(np.abs(V - V_prev))

            if (i+1) % 100 == 0 or i == 0:
                print(f"Iteration {i+1}: ΔV = {delta:.6f}")

            if delta < theta:
                print(f"Value Iteration converged after {i+1} iterations.")
                break
            
        policy = np.argmax(action_values, axis=1) + 1
        return V, policy