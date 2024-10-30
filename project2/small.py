import utils
import numpy as np
from collections import defaultdict
import random

# MLE for Transition probabilites from observed data
# returns dict (state, action) -> prob dist of next state
def fit_T(data, S, A):
    T = defaultdict(list)
    for s in range(1, S+1):
        data_s = data[data[:, 0] == s]
        for a in range(1, A+1):
            action_data = data_s[data_s[:, 1] == a]
            if action_data.size > 0:
                next_states, counts = np.unique(action_data[:, -1].astype(int), return_counts=True)
                probabilities = counts / counts.sum()
                T[(s, a)] = list(zip(probabilities, next_states))
    return T

def fit_R(data, S, A):
    R = np.zeros((S, A))
    for s in range(1, S+1):
        for a in range(1, A+1):
            R[s-1, a-1] = np.unique(data[(data[:, 0] == s) & (data[:, 1] == a), 2])[0]
            if R[s-1, a-1] != 0:
                print(s, a, R[s-1, a-1])

    return R


def init_random_policy(S, A):
    return np.array([random.randint(1, A) for _ in range(S)])

class PolicyIteration:
    def __init__(self, data, discount=0.95):
        self.data = data
        self.discount = 0.95
        self.S = int(max(data[:, 0])) # here we are assuming 
        self.A = int(max(data[:, 1])) # number of actions
        self.T = fit_T(data, self.S, self.A)
        self.R = fit_R(data, self.S, self.A)
        self.policy = init_random_policy(self.S, self.A)
        self.T_pi = self.T_policy()
        self.R_pi = self.R_policy()
        self.U_pi = None

    def T_policy(self):
        T_pi = np.zeros((self.S, self.S))
        for s in range(1, self.S+1):
            action = self.policy[s-1]
            probs = self.T[s, action]
            for p, sp in probs:
                T_pi[s-1, sp-1] = p
        
        return T_pi

    def R_policy(self):
        R_pi = np.zeros(self.S).reshape(-1, 1) 
        for s in range(1, self.S+1):
            action = self.policy[s-1]
            R_pi[s-1] = self.R[s-1, action-1]

        return R_pi

    def policy_eval(self):
        self.U_pi = np.linalg.solve(np.identity(self.S) - self.discount * self.T_pi, self.R_pi)
    
    def iter(self):
        n_iters = 0
        while True:
            self.policy_eval()  # Evaluate the current policy
            new_policy = np.copy(self.policy)
            
            for s in range(1, self.S+1):
                max_action_value = -float('inf')
                max_action = None
                
                for a in range(1, self.A+1):
                    action_value = self.R[s-1, a-1] + self.discount * \
                        sum(p * self.U_pi[sp-1] for p, sp in self.T[s, a])
                    
                    if action_value > max_action_value:
                        max_action_value = action_value
                        max_action = a
                
                if max_action:
                    new_policy[s-1] = max_action

            n_iters += 1

            if np.linalg.norm(self.policy - new_policy) < 1e-10:
                break
            
            #if np.allclose(self.policy, new_policy):
            #   break
            
            self.policy = new_policy
            self.T_pi = self.T_policy()
            self.R_pi = self.R_policy()
            print(f'sum U, {sum(self.U_pi)}')
        
        print(f'Number of iterations: {n_iters}')
        self.policy_eval()
        print(f'Final sum U, {sum(self.U_pi)}')


if __name__ == "__main__":
    # state, action, reward, next_state
    data = utils.read_data('small')
    pol_fit = PolicyIteration(data)
    pol_fit.iter()
    utils.write_policy('small', pol_fit.policy)
