import numpy as np
from tqdm import tqdm
from collections import defaultdict
import utils
import random

def split_state(state):
    state_str = f"{state:06d}"
    return state_str[:2], state_str[2:4], state_str[4:]

def prepare_data_np():
    data = utils.read_data('large').astype(int)
    np.random.shuffle(data)
    states = data[:, 0]
    actions = data[:, 1] - 1 
    rewards = data[:, 2]
    next_states = data[:,3]
    
    state_dict = {i: sorted(set(split_state(s)[i] for s in states)) for i in range(3)}
    index_map = [{x: i for i, x in enumerate(state_dict[i])} for i in range(3)]
    rev_index_map = [{v: k for k, v in d.items()} for d in index_map]

    states_segments = np.array([tuple(index_map[i][x] for i, x in enumerate(split_state(state))) for state in states])
    next_states_segments = np.array([tuple(index_map[i][x] for i, x in enumerate(split_state(state))) for state in next_states])

    return states_segments, next_states_segments, actions, rewards, state_dict, index_map, rev_index_map

def get_state_idx(state_segmented):
    return sum(x*10**(2-i) for i, x in enumerate(state_segmented))

def get_state_segments(idx):
    return (
        idx // 100,
        (idx % 100) // 10,
        idx % 10
    )

def get_transition_count_matrix(states_segments, actions, next_states_segments):
    t_counts = np.zeros((500, 9, 500), dtype=int)
    
    idx_s = np.array([get_state_idx(state) for state in states_segments])
    idx_ns = np.array([get_state_idx(state) for state in next_states_segments])
    
    np.add.at(t_counts, (idx_s, actions, idx_ns), 1)
    return t_counts

def estimate_transition_probs(t_counts):
    T = np.zeros_like(t_counts, dtype=float)
    N_sa = np.sum(t_counts, axis=2, keepdims=True)
    N_sa_safe = np.where(N_sa == 0, 1, N_sa)
    T = t_counts / N_sa_safe
    T *= (N_sa != 0)
    
    return T

def get_reward_matrix(states_segments, actions, rewards):
    r_sums = np.zeros((500, 9), dtype=float)
    idx_s = np.array([get_state_idx(state) for state in states_segments])
    np.add.at(r_sums, (idx_s, actions), rewards)
    return r_sums

def estimate_rewards(r_sums, t_counts):
    N_sa = np.sum(t_counts, axis=2)
    N_sa_safe = np.where(N_sa == 0, 1, N_sa)
    R = r_sums / N_sa_safe
    R[N_sa == 0] = 0
    return R

def value_iteration(T, R, gamma=0.95, max_iters=10_000, theta=1e-6):
    V = np.zeros(R.shape[0])
    
    for i in tqdm(range(max_iters)):
        V_prev = V.copy()
        
        expected_V = np.sum(T * V[np.newaxis, np.newaxis, :], axis=2)  
        action_values = R + gamma * expected_V 
        V = np.max(action_values, axis=1)
        
        delta = np.max(np.abs(V - V_prev))
        
        if (i+1) % 100 == 0 or i == 0:
            print(f"Iteration {i+1}: ΔV = {delta:.6f}")
        
        if delta < theta:
            print(f"Value Iteration converged after {i+1} iterations.")
            break
    
    policy = np.argmax(action_values, axis=1) + 1
    return V, policy

def write_policy_(policy, rev_index_map):
    n_states_out = 302020
    random.seed(23)
    rand = random.randint(1, 9)
    policy_out = [rand for _ in range(n_states_out)]

    for j in range(policy.shape[0]):
        segment_idx = get_state_segments(j)
        state_str = ""
        for i, x in enumerate(segment_idx):
            state_str += rev_index_map[i][x]
        
        state_idx = int(state_str) - 1
        policy_out[state_idx] = policy[j]
    
    utils.write_policy('large', policy_out)

if __name__ == '__main__':
    states_segments, next_states_segments, actions, \
         rewards, state_dict, index_map, rev_index_map \
         = prepare_data_np()
    
    t_counts = get_transition_count_matrix(states_segments, actions, next_states_segments)
    r_sums = get_reward_matrix(states_segments, actions, rewards)
    T = estimate_transition_probs(t_counts)
    R = estimate_rewards(r_sums, t_counts)

    # todo, do value iteration on this stuff
    print(T.shape, R.shape)

    _, policy = value_iteration(T, R)
    write_policy_(policy, rev_index_map)
