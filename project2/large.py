import numpy as np
import utils
import random
from value_iteration import ValueIteration
import time

def split_state(state):
    state_str = f"{state:06d}"
    return state_str[:2], state_str[2:4], state_str[4:]

# this data prep stuff is overly complicated
# much of it is leftover from some attempt
# at solving problem with deep Q learning
# could have just enumerated each of the 500 states

def prepare_data_np():
    data = utils.read_data('large').astype(int)
    np.random.shuffle(data)
    states = data[:, 0]
    actions = data[:, 1] - 1 
    rewards = data[:, 2]
    next_states = data[:, 3]
    
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

def write_policy_(policy, rev_index_map):
    n_states_out = 302020
    random.seed(23)
    rand = random.randint(1, 9) # random action for "fake" states
    policy_out = [rand for _ in range(n_states_out)]

    # write policy for actual states
    for j in range(policy.shape[0]):
        segment_idx = get_state_segments(j)
        state_str = ""
        for i, x in enumerate(segment_idx):
            state_str += rev_index_map[i][x]
        
        state_idx = int(state_str) - 1
        policy_out[state_idx] = policy[j]
    
    utils.write_policy('large', policy_out)

if __name__ == '__main__':
    start_time = time.time()
    states_segments, next_states_segments, actions, \
    rewards, state_dict, index_map, rev_index_map = prepare_data_np()

    states = np.array([get_state_idx(state) for state in states_segments])
    next_states = np.array([get_state_idx(state) for state in next_states_segments])

    val_iter = ValueIteration(states, actions, rewards, next_states, num_states=500, num_actions=9)
    _, policy = val_iter.value_iteration()
    write_policy_(policy, rev_index_map)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    