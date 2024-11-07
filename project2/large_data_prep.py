import numpy as np
import utils
from tqdm import tqdm
from collections import defaultdict
from itertools import product

def split_state(state):
    state_str = f"{state:06d}"
    return state_str[:2], state_str[2:4], state_str[4:]

def one_hot_encode(array, num_classes):
    return np.eye(num_classes)[array]

def decode_one_hot(encoded_array, num_classes):
    return np.argmax(encoded_array.reshape(-1, num_classes), axis=1)


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

    n_values = [len(state_dict[i]) for i in range(3)]
    n_actions = len(set(actions))

    states_encoded = np.hstack([one_hot_encode(states_segments[:, i], n_values[i]) for i in range(3)])
    next_states_encoded = np.hstack([one_hot_encode(next_states_segments[:, i], n_values[i]) for i in range(3)])
    actions_encoded = one_hot_encode(actions, n_actions)

    def decode_state(one_hot_encoded_state):
        segment_1 = decode_one_hot(one_hot_encoded_state[:, :n_values[0]], n_values[0])
        segment_2 = decode_one_hot(one_hot_encoded_state[:, n_values[0]:n_values[0] + n_values[1]], n_values[1])
        segment_3 = decode_one_hot(one_hot_encoded_state[:, n_values[0] + n_values[1]:], n_values[2])

        original_state_segments = np.array([
            int(rev_index_map[0][s1] + rev_index_map[1][s2] + rev_index_map[2][s3])
            for s1, s2, s3 in zip(segment_1, segment_2, segment_3)
        ])

        return original_state_segments

    def decode_action(one_hot_encoded_action):
        return decode_one_hot(one_hot_encoded_action, n_actions)
    
    def one_hot_encode_single(value, num_classes):
        encoded = np.zeros(num_classes, dtype=int)
        encoded[value] = 1
        return encoded

    all_combinations = list(product(range(n_values[0]), range(n_values[1]), range(n_values[2])))
    all_one_hot_states = []
    for combination in all_combinations:
        encoded_segments = [
            one_hot_encode_single(combination[i], n_values[i]) for i in range(3)
        ]
        full_one_hot_state = np.hstack(encoded_segments)
        all_one_hot_states.append(full_one_hot_state)

    all_one_hot_states = np.array(all_one_hot_states)
    all_one_hot_actions = np.eye(n_actions)

    return states_encoded, next_states_encoded, actions_encoded, rewards,\
        all_one_hot_states, all_one_hot_actions, decode_state, decode_action,\
        index_map, rev_index_map
