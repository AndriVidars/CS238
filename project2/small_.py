import utils
import numpy as np
from value_iteration import ValueIteration

   
if __name__ == "__main__":
    data = utils.read_data('small').astype(int)
    states = data[:, 0] - 1
    actions = data[:, 1] - 1
    rewards = data[:, 2]
    next_states = data[:, 3] - 1

    val_iter = ValueIteration(states, actions, rewards, next_states, num_states=100, num_actions=4)
    v, policy = val_iter.value_iteration()

    utils.write_policy('small', policy)
