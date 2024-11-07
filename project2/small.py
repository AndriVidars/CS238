import utils
from value_iteration import ValueIteration
import time
   
if __name__ == "__main__":
    start_time = time.time()
    data = utils.read_data('small').astype(int)
    states = data[:, 0] - 1
    actions = data[:, 1] - 1
    rewards = data[:, 2]
    next_states = data[:, 3] - 1

    val_iter = ValueIteration(states, actions, rewards, next_states, num_states=100, num_actions=4)
    _, policy = val_iter.value_iteration()

    utils.write_policy('small', policy)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
