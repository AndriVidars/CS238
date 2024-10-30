import utils
from small import fit_T
from collections import defaultdict


if __name__ == '__main__':
    data = utils.read_data('medium')
    max_r = max(data[:, 2])
    S = 50000
    A = 7
    T = fit_T(data, S, A)
    len_dict = defaultdict(int)
    for k, v in T.items():
        len_dict[len(v)] += 1



    action_to_states = defaultdict(set)

    for row in data:
        s = row[0]  # s
        a = row[1]  # a
        action_to_states[a].add(s)

    
    action_to_states = {a: sorted(list(states)) for a, states in action_to_states.items()}
    print()