import numpy as np

def read_data(file):
    return np.loadtxt(f'data/{file}.csv', delimiter=',', skiprows=1)

def write_policy(file, policy):
    n = len(policy) - 1
    out = [f'{x}\n' for x in policy[:-1]] + [str(policy[-1])]

    with open(f'output/{file}.policy', 'w') as f:
        f.writelines(out)