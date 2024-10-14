import numpy as np
import random
from bayes_network import BayesNetwork
import networkx as nx
from k2_search import  K2Search, mutual_information_ordering, perturb_ordering
import pickle
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def randint_exclude(low, high, exclude):
    while True:
        num = np.random.randint(low, high)
        if num != exclude:
            return num
    
class StochasticLocalSearch(BayesNetwork):
    def __init__(self, x, G=None, initial_temperature=1.0, cooling_rate=0.99, max_iter=1000, restart_threshold=100, max_parents=3):
        super().__init__(x)
        self.max_iter = max_iter
        self.init_temp = initial_temperature
        self.cooling_rate = cooling_rate
        self.restart_threshold = restart_threshold
        self.max_parents = max_parents
        self.cnt_restart = 0

        self.searches = {} # iter -> Graph at that stage
        if G is not None:
            self.G = G.copy()
        
    def rand_graph_neighbor(self):
        bn = self.copy()
        while True:
            i = np.random.randint(0, self.n)
            j = randint_exclude(0, self.n, i)
            if bn.G.has_edge(i, j):
                bn.G.remove_edge(i, j)
                return bn
            
            elif not nx.has_path(bn.G, j, i):
                if len(list(self.G.predecessors(i))) < self.max_parents:
                    bn.G.add_edge(i, j)
                    return bn
    
    def restart_G(self, iter):
        dy_vals = [v[1] for _, v in self.searches.items()]
        y_vals = [v[2] for _, v in self.searches.items()]

        if len(self.searches.keys()) == 1 or \
                max(y_vals) == self.searches[-1][2] or \
                max(dy_vals) == 0:
            # restart to initial value
            G, _, y = self.searches[-1]
            return G, y

        self.cnt_restart += 1
        while True:
            key = random.choice(list(self.searches.keys()))
            if iter == key:
                continue
            
            G, d_y, y = self.searches[key]
            if d_y > 0:
                return G, y

    def fit(self):
        y = self.bayesian_score()
        y_max, G_max = y, self.G.copy()

        self.searches[-1] = (self.G.copy(), 0, y)

        temp = self.init_temp
        cnt_trials = 0 # number of iterations since last update
        for iter in range(self.max_iter):
            bn_neighbor = self.rand_graph_neighbor()
            y_neighbor = bn_neighbor.bayesian_score()
            d_y = y_neighbor - y
            if d_y > 0:
                self.searches[iter] = (self.G.copy(), d_y, y_neighbor)
                y = y_neighbor
                self.G = bn_neighbor.G.copy()
                cnt_trials = 0
            else:
                threshold = np.exp(d_y / temp)
                if np.random.uniform(0, 1) < threshold:
                    self.searches[iter] = (self.G.copy(), d_y, y_neighbor)
                    y = y_neighbor
                    self.G = bn_neighbor.G.copy()
                    cnt_trials = 0
                else:
                    cnt_trials += 1
            
            temp *= self.cooling_rate
            
            # Randomized restart
            if cnt_trials == self.restart_threshold:
                if y > y_max:
                    G_max = self.G.copy()
                    y_max = y
                
                G_restart, y_restart = self.restart_G(iter)
                self.G = G_restart.copy()
                y = y_restart
                cnt_trials = 0
                temp = self.init_temp

        if self.cnt_restart == 0:
            G_max = self.G.copy()
            return self.bayesian_score()

        self.G = G_max.copy()
        return y_max


def dump_best_network(graph, M, name):
    with open(f'pickles/bootstrap_{M}_{name}.pkl', 'wb') as f:
        pickle.dump(graph, f)

def generate_ordering(x):
    x_sample = x[np.random.choice(x.shape[0], x.shape[0], replace=True)]
    mi_rank = mutual_information_ordering(x_sample)
    rank_perturbed = perturb_ordering(mi_rank)
    return tuple(rank_perturbed)

def process_ordering(args):
    x, ordering = args
    k2 = K2Search(x, ordering=ordering)
    k2_score = k2.fit(max_parents=2)
    local_search = StochasticLocalSearch(x, k2.G)
    local_search_score = local_search.fit()
    return (k2.G.copy(), k2_score), (local_search.G.copy(), local_search_score)

def boostrap_fit(x, M):
    # M number of bootstrap iters(tries, most do not result in unique ordering)
    # resample x -> x_sample
    # for each x_sample, get mutual information rank of variables
    # for all unique mutual_information ranks, run k2 with that rank
    # and then start local search from the graph generated from k2
    num_cores = multiprocessing.cpu_count()
    logging.info(f"Running bootstrap localsearch fit, M = {M}")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        orderings = list(executor.map(generate_ordering, [x] * M))
    ordering_ls = set(orderings)
    
    logging.info(f'Number of unique variable orders: {len(ordering_ls)}')
    args_list = [(x, o) for o in ordering_ls]
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_ordering, args_list))
    
    k2_networks_out, local_search_networks_out = zip(*results)
    k2_max = max(k2_networks_out, key=lambda x: x[1])
    l_max = max(local_search_networks_out, key=lambda x: x[1])
    
    logging.info(f'K2 Max: {k2_max[1]}')
    logging.info(f'Localsearch Max: {l_max[1]}')
    
    dump_best_network(k2_max[0], M, f"k2_({round(k2_max[1], 2)})")
    dump_best_network(l_max[0], M, f"local_search_({round(l_max[1], 2)})")
    return k2_max, l_max