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
    def __init__(self, x, G=None, restart_Gs=None, initial_temperature=1.0, cooling_rate=0.99, max_iter=1000, restart_threshold=100, max_parents=3):
        super().__init__(x)
        self.max_iter = max_iter
        self.init_temp = initial_temperature
        self.cooling_rate = cooling_rate
        self.restart_threshold = restart_threshold
        self.max_parents = max_parents
        self.cnt_restart = 0

        if G is not None:
            self.G = G.copy()
        
        # set of networks to init from in randomized restart
        if restart_Gs is None:
            self.restart_Gs = [G.copy()]
        else:
            self.restart_Gs = restart_Gs
        
    def rand_graph_neighbor(self):
        # return (i, j, action, parents_curr, parents_next) # i -> j edge
        while True:
            i = np.random.randint(0, self.n)
            j = randint_exclude(0, self.n, i)
            parents_curr = list(self.G.predecessors(j))
            
            if self.G.has_edge(i, j):
                parents_next = [x for x in parents_curr if x != i]
                return (i, j, 'r', tuple(sorted(parents_curr)), tuple(sorted(parents_next)))
            
            elif not nx.has_path(self.G, j, i):
                if len(parents_curr) < self.max_parents:
                    parents_next = parents_curr + [i]
                    return (i, j, 'a', tuple(sorted(parents_curr)), tuple(sorted(parents_next)))
    
    def restart_G(self):
        self.cnt_restart += 1
        idx = random.randint(0, len(self.restart_Gs) - 1)
        restart_G = self.restart_Gs[idx].copy()
        bn = BayesNetwork(self.x, restart_G)
        y = bn.bayesian_score()
        return restart_G, y
        

    def fit(self):
        y = self.bayesian_score()
        y_max, G_max = y, self.G.copy()

        temp = self.init_temp
        cnt_trials = 0 # number of iterations since last update
        for _ in range(self.max_iter):
            i, j, action, parents_curr, parents_next = self.rand_graph_neighbor()
            delta_y = self.bayesian_score_delta(j, parents_curr, parents_next)
            if delta_y > 0:
                cnt_trials = 0
                y += delta_y
                if action == 'a':
                    # add edge
                    self.G.add_edge(i, j)
                elif action == 'r':
                    # remove edge
                    self.G.remove_edge(i, j)
            else:
                threshold = np.exp(delta_y / temp)
                if random.random() < threshold:
                    cnt_trials = 0
                    y += delta_y
                    if action == 'a':
                        # add edge
                        self.G.add_edge(i, j)
                    elif action == 'r':
                        # remove edge
                        self.G.remove_edge(i, j)
                else:
                    cnt_trials += 1
            
            temp *= self.cooling_rate
                            
            # Randomized restart
            if cnt_trials == self.restart_threshold:
                if y > y_max:
                    G_max = self.G.copy()
                    y_max = y
                
                self.G, y = self.restart_G()
                cnt_trials = 0
                temp = self.init_temp

        if self.cnt_restart == 0:
            if y > y_max:
                G_max = self.G.copy()
                y_max = y

        self.G = G_max.copy()
        return y_max


def dump_best_network(graph, M, name):
    with open(f'pickles/bootstrap_{M}_{name}.pkl', 'wb') as f:
        pickle.dump(graph, f)

def generate_ordering(x, random_prob=0.25):
    if random.random() < random_prob:
        vals = list(range(x.shape[1]))
        random.shuffle(vals)
        return tuple(vals)

    x_sample = x[np.random.choice(x.shape[0], x.shape[0], replace=True)]
    mi_rank = mutual_information_ordering(x_sample)
    rank_perturbed = perturb_ordering(mi_rank)
    return tuple(rank_perturbed)

def process_ordering(args):
    x, ordering = args
    k2 = K2Search(x, ordering=ordering)
    k2_score = k2.fit() # change to 3?
    local_search = StochasticLocalSearch(x, k2.G, max_iter=1000, max_parents=min(x.shape[1] - 2, 12)) # overdoing it for large/medium? # TODO change max iter
    local_search_score = local_search.fit()
    return (k2.G.copy(), k2_score), (local_search.G.copy(), local_search_score)

def boostrap_fit(x, M):
    # M number of bootstrap iters(tries, most do not result in unique ordering)
    # resample x -> x_sample
    # for each x_sample, get mutual information rank of variables
    # for all unique mutual_information ranks, run k2 with that rank
    # and then start local search from the graph generated from k2
    num_cores = multiprocessing.cpu_count()
    logging.info(f"Running bootstrap localsearch fit, M = {M}, num_cores = {num_cores}")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        orderings = list(executor.map(generate_ordering, [x] * M))
    ordering_ls = set(orderings)
    
    logging.info(f'Number of unique variable orders: {len(ordering_ls)}')
    args_list = [(x, o) for o in ordering_ls]
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_ordering, args_list))
    
    k2_networks_out, local_search_networks_out = zip(*results)
    k2_networks_out = sorted(k2_networks_out, key=lambda x: x[1], reverse=True)
    local_search_networks_out = sorted(local_search_networks_out, key=lambda x: x[1], reverse=True)

    k2_max = k2_networks_out[0][1]
    l_max = local_search_networks_out[0][1]
    
    logging.info(f'K2 Max: {k2_max}')
    logging.info(f'Localsearch Max: {l_max}')
    
    dump_best_network(k2_networks_out[0][0], M, f"k2_({round(k2_max, 2)})")
    dump_best_network(local_search_networks_out[0][0], M, f"local_search_({round(l_max, 2)})")


    return k2_networks_out, local_search_networks_out