import numpy as np
from bayes_network import BayesNetwork, mutual_information
import random

class K2Search(BayesNetwork):
    def __init__(self, x, ordering=None):
        super().__init__(x)
        if ordering is not None:
            self.ordering = ordering
        else:
            self.ordering = range(self.n)

    def fit(self, max_parents = 2):
        max_parents=min(10, self.n - 2) # maybe comment out
        y = self.bayesian_score() # starting bayesian score
        
        for k, i in enumerate(self.ordering[1:], start=1):
            parents_curr = list(self.G.predecessors(i))
            while True:
                if len(parents_curr) == max_parents:
                    break

                delta_max, j_best = 0, None
                for j in self.ordering[:k]:
                    if not self.G.has_edge(j, i):
                        parents_next = parents_curr + [j]
                        y_delta = self.bayesian_score_delta(i, tuple(sorted(parents_curr)), tuple(sorted(parents_next)))
                        if y_delta > delta_max:
                            delta_max = y_delta
                            j_best = j
                
                if delta_max > 0:
                    self.G.add_edge(j_best, i)
                    parents_curr.append(j_best)
                    y += delta_max
                else:
                    break 
        return y
    
def mutual_information_ordering(data):
    n = data.shape[1]
    #mi_constraints = {i:{'l': [n-1]} for i in range(n - 1)} # force last column to have lowest rank ("response variable")
    mi_matrix = mutual_information(data)
    mi = []
    for i in range(n - 1):
        mi_total = sum(mi_matrix[i, j] for j in range(n) if i != j)
        mi.append(mi_total)

    mi_ordering = list(np.argsort(-np.array(mi))) + [n - 1]
    return mi_ordering

def perturb_ordering(ordering, swap_prob=0.05, max_swaps=-1):
    if max_swaps == -1:
        max_swaps = int(len(ordering)*(3/4))

    for _ in range(max_swaps):
        if random.random() < swap_prob:
            i = random.randint(0, len(ordering) - 2)
            ordering[i], ordering[i+1] = ordering[i+1], ordering[i]
    return ordering