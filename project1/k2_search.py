import numpy as np
from bayes_network import BayesNetwork, mutual_information
import random

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

def perturb_ordering(ordering, swap_prob=0.25, max_swaps=-1):
    if max_swaps == -1:
        max_swaps = int(len(ordering)*(3/4))

    for _ in range(max_swaps):
        if random.random() < swap_prob:
            i = random.randint(0, len(ordering) - 2)
            ordering[i], ordering[i+1] = ordering[i+1], ordering[i]
    return ordering


class K2Search(BayesNetwork):
    def __init__(self, x, ordering=None):
        super().__init__(x)
        if ordering is not None:
            self.ordering = ordering
        else:
            self.ordering = range(self.n)

    def fit(self, max_parents=2):
        for k, i in enumerate(self.ordering[1:], start=1):
            y = self.bayesian_score()
            while True:
                if len(list(self.G.predecessors(i))) == max_parents:
                    break

                y_best, j_best = float('-inf'), None
                for j in self.ordering[:k]:
                    if not self.G.has_edge(j, i):
                        self.G.add_edge(j, i)
                        y_ = self.bayesian_score()
                        self.G.remove_edge(j, i)
                        if y_ > y_best:
                            y_best, j_best = y_, j
                
                if y_best > y:
                    self.G.add_edge(j_best, i)
                    y = y_best
                else:
                    break 
        return y