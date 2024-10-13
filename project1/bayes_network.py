import numpy as np
import networkx as nx
from functools import lru_cache
from scipy.special import loggamma
from itertools import product

class BayesNetwork:
    def __init__(self, x, G=None):
        self.x = x
        self.x_values_range = np.max(x, axis=0)
        self.n = x.shape[1]
        self.n_obs = x.shape[0]

        if G == None:
            self.G = nx.DiGraph()
            self.G.add_nodes_from(range(self.n))
        else:
            self.G = G

        self.value_index_map = {} # node: {value: ndarray(bool)} # True if x[:, index] == value
        for i in range(self.n):
            self.value_index_map[i] = {}
            for j in range(1, self.x_values_range[i] + 1):
                self.value_index_map[i][j] = (self.x[:, i] == j)
    
    def copy(self):
        new_bn = BayesNetwork(self.x)
        new_bn.G = self.G.copy()
        new_bn.value_index_map = {k: v.copy() for k, v in self.value_index_map.items()}
        return new_bn

    # TODO: will it become necessary to encode/enumerate j
    @lru_cache(maxsize=None)
    def m(self, i, j):
        # node i
        # j, enumeration of parent node instantiation. represented with tuple of tuples ((node, val))         
        idx = np.full(self.n_obs, True)
        if j:
            for parent_node, parent_node_val in j:
                idx = np.logical_and(idx, self.value_index_map[parent_node][parent_node_val])

        i_vals = (self.x[idx])[:,i]
        m_ijk_ = np.array([np.sum(i_vals == k) for k in range(1, self.x_values_range[i] + 1)])
        m_ij0 = np.sum(idx)
        return m_ijk_, m_ij0
    
    @lru_cache(maxsize=None)
    def get_parent_instantiations(self, parents):        
        parents_vals = [list(range(1, self.x_values_range[parent] + 1)) for parent in parents]
        product_vals = product(*parents_vals)
        return [tuple(zip(parents, values)) for values in product_vals]
    
    def bayesian_score(self):
        p = 0
        for i in range(self.n):
            parents = tuple(self.G.predecessors(i))
                        
            q = self.get_parent_instantiations(parents) if parents else [()] # case where node has no parents
            r = self.x_values_range[i]
            for j in q:
                m_ijk, m_ij0 = self.m(i, j)
                alpha_ij0 = r # uniform prior, each alpha_ijk has value 1
                p += (loggamma(alpha_ij0) - loggamma(alpha_ij0 + m_ij0))
                for k in range(r): # shifted for 0-indexing
                    p += loggamma(1 + m_ijk[k]) # uniform prior, denominator term eliminated
        
        return p

# somewhat modified mutual_information       
def mutual_information(data, constraints=None):
    # constraints: {node: [lowest(list of forced lowest mi rank in order), highest()]}

    bayes_net = BayesNetwork(data)
    x_values_range = bayes_net.x_values_range
    n = bayes_net.n
    n_obs = bayes_net.n_obs

    value_index_map = bayes_net.value_index_map
    mi_matrix = np.zeros((n, n))
    
    def pairwise_mi(i, j):
        mi = 0
        for x_i in range(1, x_values_range[i] + 1):
            idx_i = value_index_map[i][x_i]
            p_xi = np.count_nonzero(idx_i) / n_obs
            for x_j in range(1, x_values_range[j] + 1):
                idx_j = value_index_map[j][x_j]
                p_xj = np.count_nonzero(idx_j) / n_obs
                idx_joint = np.logical_and(idx_i, idx_j)
                p_joint = np.count_nonzero(idx_joint) / n_obs
                if p_joint > 0:
                    mi += p_joint * (np.log(p_joint) - (np.log(p_xi) + np.log(p_xj)))
        return mi

    for i in range(n):
        for j in range(i+1, n):
            mi_matrix[i, j] = mi_matrix[j, i] = pairwise_mi(i, j)
    
    if constraints:
        mi_max_copy = mi_matrix.copy()
        mi_min_copy = mi_matrix.copy()
        np.fill_diagonal(mi_max_copy, -np.inf)
        np.fill_diagonal(mi_min_copy, np.inf)
        for i in range(n):
            if i in constraints.keys():
                std = np.std(mi_matrix[i, :]) # use this or some other

                # note this makes the mi_matrix unsymmetric, but it should be like that
                if 'h' in constraints[i].keys():
                    max_i = np.max(mi_max_copy[i, :]) + std
                    for n in reversed(constraints[i]['h']):
                        mi_matrix[i][n] = max_i
                        max_i += std
                
                if 'l' in constraints[i].keys():
                    min_i = np.min(mi_min_copy[i, :]) - std
                    for n in reversed(constraints[i]['l']):
                        mi_matrix[i][n] = min_i
                        min_i -= std
                    
    return mi_matrix