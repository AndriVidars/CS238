import numpy as np
import networkx as nx
from scipy.special import loggamma

class BayesNetwork:
    def __init__(self, x, G=None, bayesian_score_cache=None):
        self.x = x
        self.x_values_range = np.max(x, axis=0)
        self.n = x.shape[1]
        self.n_obs = x.shape[0]

        if G == None:
            self.G = nx.DiGraph()
            self.G.add_nodes_from(range(self.n))
        else:
            self.G = G
        
        self.bayesian_score_cache = bayesian_score_cache if bayesian_score_cache is not None else {}

        self.value_index_array = np.zeros((self.n, np.max(self.x_values_range), self.n_obs), dtype=bool)
        for i in range(self.n):
            for j in range(1, self.x_values_range[i] + 1):
                self.value_index_array[i, j - 1] = (self.x[:, i] == j)

    def copy(self):
        new_bn = BayesNetwork(self.x)
        new_bn.G = self.G.copy()
        new_bn.value_index_array = self.value_index_array
        return new_bn
    
    def baysian_score_component(self, i, parents):
        # i: node
        # parents: tuple of parent nodes
        if (i, parents) in self.bayesian_score_cache:
            return self.bayesian_score_cache[(i, parents)]

        p = 0
        r_i = self.x_values_range[i]
        alpha_ijk = 1
        alpha_ij0 = r_i * alpha_ijk

        if parents:
            parent_data = self.x[:, parents]
            unique_parents_values, inverse_indices = np.unique(
                parent_data, axis=0, return_inverse=True
            )

            n_unique_combinations = unique_parents_values.shape[0]
            m_ijk = np.zeros((n_unique_combinations, r_i), dtype=int)
            
            values_of_i = self.x[:, i] - 1
            np.add.at(m_ijk, (inverse_indices, values_of_i), 1)
            m_ij0 = m_ijk.sum(axis=1)
            p += np.sum(
                loggamma(alpha_ij0) - loggamma(alpha_ij0 + m_ij0) +
                np.sum(loggamma(alpha_ijk + m_ijk), axis=1)
            )

        else:
            counts = np.bincount(self.x[:, i] - 1, minlength=r_i)
            m_ijk = counts
            m_ij0 = counts.sum()
            p += loggamma(alpha_ij0) - loggamma(alpha_ij0 + m_ij0)
            p += np.sum(loggamma(alpha_ijk + m_ijk))
        
        self.bayesian_score_cache[(i, parents)] = p
        return p
    
    def bayesian_score_delta(self, node, parents_curr, parents_next):
        # change in bayesian score by adding/dropping edges
        # that is change the set of parents of a node
        # parents_curr: current parent nodes of node
        # parents_next: potential next set of parents after adding/dropping an edge
        return self.baysian_score_component(node, parents_next) - self.baysian_score_component(node, parents_curr)
        
    def bayesian_score(self):
        p = 0
        for i in range(self.n):
            parents = tuple(sorted(self.G.predecessors(i)))
            p += self.baysian_score_component(i, parents)

        return p

    
def mutual_information(data, constraints=None):
    # compute mutual information between all variables/nodes
    # somewhat modified mutual_information   
    # constraints: {node: [lowest(list of forced lowest mi rank in order), highest()]}
    bayes_net = BayesNetwork(data)
    x_values_range = bayes_net.x_values_range
    n = bayes_net.n
    n_obs = bayes_net.n_obs

    value_index_array = bayes_net.value_index_array
    mi_matrix = np.zeros((n, n))
    
    def pairwise_mi(i, j):
        mi = 0
        for x_i in range(1, x_values_range[i] + 1):
            idx_i = value_index_array[i, x_i - 1]
            p_xi = np.count_nonzero(idx_i) / n_obs
            for x_j in range(1, x_values_range[j] + 1):
                idx_j = value_index_array[j, x_j - 1]
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