import sys
import numpy as np
import networkx as nx
from functools import lru_cache
from scipy.special import loggamma
from itertools import product

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def read_csv_to_array(infile):
    with open(f'data/{infile}', 'r') as f:
        header = [s.strip('"') for s in f.readline().strip().split(',')]
        x = np.genfromtxt(f, delimiter=',', skip_header=0,dtype='int') # already read line 0
    
    return x, header

# heueristic for ordering in k2-search
import numpy as np

class MutualInformation:
    def __init__(self, x):
        self.x = x
        self.x_values_range = np.max(x, axis=0)
        self.n = x.shape[1]
        self.n_obs = x.shape[0]
        self.value_index_map = {}
        for i in range(self.n):
            self.value_index_map[i] = {}
            for j in range(1, self.x_values_range[i] + 1):
                self.value_index_map[i][j] = (self.x[:, i] == j)
    
    def pairwise_mi(self, i, j):
        mi = 0
        for x_i in range(1, self.x_values_range[i] + 1):
            idx_i = self.value_index_map[i][x_i]
            p_xi = np.count_nonzero(idx_i) / self.n_obs
            for x_j in range(1, self.x_values_range[j] + 1):
                idx_j = self.value_index_map[j][x_j] 
                p_xj = np.count_nonzero(idx_j) / self.n_obs
                idx_joint = np.logical_and(idx_i, idx_j)
                p_joint = np.count_nonzero(idx_joint) / self.n_obs
                if p_joint > 0:
                    mi += p_joint * (np.log(p_joint) - (np.log(p_xi) + np.log(p_xj)))
        return mi
    
    def mi_rank(self):
        mi = []
        for i in range(self.n):
            mi_total = sum(self.pairwise_mi(i, j) for j in range(self.n) if i != j)
            mi.append(mi_total)
        return list(np.argsort(-np.array(mi))), mi  # Sort in descending order

class K2Search:
    def __init__(self, x, ordering=None):
        self.x = x
        self.x_values_range = np.max(x, axis=0)
        self.n = x.shape[1]
        self.n_obs = x.shape[0]
    
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.n))
        
        if ordering is not None:
            self.ordering = ordering
        else:
            self.ordering = range(self.n)

        self.value_index_map = {} # node: {value: ndarray(bool)} # True if x[:, index] == value
        for i in range(self.n):
            self.value_index_map[i] = {}
            for j in range(1, self.x_values_range[i] + 1):
                self.value_index_map[i][j] = (self.x[:, i] == j)
                
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

    def fit(self, max_parents=2):
        for i in self.ordering[1:]:
            y = self.bayesian_score()
            while True:
                if len(list(self.G.predecessors(i))) == max_parents:
                    break

                y_best, j_best = float('-inf'), None
                for j in self.ordering[:i]:
                    if not self.G.has_edge(j, i) and not nx.has_path(self.G, i, j):
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


def compute(infile, outfile):
    x, x_header = read_csv_to_array(infile)
    mi = MutualInformation(x)
    mi_ordering, mi = mi.mi_rank()

    k2 = K2Search(x)
    bs = k2.fit(max_parents=3)
    print('....\n With default ordering')
    print(f'Bayesian Score: {bs}')
    print(f'Edges: {k2.G.edges}')

    ordering = mi_ordering[1:] + mi_ordering[0:1]
    k2 = K2Search(x, ordering=ordering)
    bs = k2.fit(max_parents=3)
    print('....\n With MI ordering')
    print(f'Bayesian Score: {bs}')
    print(f'Edges: {k2.G.edges}')

    # old testing code
    '''
    k2 = K2Search(x)
    m_ijk, mij_0 = k2.m(4, ((2, 1), (3, 2)))
    bs = k2.bayesian_score()

    k2.G.add_edge(0,1)
    k2.G.add_edge(0,3)
    k2.G.add_edge(1,4)
    k2.G.add_edge(1,6)
    k2.G.add_edge(2,6)

    bs1 = k2.bayesian_score()
    '''

    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()