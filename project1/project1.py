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

class K2Search:
    def __init__(self, x):
        self.x = x
        self.x_values_range = np.max(x, axis=0)
        self.n = x.shape[1]
        self.n_obs = x.shape[0]
    
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.n))
        
        self.ordering = range(self.n) # TODO: explore methods for this

        self.value_index_map = {} # node: {value: ndarray(bool)} # True if x[:, index] == value
        for i in range(self.n):
            self.value_index_map[i] = {}
            for j in range(1, self.x_values_range[i] + 1):
                self.value_index_map[i][j] = (self.x[:, i] == j)
                
    
    @lru_cache(maxsize=None)
    def m(self, i, j):
        # node i
        # j, enumeration of parent node instantiation. represented with tuple of tuples ((node, val))         
        idx = np.full(self.n_obs, True)
        for parent_node, parent_node_val in j:
            idx = np.logical_and(idx, self.value_index_map[parent_node][parent_node_val])

        i_vals = (self.x[idx])[:,i]
        m_ijk_ = np.array([np.sum(i_vals == k) for k in range(1, self.x_values_range[i] + 1)])
        m_ij0 = np.sum(idx)
        return m_ijk_, m_ij0
    
    @lru_cache(maxsize=None)
    def get_parent_instantiation(self, parents):
        parents_vals = [list(range(1, self.x_values_range[parent] + 1)) for parent in parents]
        product_vals = product(*parents_vals)
        return [tuple(zip(parents, values)) for values in product_vals]
    
    def baysian_score(self):
        p = 0
        for i in range(self.n):
            parents = tuple(self.G.predecessors(i))
            
            if not parents:
                # TODO: how is this handled?
                continue
            
            q = self.get_parent_instantiation(parents)
            r = self.x_values_range[i]
            for j in q:
                m_ijk, m_ij0 = self.m(i, j)
                alpha_ij0 = r # uniform prior, each alpha_ijk has value 1
                p += (loggamma(alpha_ij0) - loggamma(alpha_ij0 + m_ij0))
                for k in range(r): # shifted for 0-indexing
                    p += loggamma(1 + m_ijk[k]) # uniform prior, denominator term eliminated
        
        return p


    def fit(self):
        pass

def compute(infile, outfile):
    x, x_header = read_csv_to_array(infile)
    k2 = K2Search(x)

    # debug
    m_ijk, mij_0 = k2.m(4, ((2, 1), (3, 2)))

    k2.G.add_edge(0,1)
    k2.G.add_edge(0,3)
    k2.G.add_edge(1,4)
    k2.G.add_edge(1,6)
    k2.G.add_edge(2,6)

    bs = k2.baysian_score()
    pass

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