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
    
class BayesNetwork:
    def __init__(self, x):
        self.x = x
        self.x_values_range = np.max(x, axis=0)
        self.n = x.shape[1]
        self.n_obs = x.shape[0]
    
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.n))

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
    
def randint_exclude(low, high, exclude):
    while True:
        num = np.random.randint(low, high)
        if num != exclude:
            return num
    
class StochasticLocalSearch(BayesNetwork):
    def __init__(self, x, G=None):
        super().__init__(x)
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
                bn.G.add_edge(i, j)
                return bn
    
    def fit(self, max_steps):
        y = self.bayesian_score()
        for _ in range(max_steps):
            bn_neighbor = self.rand_graph_neighbor()
            y_ = bn_neighbor.bayesian_score()
            if y_ > y:
                y, self.G = y_, bn_neighbor.G.copy()
        
        return y
        
def mutual_information_rank(data):
    bayes_net = BayesNetwork(data)
    x = bayes_net.x[:, :-1]  # Exclude response variable
    x_values_range = bayes_net.x_values_range[:-1]
    n = bayes_net.n - 1
    n_obs = bayes_net.n_obs

    value_index_map = bayes_net.value_index_map
    
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

    mi = []
    for i in range(n):
        mi_total = sum(pairwise_mi(i, j) for j in range(n) if i != j)
        mi.append(mi_total)

    mi_ordering = list(np.argsort(-np.array(mi))) + [bayes_net.n - 1]
    mi_scores = mi
    return mi_ordering, mi_scores


def compute(infile, outfile):
    x, x_header = read_csv_to_array(infile)
    
    k2 = K2Search(x)
    bs = k2.fit(max_parents=3)
    print('....\n With default ordering')
    print(f'Bayesian Score: {bs}')
    print(f'Edges: {k2.G.edges}')

    mi_ordering, mi_scores = mutual_information_rank(x)
    k2 = K2Search(x, ordering=mi_ordering)
    bs = k2.fit(max_parents=3)
    print('....\n With MI ordering')
    print(f'Bayesian Score: {bs}')
    print(f'Edges: {k2.G.edges}')
    print(nx.is_directed_acyclic_graph(k2.G))


    print('\nStochasic local search')
    lSerach = StochasticLocalSearch(x)
    bs = lSerach.fit(1000)
    print(f'Bayesian Score: {bs}')
    print(f'Edges: {lSerach.G.edges}')
    print(nx.is_directed_acyclic_graph(lSerach.G))

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