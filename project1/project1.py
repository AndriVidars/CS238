import sys
import numpy as np
import networkx as nx


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def read_csv_to_array(infile):
    with open(f'data/{infile}', 'r') as f:
        header = [s.strip('"') for s in f.readline().strip().split(',')]
        x = np.genfromtxt(f, delimiter=',', skip_header=0,dtype='int') # already read line 0
    
    return x, header
    




def compute(infile, outfile):
    x, x_header = read_csv_to_array(infile)

    networks = boostrap_fit(x, 1000)

    
    """
    k2 = K2Search(x)
    bs = k2.fit(max_parents=2)
    print('....\n With default ordering')
    print(f'Bayesian Score: {bs}')
    print(f'Edges: {k2.G.edges}')

    mi_ordering, mi_scores = mutual_information_rank(x)
    k2 = K2Search(x, ordering=mi_ordering)
    bs = k2.fit(max_parents=2)
    print('....\n With MI ordering')
    print(f'Bayesian Score: {bs}')
    print(f'Edges: {k2.G.edges}')
    print(nx.is_directed_acyclic_graph(k2.G))


    print('\nStochasic local search')
    lSearch = StochasticLocalSearch(x, max_iter=10000)
    bs = lSearch.fit()
    print(f'Bayesian Score: {bs}')
    print(f'Edges: {lSearch.G.edges}')
    print(nx.is_directed_acyclic_graph(lSearch.G))
    print(f'Number of restarts: {lSearch.cnt_restart}')
    #print('\n',[(k, v[1]) for k, v in lSearch.searches.items()])

    print('\nStochasic local search with graph initialized from k2')
    lSearch = StochasticLocalSearch(x, G=k2.G, max_iter=10000)
    bs = lSearch.fit()
    print(f'Bayesian Score: {bs}')
    print(f'Edges: {lSearch.G.edges}')
    print(nx.is_directed_acyclic_graph(lSearch.G))
    print(f'Number of restarts: {lSearch.cnt_restart}')
    """


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