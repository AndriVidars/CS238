from pgmpy.estimators import HillClimbSearch, StructureScore
from pgmpy.models import BayesianNetwork
import networkx as nx
import pandas as pd
from genetic_search import BayesNetwork

def read_csv_to_dataframe(infile):
    df = pd.read_csv(f'data/{infile}', dtype=int)
    return df

def main():
    df = read_csv_to_dataframe('small.csv')
    cols = list(df.columns)
    col_map = {v:i for i, v in enumerate(cols)}

    hc = HillClimbSearch(df)
    best_model = hc.estimate()
    edges = []
    for e in best_model.edges:
        edges.append((col_map[e[0]], col_map[e[1]]))
    nx_graph = nx.DiGraph(edges)

    x = df.to_numpy()
    bn = BayesNetwork(x, G=nx_graph)
    score = bn.bayesian_score()
    print(score)

if __name__ == '__main__':
    main() # this is not nearly as good as my implementation(genetic search)
