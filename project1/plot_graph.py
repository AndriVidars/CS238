import networkx as nx
import matplotlib.pyplot as plt
from project1 import read_csv_to_array

meta = {
    'small': ('red', 3200),
    'medium': ('blue', 800),
    'large': ('green', 200),
}

def plot_graph(gph_file):
    source_data = f"{gph_file.split('_')[2]}"
    _, labels = read_csv_to_array(f"{source_data}.csv")

    node_id_map = {x: i for i, x in enumerate(labels)}

    G = nx.DiGraph()
    with open(f'graphs/{gph_file}') as f:
        lines = f.readlines()
    
    edges = [(node_id_map[e.split(', ')[0].strip()],
        node_id_map[e.split(', ')[1].strip()])
        for e in lines]
    
    node_color, node_size = meta[source_data]
    
    G = nx.DiGraph()
    G.add_edges_from(edges)

    if source_data == 'small':
        plt.figure(figsize=(8, 6))
    elif source_data == 'medium':
        plt.figure(figsize=(11, 8))
    else:
        plt.figure(figsize=(12, 9))

    pos = nx.spring_layout(G)  # Layout for the graph
    nx.draw(G, pos, with_labels=True, labels={i: labels[i] for i in G.nodes()},
            node_color=node_color, arrows=True, 
            node_size=node_size, font_size=8, alpha=0.4)
    
    plt.title(f"Bayesian Network structure for {source_data} dataset")
    plt.savefig(f"graph_plots/{gph_file.split('.')[0]}_plot.png")

if __name__ == '__main__':
    gph_small = 'parallel_local_small_(-3794.86).gph'
    gph_medium = 'parallel_local_medium_(-96340.36).gph'
    gph_large = 'parallel_local_large_(-407808.82).gph'

    plot_graph(gph_small)
    plot_graph(gph_medium)
    plot_graph(gph_large)
