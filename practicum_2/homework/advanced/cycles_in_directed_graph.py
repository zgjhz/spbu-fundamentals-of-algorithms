import networkx as nx
import matplotlib.pyplot as plt
from typing import Any
import time as time

TEST_GRAPH_FILES = [
    "graph_1_wo_cycles.edgelist",
    "graph_2_wo_cycles.edgelist",
    "graph_3_w_cycles.edgelist",
]

def plot_graph(G):
    options = dict(
        font_size=12,
        node_size=500,
        node_color="white",
        edgecolors="black",
    )
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, **options)
    if nx.is_weighted(G):
        labels = {e: G.edges[e]['weight'] for e in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

def has_cycles(g: nx.DiGraph):
    def dfs(node, visited=set(), path=set()):
        if node in path:
            return True
        if node in visited:
            return False

        visited.add(node)
        path.add(node)

        for neighbor in g.neighbors(node):
            if dfs(neighbor, visited, path):
                return True
        path.remove(node)
        return False

    if dfs(list(g.nodes)[0]):
        return True

    return False





if __name__ == "__main__":
    for filename in TEST_GRAPH_FILES:
        # Load the graph
        G = nx.read_edgelist(f'/Users/mac/Desktop/Pershin/spbu-fundamentals-of-algorithms/practicum_2/homework/advanced/{filename}', create_using=nx.DiGraph)
        # Output whether it has cycles
        start_time = time.time()
        print(f"Graph {filename} has cycles: {has_cycles(G)}")
        print(time.time() - start_time)
