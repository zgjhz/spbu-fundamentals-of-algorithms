import os

import networkx as nx


TEST_GRAPH_FILES = [
    "graph_1_wo_cycles.edgelist",
    "graph_2_w_cycles.edgelist",
]


def has_cycles(g: nx.DiGraph):
    ##########################
    ### PUT YOUR CODE HERE ###
    ##########################

    pass


if __name__ == "__main__":
    for filename in TEST_GRAPH_FILES:
        # Load the graph
        G = nx.read_edgelist(
            os.path.join("practicum_2", "homework", filename), create_using=nx.DiGraph
        )
        # Output whether it has cycles
        print(f"Graph {filename} has cycles: {has_cycles(G)}")
