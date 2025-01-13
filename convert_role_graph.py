import json
from collections import defaultdict

import networkx as nx
import pandas as pd


def get_role_graph_data():
    role_graph_path = "checklist_data/HIPAA/role_kg.graphml"
    role_graph = nx.read_graphml(role_graph_path)

    visited = set()
    edge_map = defaultdict(list)

    for u, v in role_graph.edges():
        print(u, v, role_graph.get_edge_data(u, v))
        edge_map[u].append(v)

    with open("data/role_graph_data.json", "w") as f:
        json.dump(edge_map, f, indent=4)


def role_graph_data_2csv():
    with open("data/role_graph_data.json", "r") as f:
        role_graph_data = json.load(f)

    dataset_dict = {"query": [], "positive": [], "label": []}
    for term, keywords in role_graph_data.items():
        for word in keywords:
            dataset_dict["query"].append(term)
            dataset_dict["positive"].append(word)
            dataset_dict["label"].append(
                -1
            )  # currently for role graph, label is -1. I will find a way to make use of it.

    df = pd.DataFrame(dataset_dict)
    df.to_csv("data/role_graph_data.csv", index=False)


if __name__ == "__main__":
    # get_role_graph_data()

    # role_graph_data_2csv()

    # combine keywords.csv and role_graph_data.csv
    df = pd.read_csv("data/keywords.csv")
    df = pd.concat([df, pd.read_csv("data/role_graph_data.csv")], axis=0)
    df.to_csv("data/keywords_with_rolekg.csv", index=False)
