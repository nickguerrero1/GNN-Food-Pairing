import pandas as pd
import ssl
from urllib.request import urlopen
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.transforms import ToUndirected
import numpy as np

edge_url="https://raw.githubusercontent.com/lamypark/FlavorGraph/master/input/edges_191120.csv"
edges_df=pd.read_csv(edge_url)
edges_df = edges_df.iloc[:111355,:]

node_url="https://raw.githubusercontent.com/lamypark/FlavorGraph/master/input/nodes_191120.csv"
nodes_df=pd.read_csv(node_url)
nodes_df = nodes_df[nodes_df['node_id'].isin(
    (set(edges_df.id_1.values).union(set(edges_df.id_2.values))))]

flavorGraph = Data()

edge_weight = torch.tensor(edges_df['score'], dtype=torch.float)
node_index = torch.tensor(nodes_df['node_id'].values, dtype=torch.long)

node_map = dict()
for i in range(len(node_index)):
    node_map[(int(node_index[i]))] = i

edge_index_np = np.array([
    edges_df.id_1.apply(lambda x: node_map[x]).values,
    edges_df.id_2.apply(lambda x: node_map[x]).values
])
edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)

flavorGraph.x = node_index.view(node_index.size(0), -1)
flavorGraph.edge_index = edge_index.type(torch.int64)
flavorGraph.edge_weight = edge_weight

#transform = RandomLinkSplit(is_undirected=True,add_negative_train_samples=False,disjoint_train_ratio=0.35)
#train_data, val_data, test_data = transform(flavorGraph)