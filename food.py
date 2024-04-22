import pandas as pd
import ssl
from urllib.request import urlopen
import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.data import download_url, extract_zip
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score,roc_auc_score, precision_score
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt

edge_url="https://raw.githubusercontent.com/lamypark/FlavorGraph/master/input/edges_191120.csv"
edges_df=pd.read_csv(edge_url)
node_url="https://raw.githubusercontent.com/lamypark/FlavorGraph/master/input/nodes_191120.csv"
nodes_df=pd.read_csv(node_url)

edges_df = edges_df[edges_df['edge_type'] == 'ingr-ingr']
nodes_df = nodes_df[nodes_df['node_type'] == 'ingredient']

flavorGraph = Data()

edge_weight = torch.tensor(edges_df['score'], dtype=torch.float)
node_index = torch.tensor(nodes_df['node_id'], dtype=torch.int64).unsqueeze(1)

node_map = dict()
for i in range(len(node_index)):
    node_map[(int(node_index[i]))] = i

edge_index_np = np.array([
    edges_df.id_1.apply(lambda x: node_map[x]).values,
    edges_df.id_2.apply(lambda x: node_map[x]).values
])
edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)

flavorGraph.x = node_index # 2D Tensor
flavorGraph.edge_index = edge_index # 2D Tensor
flavorGraph.edge_weight = edge_weight # 1D Tensor

transform = RandomLinkSplit(is_undirected=True,add_negative_train_samples=False,disjoint_train_ratio=0.35)
train_data, val_data, test_data = transform(flavorGraph)

# ---------------------------------
# Building, training, and testing GCN model
# ---------------------------------
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # The main GNN layers, two graph conv layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x.float(), edge_index).relu()
        return self.conv2(x, edge_index)

    # Simple dot product based decoder
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    # Want probabilities out to easily interpret results
    def decode_all(self, z):
      prob_adj = z @ z.t()
      prob_adj = torch.sigmoid(prob_adj)  # Apply sigmoid function to get probabilities
      return prob_adj

model = Net(flavorGraph.num_features, 128, 64)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    y = data.edge_label.cpu().numpy()
    pred = out.cpu().numpy()
    return roc_auc_score(y, pred)

validationMetrics_GCN = []

best_val_auc = final_test_auc = 0
for epoch in range(1, 150):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    validationMetrics_GCN.append([val_auc, test_auc])
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')

z = model.encode(test_data.x, test_data.edge_index)
final_edge_probs_GCN = model.decode_all(z)

#plotting results
plt.plot(np.arange(len(validationMetrics_GCN)),np.array(validationMetrics_GCN)[:,1],label='test_auc')
plt.legend()
"""plt.savefig('/Users/nicholasguerrero/Desktop/Coursework/CS365/GNN-Food-Pairing/plot1.png')"""
plt.show()

start_node = np.random.randint(0, 6653)
for i in range(1, min(np.random.poisson(4, 1)[0] + 1, 7)):
  top_nodes = nodes_df.iloc[final_edge_probs_GCN.topk(10, dim=1).indices[start_node, ]]
  which_one = np.random.randint(0, 10)
  start_node_id = top_nodes.iloc[which_one]["node_id"]
  print(top_nodes.iloc[which_one]["name"])

# ---------------------------------
# Building, training, and testing SAGE model
# ---------------------------------
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x.float(), edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
      prob_adj = z @ z.t()
      prob_adj = torch.sigmoid(prob_adj)  # Apply sigmoid function to get probabilities
      return prob_adj

model = Net(flavorGraph.num_features, 128, 64)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    y = data.edge_label.cpu().numpy()
    pred = out.cpu().numpy()
    return roc_auc_score(y, pred)

validationMetrics_SAGE = []


best_val_auc = final_test_auc = 0
for epoch in range(1, 150):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    validationMetrics_SAGE.append([val_auc, test_auc])
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')

z = model.encode(test_data.x, test_data.edge_index)
final_edge_probs_SAGE = model.decode_all(z)
# Graphing results
plt.plot(np.arange(len(validationMetrics_SAGE)),np.array(validationMetrics_SAGE)[:,1],label='test_auc')
plt.legend()
"""plt.savefig('/Users/nicholasguerrero/Desktop/Coursework/CS365/GNN-Food-Pairing/plot2.png')"""
plt.show()
start_node = np.random.randint(0, 6653)
for i in range(1, min(np.random.poisson(4, 1)[0] + 1, 7)):
  top_nodes = nodes_df.iloc[final_edge_probs_SAGE.topk(10, dim=1).indices[start_node, ]]
  which_one = np.random.randint(0, 10)
  start_node_id = top_nodes.iloc[which_one]["node_id"]
  print(top_nodes.iloc[which_one]["name"])

# #Grahping results of both models
plt.plot(np.arange(len(validationMetrics_GCN)),np.array(validationMetrics_GCN)[:,1],label='test_auc_GCN')
plt.plot(np.arange(len(validationMetrics_SAGE)),np.array(validationMetrics_SAGE)[:,1],label='test_auc_SAGE')
plt.legend()
"""plt.savefig('/Users/nicholasguerrero/Desktop/Coursework/CS365/GNN-Food-Pairing/plot3.png')"""
plt.show()