import torch
import math
import torch.nn as nn

import node_edge_switch
from graph_diffusion import graph_diffusion


class Node_Edge_Embedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        input_dropoout_rate=0.1,
        dataset_name="ogbg-molhiv",
        num_hops=5
    ):
        super().__init__()
        if dataset_name == 'ZINC':
            self.atom_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(64, num_heads, padding_idx=0)
        else:
            self.atom_encoder = nn.Embedding(
                512*9 + 1, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(
                512*3 + 1, num_heads, padding_idx=0)
        self.degree_encoder = nn.Embedding(
            512, hidden_dim, padding_idx=0)

        self.node_vnode = nn.Embedding(1, hidden_dim).weight
        self.node_vnode_distance=nn.Embedding(1,num_heads).weight
        self.input_dropout=nn.Dropout(input_dropoout_rate)

        self.node_diffusion_weight = nn.Parameter(torch.randn(num_hops, num_heads))
        torch.nn.init.xavier_normal_(self.node_diffusion_weight, gain=1.44)

    def forward(self,x):
        node_data = x[0]
        edge_data = x[1]
        node_features = self.atom_encoder(node_data[0]).sum(dim=-2)+self.degree_encoder(node_data[-1].long())
        edge_features = self.edge_encoder(edge_data[0]).sum(dim=-2)

        node_features=self.input_dropout(node_features)
        node_vnode=self.input_dropout(self.node_vnode)

        x[0][0]=node_features
        x[1][0]=edge_features

        x = node_edge_switch.node_edge_features_switch(x)

        position_bias = x[0][3]
        BFS_bias, explored = graph_diffusion(node_adjacent_origin=x[0][1],
                                             edge_attr_origin=x[0][3],
                                             weight=self.node_diffusion_weight)
        att_bias = position_bias + BFS_bias
        x.append([[node_vnode,self.node_vnode_distance],
                  [att_bias,explored]])
        return x

if __name__=="__main__":
    torch.manual_seed(114514)
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.data import DataLoader
    from data import process_graph

    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='../../dataset/')
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)

    my_embedding=Node_Edge_Embedding(
        hidden_dim=512,
        num_heads=32,
        dataset_name="ogbg-molhiv",
    )

    for (stepi, x) in enumerate(train_loader, start=1):
        print(x[0].x.shape, x[0].edge_index.shape, x[0].edge_attr.shape)
        processed=process_graph(x[0])
        node_graph,edge_graph,T=processed
        node_features,node_adjacent_feature_number,node_adjacent, node_edge_features, node_indegree = processed[0]
        edge_features,edge_adjacent_feature_number,edge_adjacent, edge_edge_features, edge_indegree = processed[1]
        print(node_adjacent.shape, node_edge_features.shape, node_indegree.shape)
        print(edge_adjacent.shape, edge_edge_features.shape, edge_indegree.shape)
        print(T.shape)
        print("-"*10)

        x=my_embedding(processed)
        node_features, node_adjacent_feature_number, node_adjacent, node_edge_features, node_indegree = x[0]
        edge_features, edge_adjacent_feature_number, edge_adjacent, edge_edge_features, edge_indegree = x[1]
        T=x[2]
        print(node_features.shape,node_adjacent.shape, node_edge_features.shape, node_indegree.shape)
        print(edge_features.shape,edge_adjacent.shape, edge_edge_features.shape, edge_indegree.shape)
        print(T.shape)
        break