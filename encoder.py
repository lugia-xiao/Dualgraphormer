import torch
import math
import torch.nn as nn

import node_edge_switch
from attention import EncoderLayer
from graph_diffusion import graph_diffusion
from GAT import Edge_GAT,Edge_GCN
from node_edge_fusion import Node_Edge_Fusion

def concat_vnode(x,att_bias_origin,node_or_edge="node"):
    if node_or_edge=="node":
        num_heads,num_nodes=att_bias_origin.shape[-1],att_bias_origin.shape[0]
        v_node=x[3][0][0]
        if len(v_node.shape)!=2:
            v_node=v_node.unsqueeze(dim=0)
        v_dis=x[3][0][1]
        node_features_new=torch.concat([v_node,x[0][0]],dim=0)
        att_bias0=torch.zeros((num_nodes+1,num_nodes+1,num_heads),device=v_node.device)
        att_bias0[0,1:,:]=v_dis
        att_bias0[1:, 0, :] = v_dis
        att_bias0[:,0,0]=0
        att_bias0[1:,1:,:]=att_bias0[1:,1:,:]+att_bias_origin
        return node_features_new,att_bias0.permute(2,0,1)
    elif node_or_edge=="edge":
        num_heads, num_nodes = att_bias_origin.shape[-1], att_bias_origin.shape[1]
        v_node = x[3][1][0]
        v_dis = x[3][1][1]
        if len(v_node.shape)!=2:
            v_node=v_node.unsqueeze(dim=0)
        edge_features_new = torch.concat([v_node, x[1][0]], dim=0)
        att_bias0 = torch.zeros((num_nodes + 1, num_nodes + 1, num_heads),device=v_node.device)
        att_bias0[0, 1:, :] = v_dis
        att_bias0[1:, 0, :] = v_dis
        att_bias0[:, 0, 0] = 0
        att_bias0[1:, 1:, :] = att_bias0[1:, 1:, :] + att_bias_origin
        return edge_features_new, att_bias0.permute(2, 0, 1)
    else:
        print("wrong node_or_edge type")
        return


class Encoder(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, num_heads,num_hops=8):
        super().__init__()
        self.node_encoder=EncoderLayer(hidden_size, ffn_size, dropout_rate,
                                  attention_dropout_rate, num_heads)

    def forward(self,x):
        att_bias=x[3][1][0]
        explored=x[3][1][1]

        node_features,att_bias=concat_vnode(x,att_bias,"node")
        new_node_features=self.node_encoder(node_features,att_bias,explored)
        x[0][0]=new_node_features[1:,:]
        x[3][0][0]=new_node_features[0,:]
        return x

class Switch_Layer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, num_heads,num_hops=8):
        super().__init__()
        self.edge_GAT = Edge_GAT(num_heads, num_heads, hidden_size)  # Edge_GAT(num_heads,num_heads,hidden_size)
        self.node_edge_fusion = Node_Edge_Fusion(num_heads, hidden_size)

        self.node_diffusion_weight=nn.Parameter(torch.randn(num_hops,num_heads))
        torch.nn.init.xavier_normal_(self.node_diffusion_weight, gain=1.44)

    def forward(self,x):
        x = self.edge_GAT(x)
        x = self.node_edge_fusion(x)
        x = node_edge_switch.node_edge_features_switch(x)

        position_bias = x[0][3]
        BFS_bias, explored = graph_diffusion(node_adjacent_origin=x[0][1],
                                             edge_attr_origin=x[0][3],
                                             weight=self.node_diffusion_weight)
        att_bias = position_bias + BFS_bias
        x[3][1][0] = att_bias
        x[3][1][1] = explored
        return x

if __name__=="__main__":
    torch.manual_seed(114514)
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.data import DataLoader
    from data import process_graph
    from embedding import Node_Edge_Embedding

    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='../../dataset/')
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)

    my_embedding=Node_Edge_Embedding(
        hidden_dim=512,
        num_heads=32,
        dataset_name="ogbg-molhiv",
    )

    my_encoder=Encoder(
        hidden_size=512,
        ffn_size=512,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        num_heads=32,
        num_hops=8
    )

    for (stepi, x) in enumerate(train_loader, start=1):
        print(x[0].x.shape, x[0].edge_index.shape, x[0].edge_attr.shape)
        processed=process_graph(x[0],to_cuda=False)
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

        print("-"*10)
        x=node_edge_switch.node_edge_features_switch(x)
        node_features, node_adjacent_feature_number, node_adjacent, node_edge_features, node_indegree = x[0]
        edge_features, edge_adjacent_feature_number, edge_adjacent, edge_edge_features, edge_indegree = x[1]
        T = x[2]
        print(node_features.shape, node_adjacent.shape, node_edge_features.shape, node_indegree.shape)
        print(edge_features.shape, edge_adjacent.shape, edge_edge_features.shape, edge_indegree.shape)
        print(T.shape)

        print("-" * 10)
        print("encoder:")
        x = my_encoder(x)
        node_features, node_adjacent_feature_number, node_adjacent, node_edge_features, node_indegree = x[0]
        edge_features, edge_adjacent_feature_number, edge_adjacent, edge_edge_features, edge_indegree = x[1]
        T = x[2]
        print(node_features.shape, node_adjacent.shape, node_edge_features.shape, node_indegree.shape)
        print(edge_features.shape, edge_adjacent.shape, edge_edge_features.shape, edge_indegree.shape)
        print(T.shape)
        break

