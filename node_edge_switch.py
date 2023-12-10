import torch

def get_node_adjacent_matrix(edge_index,num_nodes=None,sparse=False):
    '''

    :param edge_index:
    :param num_nodes:
    :param sparse:
    :return: 0/1 Node adjacent matrix without self-loop, and no normalization
    '''
    if num_nodes is None:
        num_nodes = round(float(torch.max(edge_index) + 1))
    node_adjacent=torch.sparse_coo_tensor(indices=edge_index,
                                          values=torch.ones((edge_index.shape[-1])),
                                          size=(num_nodes,num_nodes))
    if sparse==False:
        node_adjacent=node_adjacent.to_dense()
    return node_adjacent

def get_edge_adjacent_and_attr(edge_index_origin,node_features):
    '''

    :param edge_index_origin: pure edge_index, (2,num_edges)
    :param node_features: (num_nodes,ndim_node_features)
    :return:
    edge_adjacent: 0/1 (num_edges,num_edges)
    edge_attr_large: the edge features for the edge graph, (num_edge,num_edge,num_features)
    '''
    edge_index=edge_index_origin[:,::2]
    edge_index_start=(edge_index[0]+1).unsqueeze(dim=-1)
    edge_index_end=(edge_index[1]+1).unsqueeze(dim=-1)

    connection_start_start = edge_index_start @ edge_index_start.transpose(-2, -1)
    connection_start_start[connection_start_start != edge_index_start ** 2] = 0

    connection_start_end = edge_index_start @ edge_index_end.transpose(-2, -1)
    connection_start_end[connection_start_end != edge_index_start ** 2] = 0

    connection_end_start = edge_index_end @ edge_index_start.transpose(-2, -1)
    connection_end_start[connection_end_start != edge_index_end ** 2] = 0

    connection_end_end = edge_index_end @ edge_index_end.transpose(-2, -1)
    connection_end_end[connection_end_end != edge_index_end ** 2] = 0

    edge_adjacent=torch.sqrt(connection_start_start)+torch.sqrt(connection_start_end)+torch.sqrt(connection_end_start)+torch.sqrt(connection_end_end)
    edge_adjacent=edge_adjacent-torch.diag_embed(torch.diag(edge_adjacent))
    edge_adjacent_feature_number=edge_adjacent

    edge_adjacent_index = torch.nonzero(edge_adjacent).transpose(-1, -2).contiguous().to(torch.long)

    #edge_attr=node_features[(edge_adjacent[edge_adjacent>0]-1).to(torch.long)]
    ndim_node_features=node_features.shape[-1]
    edge_attr_large=torch.zeros((edge_adjacent.shape[0],edge_adjacent.shape[0],ndim_node_features))
    if node_features.dtype is torch.long:
        edge_attr_large=edge_attr_large.to(torch.long)
    edge_attr_large[edge_adjacent_index[0],edge_adjacent_index[1],:]=node_features[edge_adjacent[edge_adjacent_index[0],edge_adjacent_index[1]].to(torch.long)-1,:]

    #print(edge_attr_large[edge_adjacent_index[0],edge_adjacent_index[1],:].shape,node_features[edge_adjacent[edge_adjacent_index[0],edge_adjacent_index[1]].to(torch.long)-1,:].shape)
    edge_adjacent=torch.where(edge_adjacent>0,torch.ones_like(edge_adjacent),torch.zeros_like(edge_adjacent))
    return edge_adjacent_feature_number.to(torch.long),edge_adjacent,edge_attr_large

def get_transition_matrix(edge_index_origin,num_nodes=None,sparse=False):
    '''
    The transition matrix T \in R^{NV×NE}. T_{i,m} = 1 if node_i is the starting point
     or ending point of edge_m, and all other elements in the matrix are 0
    :param edge_index:
    :return: Transition matrix T, where T@T.T=Node_Adjacent_Matrix (with self-loop)
    '''
    edge_index=edge_index_origin[:,::2]
    if num_nodes is None:
        num_nodes=round(float(torch.max(edge_index)+1))
    row_index=edge_index.transpose(-1,-2).contiguous().reshape((-1))
    col_index=torch.arange(start=0,end=edge_index.shape[-1],step=1)
    col_index=col_index.unsqueeze(dim=-1).repeat((1,2)).contiguous().reshape((-1))
    values=torch.ones((edge_index.shape[-1])).repeat((2))
    T=torch.sparse_coo_tensor(indices=torch.stack([row_index,col_index]),values=values,
                              size=(num_nodes,edge_index.shape[-1]))
    #print([row_index,col_index])
    if sparse==False:
        T=T.to_dense()
    #print(torch.mean(T,dim=-1).unsqueeze(dim=-1).repeat((1,T.shape[-1])).shape)
    return T

def edge_node_features_to_node_edge_features(node_adjacent_feature_number,edge_node_features):
    num_nodes=node_adjacent_feature_number.shape[0]
    node_edge_features_new=torch.zeros((num_nodes,num_nodes,edge_node_features.shape[-1]),device=edge_node_features.device)
    node_edge_feature_number_index = torch.nonzero(node_adjacent_feature_number).transpose(-1, -2).contiguous().to(torch.long)
    node_edge_features_new[node_edge_feature_number_index[0],node_edge_feature_number_index[1],:]=edge_node_features[
        node_adjacent_feature_number[node_edge_feature_number_index[0],node_edge_feature_number_index[1]]-1,:
    ]
    return node_edge_features_new

def node_edge_features_switch_(node_adjacent_feature_number,
                              edge_adjacent_feature_number,
                              node_node_features,edge_node_features):
    node_edge_features_new=edge_node_features_to_node_edge_features(
        node_adjacent_feature_number,
        edge_node_features
    )
    edge_edge_features_new=edge_node_features_to_node_edge_features(
        edge_adjacent_feature_number,
        node_node_features
    )
    return node_edge_features_new,edge_edge_features_new

def node_edge_features_switch(x):
    node_adjacent_feature_number,node_node_features=x[0][1],x[0][0]
    edge_adjacent_feature_number,edge_node_features=x[1][1],x[1][0]
    node_edge_features_new, edge_edge_features_new=node_edge_features_switch_(
        node_adjacent_feature_number,
        edge_adjacent_feature_number,
        node_node_features,edge_node_features)
    x[0][3]=node_edge_features_new
    x[1][3]=edge_edge_features_new
    return x

if __name__=="__main__":
    import numpy as np
    edge_index=torch.Tensor(np.load("tmp_data/edge_index.npy"))

    node_adjacent=get_node_adjacent_matrix(edge_index=edge_index)
    #print("node_adjacent",node_adjacent)

    edge_adjacent,edge_attr_for_edge_graph=get_edge_adjacent_and_attr(edge_index,torch.randn((22,128)))
    print("edge_adjacent",edge_adjacent.shape)
    print("edge_attr", edge_attr_for_edge_graph.shape,len(torch.nonzero(edge_attr_for_edge_graph)))

    T=get_transition_matrix(edge_index=edge_index)
    print(T.shape)

    print("verify:\n\n")
    print((T@T.transpose(-1,-2)).shape,"\n",node_adjacent.shape)

