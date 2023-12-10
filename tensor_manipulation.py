import torch

def integrate_NodeAdjacent_EdgeFeature(edge_index,edge_attr,num_nodes=None):
    '''

    :param edge_index:
    :param edge_attr:
    :param num_nodes:
    :return: (num_nodes,num_nodes,ndim_edge_feature)
    '''
    if num_nodes is None:
        num_nodes=round(float(torch.max(edge_index)+1))
    ndim_edge_feature=edge_attr.shape[-1]
    edge_attr_large=torch.zeros((num_nodes,num_nodes,ndim_edge_feature))
    if edge_attr.dtype is torch.long:
        edge_attr_large=edge_attr_large.to(torch.long)
    edge_attr_large[edge_index[0],edge_index[1],:]=edge_attr
    return edge_attr_large

def get_degree(adjacent_matrix):
    '''

    :param adjacent_matrix: tensor (num_nodes,num_nodes)
    :return:
    '''
    return torch.sum(adjacent_matrix,dim=-1)

def unsqueeze_all(x):
    x_new=[None,None]
    x_new[0]=[x[0][i].unsqueeze(dim=-1) for i in range(len(x[0]))]
    x_new[1] = [x[1][i].unsqueeze(dim=-1) for i in range(len(x[1]))]
    return x_new

if __name__=="__main__":
    import numpy as np

    edge_index = torch.Tensor(np.load("tmp_data/edge_index.npy"))
    edge_attr=torch.randn((48,512))


