import torch

def remove_self_loop(x):
    '''

    :param x: (num_nodes,num_nodes)
    :return:
    '''
    return x-torch.diag_embed(torch.diag(x))

def get_01_adjacent_matrix(x):
    '''

    :param x: (num_nodes,num_nodes)
    :return:
    '''
    return torch.where(x>0,torch.ones_like(x),torch.zeros_like(x))

def remove_explored(x,explored):
    '''

    :param x: (num_nodes,num_nodes)
    :return:
    '''
    return torch.where(explored > 0, x, torch.zeros_like(x))

def expand_explored(x,explored):
    '''

    :param x: (num_nodes,num_nodes)
    :param explored: (num_nodes,num_nodes)
    :return:
    '''
    explored=explored+x
    return remove_self_loop(get_01_adjacent_matrix(explored))

def graph_diffusion(node_adjacent_origin,edge_attr_origin,weight,hop=None,
                    explored=None,att_map_bias=None,node_adjacent_hop_origin=None):
    '''

    :param node_adjacent_hop_origin multi-hop adjacent matrix,
    (num_nodes,num_nodes), must be 0/1, no self-loop
    :param att_map_bias: (ndim_edge_features(num_heads),num_nodes,num_nodes)
    :param node_adjacent_origin: (num_nodes,num_nodes), must be 0/1, no self-loop
    :param edge_attr_origin: (num_nodes,num_nodes,ndim_edge_features(num_heads))
    :param weight: (max_distance,ndim_edge_features(num_heads))
    :param hop:
    :return:
    '''
    num_heads, num_nodes = edge_attr_origin.shape[-1], edge_attr_origin.shape[-2]

    node_adjacent_origin=remove_self_loop(node_adjacent_origin)

    # initialize
    if hop is None:
        explored=get_01_adjacent_matrix(torch.diag_embed(torch.ones(num_nodes)).to(node_adjacent_origin.device)+
                                        node_adjacent_origin)
        hop=weight.shape[0]-1
        att_map_bias=torch.zeros_like(edge_attr_origin).to(torch.float32)
        node_adjacent_hop_origin=node_adjacent_origin

    # num_nodes,num_nodes,ndim_edge_features->ndim_edge_features,num_nodes,num_nodes
    edge_attr=edge_attr_origin.permute(2,0,1)
    node_adjacent_hop=node_adjacent_hop_origin.unsqueeze(dim=-1).repeat((1,1,num_heads)).permute(2,0,1)

    node_adjacent_hop_origin=node_adjacent_hop_origin.to(torch.float32)
    node_adjacent_origin=node_adjacent_origin.to(torch.float32)
    mask=torch.where(explored==node_adjacent_origin@node_adjacent_hop_origin,
                     torch.zeros_like(node_adjacent_origin,device=weight.device),
                     node_adjacent_origin@node_adjacent_hop_origin)
    mask=mask.to(torch.float32)
    # Attention: weight[hop] is the (weight.shape[0]-hop)th embedding
    #print(att_map_bias.shape,((node_adjacent_hop.to(torch.float32)@edge_attr)*mask).shape,weight[hop].shape)
    att_map_bias=att_map_bias+((node_adjacent_hop.to(torch.float32)@edge_attr)*mask).permute(1,2,0)*weight[hop]

    if hop>0:
        hop=hop-1
        node_adjacent_new = node_adjacent_origin @ node_adjacent_hop_origin
        node_adjacent_new=get_01_adjacent_matrix(node_adjacent_new)
        #node_adjacent_new=remove_explored(x=node_adjacent_new,explored=explored)
        node_adjacent_new=remove_self_loop(node_adjacent_new)
        explored=expand_explored(x=node_adjacent_new,explored=explored)
        return graph_diffusion(
            node_adjacent_origin=node_adjacent_origin,
            edge_attr_origin=edge_attr_origin,
            weight=weight,
            hop=hop,
            explored=explored,
            att_map_bias=att_map_bias,
            node_adjacent_hop_origin=node_adjacent_new
        )
    else:
        node_adjacent_new = node_adjacent_origin @ node_adjacent_hop_origin
        node_adjacent_new = get_01_adjacent_matrix(node_adjacent_new)
        # node_adjacent_new=remove_explored(x=node_adjacent_new,explored=explored)
        node_adjacent_new = remove_self_loop(node_adjacent_new)
        explored = expand_explored(x=node_adjacent_new, explored=explored)
        return att_map_bias,explored


if __name__=="__main__":
    import numpy as np
    edge_index=torch.Tensor(np.load("tmp_data/edge_index.npy")).to(torch.long)
    from node_edge_switch import get_node_adjacent_matrix
    node_adjacent = get_node_adjacent_matrix(edge_index=edge_index)
    edge_attr=torch.randn(48,32)
    from tensor_manipulation import integrate_NodeAdjacent_EdgeFeature
    edge_attr=integrate_NodeAdjacent_EdgeFeature(edge_index,edge_attr)

    att_bias=graph_diffusion(
        node_adjacent_origin=node_adjacent,
        edge_attr_origin=edge_attr,
        weight=torch.randn(8,32)
    )
    print(att_bias)
