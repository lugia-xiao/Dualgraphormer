import torch
import node_edge_switch
import tensor_manipulation

def process_graph(x,to_cuda=True):
    node_adjacent=node_edge_switch.get_node_adjacent_matrix(edge_index=x.edge_index,
                                                            num_nodes=x.x.shape[0])
    node_edge_features=tensor_manipulation.integrate_NodeAdjacent_EdgeFeature(
        edge_index=x.edge_index,
        edge_attr=x.edge_attr,
        num_nodes=x.x.shape[0]
    )
    node_indegree=tensor_manipulation.get_degree(adjacent_matrix=node_adjacent)
    tmp_arrange=torch.arange(1,x.edge_index.shape[-1]/2+1).unsqueeze(dim=-1)
    node_adjacent_feature_number=torch.sparse_coo_tensor(
        indices=x.edge_index,
        values=torch.concat([tmp_arrange,tmp_arrange],dim=-1).view(-1),
        size=(x.x.shape[0],x.x.shape[0])
    ).to_dense().to(torch.long)

    edge_adjacent_feature_number,edge_adjacent,edge_edge_features=node_edge_switch.get_edge_adjacent_and_attr(
        edge_index_origin=x.edge_index,node_features=x.x
    )
    edge_indegree=tensor_manipulation.get_degree(adjacent_matrix=edge_adjacent)

    T=node_edge_switch.get_transition_matrix(edge_index_origin=x.edge_index
                                             ,num_nodes=x.x.shape[0])
    if to_cuda:
        return [[x.x.cuda(), node_adjacent_feature_number.cuda(), node_adjacent.cuda(), node_edge_features.cuda(), node_indegree.cuda()],
                [x.edge_attr[::2, :].cuda(), edge_adjacent_feature_number.cuda(), edge_adjacent.cuda(), edge_edge_features.cuda(), edge_indegree.cuda()],
                T.cuda()]
    else:
        return [[x.x, node_adjacent_feature_number, node_adjacent, node_edge_features, node_indegree],
                [x.edge_attr[::2, :], edge_adjacent_feature_number, edge_adjacent, edge_edge_features, edge_indegree],
                T]

if __name__=="__main__":
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.data import DataLoader

    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='../../dataset/')
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)

    for (stepi, x) in enumerate(train_loader, start=1):
        print(x[0].x.shape, x[0].edge_index.shape, x[0].edge_attr.shape)
        print(x[0].edge_index)
        processed=process_graph(x[0])
        node_features,node_adjacent_feature_number,node_adjacent, node_edge_features, node_indegree=processed[0]
        edge_features,edge_adjacent_feature_number,edge_adjacent, edge_edge_features, edge_indegree=processed[1]
        T=processed[2]
        print(node_features.shape,node_adjacent_feature_number.shape,node_adjacent.shape, node_edge_features.shape, node_indegree.shape)
        print(edge_features.shape,edge_adjacent_feature_number.shape,edge_adjacent.shape, edge_edge_features.shape, edge_indegree.shape)
        print(T.shape)
        break