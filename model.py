import torch
import math
import torch.nn as nn

from embedding import Node_Edge_Embedding
from encoder import Encoder,Switch_Layer
from data import process_graph

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class DualGraphormer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, num_heads, num_classes,
                 input_dropout_rate=0.1,n_layers=6,num_hops=8,
                 dataset_name="ogbg-molhiv"):
        assert n_layers%3==0

        super().__init__()
        self.num_classes=num_classes
        if dataset_name=='ogbg-molpcba':
            self.num_classes=128
        elif dataset_name=="PCQM4Mv2":
            self.num_classes=1

        self.embedding=Node_Edge_Embedding(
            hidden_dim=hidden_size,
            num_heads=num_heads,
            input_dropoout_rate=input_dropout_rate,
            dataset_name=dataset_name,
            num_hops=num_hops
        )

        self.n_layers=n_layers
        self.encoders=[]
        for i in range(3):
            self.encoders=self.encoders+[
                Encoder(
                    hidden_size, ffn_size, dropout_rate,
                    attention_dropout_rate, num_heads, num_hops=num_hops
                ) for j in range(n_layers//3)
            ]
            self.encoders=self.encoders+[
                Switch_Layer(
                    hidden_size, ffn_size, dropout_rate,
                    attention_dropout_rate, num_heads, num_hops=num_hops
                )
            ]
        self.encoders=nn.ModuleList(self.encoders)

        self.final_ln = nn.LayerNorm(hidden_size)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

        self.head = nn.Linear(hidden_size, self.num_classes)

    def forward_single_one(self,x):
        #print_x(x)
        x=self.embedding(x)
        for i in range(len(self.encoders)):
            x=self.encoders[i](x)
        vnodes=x[3][0][0].view(-1)
        return self.head(self.final_ln(vnodes))

    def forward(self,x):
        batch_size=len(x)
        output=[]
        for i in range(batch_size):
            output.append(self.forward_single_one(process_graph(x[i],to_cuda=True)))
        return torch.stack(output)

class Unbalanced_BCE_logits_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y_pred,y):
        y_pred=torch.nn.functional.sigmoid(y_pred)
        num0=torch.sum(y==0)
        num1=torch.sum(y==1)
        num0=num0/(num1+num0)*2
        num1 = num1 / (num1 + num0) * 2
        weight=torch.zeros_like(y_pred,device=y.device)
        weight[y==0]=num1
        weight[y==1]=num0
        return -torch.mean((y*torch.log(y_pred)+(1-y)*torch.log(1-y_pred))*weight)

class Nan_BCE_logits_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y_pred,y):
        y_pred=torch.nn.functional.sigmoid(y_pred)
        y_pred=y_pred[torch.isnan(y)==False]
        y=y[torch.isnan(y)==False]
        loss=(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
        return -torch.mean(loss)

def print_x(x):
    print("-" * 10)
    node_features, node_adjacent_feature_number, node_adjacent, node_edge_features, node_indegree = x[0]
    edge_features, edge_adjacent_feature_number, edge_adjacent, edge_edge_features, edge_indegree = x[1]
    T = x[2]
    print(node_features)
    print(node_features.shape, node_adjacent.shape, node_edge_features.shape, node_indegree.shape)
    print(edge_features.shape, edge_adjacent.shape, edge_edge_features.shape, edge_indegree.shape)
    print(T.shape)

if __name__=="__main__":
    torch.manual_seed(114)
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.data import DataLoader
    from data import process_graph

    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='../../dataset/')
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=10, shuffle=True)

    my_model=DualGraphormer(
        hidden_size=512,
        ffn_size=512,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        num_heads=64,
        num_classes=1,
        input_dropout_rate=0.1,
        n_layers=6,
        num_hops=3,
        dataset_name="ogbg-molhiv",
    ).cuda()
    import time
    start=time.time()
    for (stepi, x) in enumerate(train_loader, start=1):
        y_pred=my_model(x)
        print(time.time()-start)
        print(y_pred)
        break