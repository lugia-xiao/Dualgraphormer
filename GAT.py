import torch
import torch.nn as nn

def normalize_adjacent_matrix(x_origin):
    '''
    A=D^{-0.5}(A+I)D^{-0.5}
    :param x_origin:
    :return:
    '''
    x=x_origin
    x=x+torch.diag_embed(torch.ones(x.shape[-1],device=x.device))
    degree=1/torch.sqrt(torch.sum(x,dim=-1))
    degree[torch.isnan(degree)] = 0
    D_05=torch.diag_embed(degree)
    return D_05@x@D_05

class GAT_layer_(nn.Module):
    def __init__(self,ndim_node_in,ndim_node_out,ndim_edge):
        super().__init__()
        self.linear = nn.Sequential(
            nn.LayerNorm(ndim_node_in),
            nn.Linear(ndim_node_in, ndim_node_out)
        )
        self.leaky_relu=nn.LeakyReLU(negative_slope=0.2)
        self.attention_projection = nn.Sequential(
            nn.Linear(2 * ndim_node_out + ndim_edge, (2 * ndim_node_out + ndim_edge)),
            nn.GELU(),
            nn.Linear((2 * ndim_node_out + ndim_edge), 1)
        )
        self.last_ln=nn.LayerNorm(ndim_node_out)

    def forward(self,node_features,edge_features,node_adjacent):
        if sum(node_features.shape)==0:
            print(node_features,edge_features,node_adjacent)
            return node_features
        node_features=self.linear(node_features)
        N,p=node_features.shape[0],node_features.shape[1]

        node_adjacent1=normalize_adjacent_matrix(node_adjacent)
        #print(node_features.shape,N,p)
        node_features_expand=torch.concat([node_features.repeat(1,N).view(N*N,p),node_features.repeat(N,1)],dim=-1).view(N,N,2*p)
        node_features_expand=torch.concat([node_features_expand,edge_features],dim=-1)

        attention_score=self.attention_projection(node_features_expand)
        attention_score=attention_score.squeeze(dim=-1)
        attention_score=self.leaky_relu(attention_score)
        attention_score=torch.where(node_adjacent1>0,attention_score,torch.ones_like(attention_score,device=attention_score.device)*(-1e12))
        attention_score=torch.nn.functional.softmax(attention_score,dim=-1)
        return self.last_ln(attention_score@node_features)

class GAT_layer(nn.Module):
    def __init__(self, ndim_node_in, ndim_node_out, ndim_edge, num_heads):
        super().__init__()
        assert ndim_node_out%num_heads==0

        self.num_heads=num_heads
        self.attentions=[
            GAT_layer_(
                ndim_node_in=ndim_node_in, ndim_node_out=ndim_node_out//num_heads,
                ndim_edge=ndim_edge
            ) for i in range(num_heads)
        ]
        self.attentions=nn.ModuleList(self.attentions)

    def forward(self,node_features,edge_features,node_adjacent):
        output=[]
        for i in range(self.num_heads):
            output.append(self.attentions[i](node_features,edge_features,node_adjacent))
        return torch.concat(output,dim=-1)

class Edge_GAT(nn.Module):
    def __init__(self, ndim_node_in, ndim_node_out, ndim_edge, num_heads=4):
        super().__init__()

        self.GAT=GAT_layer(ndim_node_in, ndim_node_out, ndim_edge, num_heads)

    def forward(self,x):
        node_features, edge_features, node_adjacent=x[1][0],x[1][3],x[1][2]
        x[1][0]=x[1][0]+self.GAT(node_features, edge_features, node_adjacent)
        return x

class Edge_GCN(nn.Module):
    def __init__(self, ndim_node_in, ndim_node_out, ndim_edge, num_heads=4):
        super().__init__()

        self.linear=nn.Linear(ndim_node_in,ndim_node_out)

    def forward(self,x):
        node_features, edge_features, node_adjacent=x[1][0],x[1][3],x[1][2]
        node_adjacent1=normalize_adjacent_matrix(node_adjacent)
        x[1][0]=x[1][0]+node_adjacent1@self.linear(node_features)
        return x



if __name__=="__main__":
    A=torch.randint(2, (4, 4))
    print(A)
    print(normalize_adjacent_matrix(A))


