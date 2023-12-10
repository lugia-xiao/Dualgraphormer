import torch
import torch.nn as nn

class Node_Edge_Fusion(nn.Module):
    def __init__(self,ndim_edge,ndim_node):
        super().__init__()

        self.linear=nn.Sequential(
            nn.Linear(ndim_edge + ndim_node, 2 * (ndim_edge + ndim_node)),
            nn.GELU(),
            nn.Linear(2 * (ndim_edge + ndim_node), ndim_node)
        )

    def forward(self,x):
        T=x[2]
        divided = torch.sum(T, dim=-1).unsqueeze(dim=-1).repeat((1, T.shape[-1]))
        divided[divided == 0] = 1
        tmp = (T / divided) @ x[1][0]
        x[0][0]=self.linear(torch.concat([x[0][0],tmp],dim=-1))
        return x