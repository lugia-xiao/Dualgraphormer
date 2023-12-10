import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size*4)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size*4, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

    def forward(self, q, k, v, attn_bias=None,explored=None):
        orig_v_size = q.size()

        d_k = self.att_size
        d_v = self.att_size

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view( -1, self.num_heads, d_k)
        k = self.linear_k(k).view( -1, self.num_heads, d_k)
        v = self.linear_v(v).view( -1, self.num_heads, d_v)

        q = q.transpose(0,1)                  # [b, h, q_len, d_k]
        v = v.transpose(0,1)                  # [b, h, v_len, d_v]
        k = k.transpose(0,1).transpose(1,2)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        else:
            print("wrong! Where is attention bias?!")
        if explored is not None:
            explored1=explored.unsqueeze(dim=0).repeat(self.num_heads,1,1)
            x[:,1:,1:] = torch.where(explored1==0,torch.zeros_like(x[:,1:,1:] ,device=x.device),x[:,1:,1:])

        x = torch.softmax(x, dim=-1)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        #print(x[0,:,:])

        x = x.transpose(0,1).contiguous()  # [b, q_len, h, attn]
        x = x.view(-1, self.num_heads * d_v)

        assert x.size() == orig_v_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None,explored=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias,explored)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

'''class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.node_encoder=EncoderLayer_(hidden_size, ffn_size, dropout_rate,
                                 attention_dropout_rate, num_heads)
        self.edge_encoder=EncoderLayer_(hidden_size, ffn_size, dropout_rate,
                                 attention_dropout_rate, num_heads)

    def forward(self, x):
        # for node
        vnode=x[3]'''

class Node_Edge_Switch_Attention(nn.Module):
    def __init__(self,hidden_size,attention_dropout_rate,num_heads,edge_to_node=True):
        super().__init__()
        self.edge_to_node=edge_to_node
        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, T):
        '''
        if edge_to_node, q is edge feature, softmax(T@q@K)@V
        else q is node feature, others are edge
        :param q:
        :param k:
        :param v:
        :param T: transition matrix
        :return:
        '''
        orig_v_size = v.size()

        d_k = self.att_size
        d_v = self.att_size

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(-1, self.num_heads, d_k)
        k = self.linear_k(k).view(-1, self.num_heads, d_k)
        v = self.linear_v(v).view(-1, self.num_heads, d_v)

        q = q.transpose(0, 1)  # [b, h, q_len, d_k]
        v = v.transpose(0, 1)  # [b, h, v_len, d_v]
        k = k.transpose(0, 1).transpose(1, 2)  # [b, h, d_k, k_len]

        # Node-Edge-Switch
        if self.edge_to_node:
            divided=torch.sum(T,dim=-1).unsqueeze(dim=-1).repeat((1,T.shape[-1]))
            divided[divided==0]=1
            q=(T/divided)@q
        else:
            divided=torch.sum(T,dim=0).unsqueeze(dim=0).repeat((T.shape[0],1))
            divided[divided==0]=1
            q=((T/divided).T)@q

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]

        x = torch.softmax(x, dim=-1)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(0,1).contiguous()  # [b, q_len, h, attn]
        x = x.view(-1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_v_size
        return x

class Node_Edge_Switch_Transformer_(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, num_heads,edge_to_node=True):
        super().__init__()
        self.edge_to_node=edge_to_node
        self.self_attention_norm_node = nn.LayerNorm(hidden_size)
        self.self_attention_norm_edge = nn.LayerNorm(hidden_size)
        self.self_attention = Node_Edge_Switch_Attention(
            hidden_size, attention_dropout_rate, num_heads,edge_to_node)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self,node_feature,edge_feature,T):
        if self.edge_to_node:
            y = self.self_attention_norm_node(node_feature)
            edge_feature1 = self.self_attention_norm_edge(edge_feature)
            y = self.self_attention(edge_feature1, y, y, T)
            y = self.self_attention_dropout(y)
            x = node_feature + y
            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = x + y
            return x
        else:
            y = self.self_attention_norm_edge(edge_feature)
            node_feature1 = self.self_attention_norm_node(node_feature)
            y = self.self_attention(node_feature1, y, y, T)
            y = self.self_attention_dropout(y)
            x = edge_feature + y

            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = x + y
            return x

class Node_Edge_Switch_Transformer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, num_heads,edge_to_node=True):
        super().__init__()
        self.node_transformer=Node_Edge_Switch_Transformer_(
            hidden_size, ffn_size, dropout_rate,
            attention_dropout_rate, num_heads, True
        )
        self.edge_transformer=Node_Edge_Switch_Transformer_(
            hidden_size, ffn_size, dropout_rate,
            attention_dropout_rate, num_heads, False
        )

    def forward(self,x):
        x_origin=x
        x[0][0]=self.node_transformer(x[0][0],x[1][0],x[2])
        x[1][0] = self.edge_transformer(x_origin[0][0], x_origin[1][0], x[2])
        return x