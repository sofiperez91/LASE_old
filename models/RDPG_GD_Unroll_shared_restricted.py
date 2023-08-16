import torch.nn as nn
from torch_geometric.nn import TAGConv
from models.Transformer_Block_v2 import Transformer_Block


class GD_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.gcn = TAGConv(in_channels, out_channels, K=1, normalize=False, bias=False)
        self.gat = Transformer_Block(in_channels, out_channels)

    def forward(self, input, edge_index, edge_index_2): 
        x_1 = self.gcn(input, edge_index)
        x_2 = self.gat(input, edge_index_2, use_softmax=False, return_attn_matrix=False)
        return x_1 - x_2

class GD_Unroll(nn.Module):
    def __init__(self, in_channels, out_channels, gd_steps):
        super().__init__()

        self.gd_steps = gd_steps
        self.gd = GD_Block(in_channels,out_channels)

    def forward(self, input, edge_index, edge_index_2): 
        x = input
        for i in range(self.gd_steps):
             x = self.gd(x, edge_index, edge_index_2)
        return x
