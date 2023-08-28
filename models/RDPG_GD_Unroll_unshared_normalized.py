import torch.nn as nn
from torch_geometric.nn import TAGConv, Sequential
from models.Transformer_Block import Transformer_Block



class GD_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.gcn = TAGConv(in_channels, out_channels, K=1, normalize=False, bias=False)
        self.gat = Transformer_Block(in_channels, out_channels)

    def forward(self, input, edge_index, edge_index_2): 
        x_1 = self.gcn(input, edge_index) / input.shape[0] + ((input.shape[0]-1)/input.shape[0])*input
        x_2 = self.gat(input, edge_index_2, use_softmax=False, return_attn_matrix=False) / input.shape[0]**3
        return x_1 - x_2

class GD_Unroll(nn.Module):
    def __init__(self, in_channels, out_channels, gd_steps):
        super().__init__()

        self.gd_steps = gd_steps
        layers = []

        for _ in range(gd_steps):
            layers.append((GD_Block(in_channels, out_channels), 'x, edge_index, edge_index_2 -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2', [layer for layer in layers])

    def forward(self, input, edge_index, edge_index_2): 
        x = input
        x = self.gd(x, edge_index, edge_index_2)        
        return x
