import torch
import torch.nn as nn
from torch_geometric.nn import TAGConv, Sequential
from models.Transformer_Block_v2 import Transformer_Block

class GD_Block(nn.Module):
    def __init__(self, in_channels, out_channels, device='cpu'):
        super().__init__()
        self.gcn = TAGConv(in_channels, out_channels, K=1, normalize=False, bias = False)
        self.gat = Transformer_Block(in_channels, out_channels, device=device)

    def forward(self, input, edge_index, edge_index_2):
        x_1 = self.gcn(input, edge_index)
        x_2 = self.gat(input, edge_index_2, use_softmax=False, return_attn_matrix=False)
        x = x_1 - x_2
        return x

class GD_Unroll(nn.Module):
    def __init__(self, in_channels, out_channels, gd_steps, device='cpucs'):
        super().__init__()
        layers = []

        for _ in range(gd_steps):
            layers.append((GD_Block(in_channels, out_channels, device=device), 'x, edge_index, edge_index_2 -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2', [layer for layer in layers])

    def forward(self, input, edge_index, edge_index_2):
        x = input
        x = self.gd(x, edge_index, edge_index_2)
        return x