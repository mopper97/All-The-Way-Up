import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils import create_dataloader


def create_index(seqs):
    '''
    Creates the index tensor for updating the adjacency matrix
    param seqs: tensor of shape (batch_size, seq_len) containing seqs of embedding indices
    return: tensor of shape (batch_size, seq_len) containing inital indices
    '''
    unique_values = torch.unique(seqs)
    lookup = torch.zeros(unique_values.max() + 1, dtype=torch.long, device=seqs.device)
    lookup[unique_values] = torch.arange(len(unique_values), device=seqs.device)
    # set the final value to -1 to represent padding
    lookup[-1] = -1
    index = lookup[seqs]
    return index, unique_values[:-1] 


class Banyan(nn.Module):
    def __init__(self, e_dim):
        super(Banyan, self).__init__()
        self.E = e_dim 

    def compose(self, seqs):
        frontiers, leaves = create_index(seqs)
        print(frontiers)

    def forward(self, x):
        print(x)
        self.compose(x)

        quit()
        return x
            







dl = create_dataloader('../data/small_train.txt', 4, shuffle=True)
model = Banyan(16)

for i in dl:
    model(i)
    break