import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
import dgl
import dgl.function as fnc

from utils import create_dataloader


class LCCompose(nn.Module):
    def __init__(self, embedding_size, channel_size):
        super(LCCompose, self).__init__()
        self.E = embedding_size
        self.c = channel_size
        self.e = int(self.E / self.c)
        self.comp_l = nn.Parameter(torch.rand(self.e).uniform_(0.6, 0.9), requires_grad=True)
        self.comp_r = nn.Parameter(torch.rand(self.e).uniform_(0.6, 0.9), requires_grad=True)
        self.cb = nn.Parameter(torch.zeros(self.e), requires_grad=True)
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, in_feats, words=False):
        if words:
            l_c = in_feats[0] * F.sigmoid(self.comp_l)
            r_c = in_feats[1] * F.sigmoid(self.comp_r)
            return (l_c + r_c + self.cb).view(-1, self.E)

        N, children, _, _ = in_feats.shape
        assert children == 2, "Expected to have only 2 children"
        t_in = in_feats.transpose(0, 1)
        l_c = t_in[0] * F.sigmoid(self.comp_l)
        r_c = t_in[1] * F.sigmoid(self.comp_r)
        return self.dropout(((l_c + r_c) + self.cb).view(-1, self.E))

def tensor_delete(tensor, indices):
    mask = torch.ones(tensor.shape, dtype=bool)
    mask[indices] = False
    return tensor[mask].view(-1, tensor.shape[1])


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


def get_complete(frontiers):
    return ~torch.all(frontiers[:, 1:] == -1, dim=1)

@torch.no_grad()
def get_neighbours(index, retrieval, range_tensor):
    n_hood = torch.zeros(range_tensor.shape[0], 3, device=range_tensor.device)

    # step 1: gets left the composed indices 
    targs = index[range_tensor, retrieval].view(-1, 2)
    # now we need the left of the lefts 
    lefts = targs[:, 0] 
    l_s = index[range_tensor[::2], retrieval[::2]-1] # set to -1 this will retrieve <sos> from the node stack later on 
    l_s[retrieval[::2]-1 == -1] = -1 # protect against leak for the longest sequence
    rights = targs[:, 1]
    l_t = torch.cat((lefts.unsqueeze(1), l_s.unsqueeze(1), rights.unsqueeze(1)), dim=-1)
    # set the lefts
    n_hood[:l_t.shape[0]] = l_t
    
    r_s = index[range_tensor[::2], torch.clamp(retrieval[1::2]+1, max=index.shape[1]-1)] # protect against leak for the longest sequence
    r_s[retrieval[1::2]+1 == index.shape[1]] = -2 # set to -2 this will retrieve <eos> from the node stack later on
    r_t = torch.cat((rights.unsqueeze(1), lefts.unsqueeze(1), r_s.unsqueeze(1)), dim=-1)
    # set the rights
    n_hood[l_t.shape[0]:] = r_t
    return n_hood

@torch.no_grad()
def get_sims(nodes, index):
    # empty to tensor to perform the similarity check
    sims = torch.full((index.shape[0], index.shape[1], nodes.shape[1]), -np.inf, device=nodes.device)
    # fill sims with the actual node embedding values
    sims[index != -1] = nodes[index[index != -1]]
    # take similarity between adjacent nodes in each frontier 
    cosines = F.cosine_similarity(sims[:, :-1, :], sims[:, 1:, :], dim=2)
    # mask the padded values
    cosines = cosines.masked_fill_((sims == -np.inf).all(dim=2)[:, 1:], -np.inf)
    # get the most similar pairs in each frontier
    max_sim = torch.argmax(cosines, dim=1)
    # additionally get the retrieval tensor (max_sim, max_sim + 1)
    retrieval = torch.cat((max_sim.unsqueeze(0), (max_sim + 1).unsqueeze(0)), dim=0).T.reshape(-1)
    return max_sim.long(), retrieval.long()

@torch.no_grad()
def reduce_frontier(index, completion_mask, range_tensor, max_indices):
    # create a mask to perform frontier reduction
    batch_remaining_mask = torch.ones_like(index, dtype=torch.bool)
    # remove left child of composed sequences
    batch_remaining_mask[range_tensor[completion_mask], max_indices[completion_mask]] = False
    # for completed sequences remove padding element so shapes fit
    if torch.where(completion_mask == 0)[0].numel() != 0:
        batch_remaining_mask[torch.where(completion_mask == 0, True, False), -1] = False
    # reduce the index tensor
    index = index[batch_remaining_mask].view(index.shape[0], -1)
    return index


class Banyan(nn.Module):
    def __init__(self, e_dim, vocab_size, device, max_nodes=50000):
        super(Banyan, self).__init__()
        self.E = e_dim 
        self.V = vocab_size
        self.embedding = nn.Embedding(self.V, self.E)
        self.comp_fn = LCCompose(self.E, 16)
        # <sos> and <eos> tokens    
        self.sos = nn.Parameter(torch.zeros(self.E), requires_grad=True)
        self.eos = nn.Parameter(torch.zeros(self.E), requires_grad=True)
        self.dropout = nn.Dropout(0.1)
        self.device = device



    def update_graph(self, graph, retrieval, index):
        # range used for indexing
        range_tensor = torch.arange(index.shape[0], device=index.device, dtype=torch.long).repeat_interleave(2)
        # get src
        src = index[range_tensor, retrieval].view(-1,2)
        # get existing edges in the graph
        ex_src, ex_dst = graph.edges() if graph.num_edges() > 0 else (None, None)
        # get the new src and dst 
        if ex_src is not None:
            # check whether edges already exist in graph
            mask = ~torch.eq(src.unsqueeze(1), ex_src.view(-1, 2)).all(dim=2).any(dim=1)
            # filter src according to mask
            src = src[mask]
            # make sure the new edges are unique
            src = torch.unique(src, dim=0) 
            # set indices for the new nodes
            dst = torch.max(ex_dst) + 1 + torch.arange(src.shape[0], device=src.device)
        else:
            src = torch.unique(src, dim=0)
            dst = torch.max(index) + 1 + torch.arange(src.shape[0], device=src.device)
        # update the graph
        # 1. add the new nodes and their representations 
        graph.add_nodes(dst.shape[0], {'comp': self.comp_fn(graph.ndata['comp'][src].view(-1, 2, 16, 16))})
        # 2. add the new edges
        graph.add_edges(src.flatten(), dst.repeat_interleave(2).flatten())
        # save neighbours 
        # print('ind n block')
        # print(index)
        n_hood = get_neighbours(index, retrieval, range_tensor)
        # print(n_hood)
        # print('\n')
        # update index tensor 
        # 1. recreate the original src tensor
        src = index[range_tensor, retrieval].view(-1,2)
        # 2. find which edges contain src
        ex_src, ex_dst = graph.edges()
        locs = torch.where(src.unsqueeze(1) == ex_src.view(-1, 2), 1, 0).all(dim=-1).nonzero()[:, 1]
        # 3. get the corresponding values from dst and fill index accordingly 
        update = ex_dst.view(-1, 2)[locs] 
        index[range_tensor, retrieval] = update.view(-1)
        # return the updated graph and index tensor
        return graph, index, n_hood



    def compose(self, seqs, roots=False):
        # we need the range for several operations and no point casting it to cuda each time
        range_tensor = torch.tensor(range(seqs.shape[0]), dtype=torch.long, device=self.device)
        # embed the initial froniter 
        frontier = self.embedding(seqs) # shape (b_size, seq_len, E)
        # get the node indexes for the embeddings and the tokens they map to 
        index, tokens = create_index(seqs) # index shape (b_size, seq_len) tokens shape set(tokens in seqs)
        # create the graph 
        g = dgl.graph(([], []), device=self.device) # empty graph
        g.add_nodes(tokens.shape[0]) # add the nodes
        g.ndata['comp'] = self.dropout(self.embedding(tokens)) # add the embeddings for the nodes
        targets = []
        # reduce the frontiers till we get to the root 
        while index.shape[1] != 1:
            # get the merge indices
            max_sim, retrieval = get_sims(g.ndata['comp'].detach(), index)
            # get the completion mask
            completion_mask = get_complete(index)
            # update the graph 
            g, index[completion_mask], n_hood = self.update_graph(g, retrieval[completion_mask.repeat_interleave(2)], index[completion_mask])
            # add neighbours to targets
            targets.append(n_hood)
            # reduce the frontiers
            index = reduce_frontier(index, completion_mask, range_tensor, max_sim)
        # cat the targets
        targets = torch.cat(targets, dim=0)
        # add the <sos> and <eos> tokens to the graph
        g.add_nodes(2, {'comp': torch.stack([self.eos, self.sos])})
        return g, targets.long()




    def forward(self, x):
        # reset cache 
        g, targets = self.compose(x)
        nodes = g.ndata['comp']
        # print('final')
        # print(targets)
        lefts = nodes[targets[:, 1]]
        rights = nodes[targets[:, 2]]
        outs = self.comp_fn(torch.cat((lefts, rights), dim=1).view(-1, 2, 16,16))
        outs = outs @ nodes.T
        return F.cross_entropy(outs, targets[:, 0])







dl = create_dataloader('/Users/mopper/Desktop/All-The-Way-Up/data/small_train.txt', 256, shuffle=True)
model = Banyan(256, 25001, 'cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
# with torch.autograd.set_detect_anomaly(True):
for e in range(10):
    epoch_loss = 0
    for i in tqdm(dl):
        # model(i)
        optimizer.zero_grad()
        loss = model(i)
        loss.backward()
        # loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(epoch_loss / len(dl))

