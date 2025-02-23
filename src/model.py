import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm

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
        self.comp_fn = LCCompose(self.E, 4)
        # node cache: to hold node features 
        self.register_buffer('n_c', torch.zeros(max_nodes, self.E, device=device, requires_grad=False))
        # target cache: to hold the target values for the nodes
        self.register_buffer('t_c', torch.full((max_nodes, 3), -1, device=device, requires_grad=False)) # [target, left, right] 
        # composition cache: to hold the children of the nodes
        self.register_buffer('c_c', torch.full((max_nodes, 2), -1, device=device, requires_grad=False)) # [left, right]
        # just some other junk
        self.dropout = nn.Dropout(0.1)
        self.device = device

    @torch.no_grad()
    def reset_cache(self):
        # self.n_c.fill_(0)
        # self.t_c.fill_(-1)
        # self.c_c.fill_(-1)
        # Instead of in-place fill_
        self.n_c = torch.zeros_like(self.n_c)
        self.t_c = torch.full_like(self.t_c, -1)
        self.c_c = torch.full_like(self.c_c, -1)

    def update_graph(self, retrieval, index, md):
        # range used for indexing
        range_tensor = torch.arange(index.shape[0], device=index.device, dtype=torch.long).repeat_interleave(2)
        # get src
        src = index[range_tensor, retrieval].view(-1,2)
        # check whether parents already cached and filter
        mask = ~torch.eq(src.unsqueeze(1), self.c_c).all(dim=2).any(dim=1) # this is weird check sober
        src = src[mask]
        # make sure src is unique (no duplicate parents)
        src = torch.unique(src, dim=0) 
        if not src.numel() == 0:
            # set indices for the new nodes 
            dst = md + 1 + torch.arange(src.shape[0], device=src.device)
            md = torch.max(dst)
            self.n_c[dst] = self.comp_fn(self.n_c[src].view(-1, 2, 4, 4))
            # add composition indices to the cache
            self.c_c[dst] = src
            # add target indices to the cache
            l_t  = torch.cat((retrieval[::2], (retrieval[::2]-1), retrieval[1::2]), dim=0)
            # make sure right neighbour is not out of bounds
            r_n = retrieval[1::2] + 1
            r_n[r_n > torch.max(index)] = -1  
            r_t = torch.cat((retrieval[1::2], retrieval[::2], r_n), dim=0)
            # apply same filtering as to src to ensure no duplicates
            t = torch.cat((l_t, r_t), dim=0).view(-1, 3)
            t = torch.unique(t, dim=0)
            mask = ~torch.eq(t.unsqueeze(1), self.t_c).all(dim=2).any(dim=1)
            t = t[mask]
            # add target indices to the cache
            adds = (self.t_c != -1).all(dim=1).sum()
            self.t_c[adds: adds + t.shape[0]] = t
        return index, md


    def compose(self, seqs):
        # we need the range for several operations and no point casting it to cuda each time
        range_tensor = torch.tensor(range(seqs.shape[0]), dtype=torch.long, device=self.device)
        # index - tensor that is going to keep track of which nodes constitute the frontier
        # leaves - indices to get the initial embeddings from the embedding matrix
        index, leaves = create_index(seqs)
        # add the initial leaves to the node cache
        self.n_c[:leaves.shape[0]] = self.dropout(self.embedding(leaves))
        # get the max of index as the initial max dist
        md = torch.max(index)
        # reduce the frontiers till we get to the root 
        while index.shape[1] != 1:
            # get the merge indices
            max_sim, retrieval = get_sims(self.n_c.detach(), index)
            # get the completion mask
            completion_mask = get_complete(index)
            # update the graph 
            # check corner case where max index is masked out by completion mask and therefore you might be overwriting rows in the cache
            index[completion_mask], md = self.update_graph(retrieval[completion_mask.repeat_interleave(2)], index[completion_mask], md)
            # reduce the frontiers
            index = reduce_frontier(index, completion_mask, range_tensor, max_sim)

        
        # return all filled values of node cache 
        nodes = self.n_c[:md + 1]
        # return all filled values of target cache
        targets = self.t_c[(self.t_c != -1).all(dim=1)]
        return nodes, targets



    def forward(self, x):
        # reset cache 
        self.reset_cache()
        nodes, targets = self.compose(x)
        lefts = nodes[targets[:, 1]]
        rights = nodes[targets[:, 2]]
        outs = self.comp_fn(torch.cat((lefts, rights), dim=1).view(-1, 2, 4,4))
        outs = outs @ nodes.T
        return F.cross_entropy(outs, targets[:, 0])
            







dl = create_dataloader('/Users/mopper/Desktop/All-The-Way-Up/data/small_train.txt', 2, shuffle=True)
model = Banyan(16, 25001, 'cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
# with torch.autograd.set_detect_anomaly(True):
for e in range(10):
    epoch_loss = 0
    for i in tqdm(dl):
        optimizer.zero_grad()
        loss = model(i)
        loss.backward()
        # loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(epoch_loss / len(dl))

