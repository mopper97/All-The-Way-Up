from torch.utils.data import Dataset, DataLoader
import pandas as pd
from bpemb import BPEmb
from tqdm import tqdm 
import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import ByteLevelBPETokenizer


def tok(sent, tokenizer):
    sent = " ".join(sent)
    encoded = tokenizer.encode(sent.lower()).ids
    return torch.tensor([torch.tensor(x) for x in encoded])


def sent_to_bpe(sent, bpe):
    sent = " ".join(sent)
    encoded = bpe.encode_ids(sent)
    return torch.tensor([torch.tensor(x) for x in encoded])

def process_dataset(data_path):
    dataset = []
    bpemb_en = BPEmb(lang='en', vs=25000, dim=100)
    # tokeniser = ByteLevelBPETokenizer("ha1m/vocab.json", "ha1m/merges.txt")
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines[:-1]):
            # parse the line
            line = line.strip('\n').split(' ')
            if 3 < len(line) <= 150:
                # emb = tok(line, tokeniser)
                emb = sent_to_bpe(line, bpemb_en)
                if emb.size(dim=0) > 2:
                    dataset.append(emb)

    return dataset

class StrAEDataset(Dataset):
    def __init__(self, data):
        self.sequences = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.sequences[index]

    def __len__(self):
        return self.n_samples


def collate_fn(data):
    return pad_sequence(data, batch_first=True, padding_value=25000)

def process_sts_dataset(data_path):
    df = pd.read_csv(data_path)
    bpemb_en = BPEmb(lang='en', vs=25000, dim=100)
    # tokeniser = ByteLevelBPETokenizer("ha1m/vocab.json", "ha1m/merges.txt")
    sents1 = [sent_to_bpe(x.split(' '), bpemb_en) for x in df['sent1']]
    sents2 = [sent_to_bpe(x.split(' '), bpemb_en) for x in df['sents2']]
    # sents1 = [tok(x.split(' '), tokeniser) for x in df['sent1']]
    # sents2 = [tok(x.split(' '), tokeniser) for x in df['sents2']]
    scores = [torch.tensor(x) for x in df['score']]
    dataset = [(sents1[x], sents2[x], scores[x]) for x in range(len(sents1))]
    return dataset


class STSDataset(Dataset):
    def __init__(self, data):
        self.sequences = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.sequences[index]

    def __len__(self):
        return self.n_samples


def collate_fn_sts(data):
    sents_1 = pad_sequence([x[0] for x in data], batch_first=True, padding_value=25000)
    sents_2 = pad_sequence([x[1] for x in data], batch_first=True, padding_value=25000)
    scores = torch.stack([x[2] for x in data], dim=0)
    return sents_1, sents_2, scores


def create_sts_dataloader(data_path, batch_size, shuffle=False):
    data = process_sts_dataset(data_path)
    dataset = STSDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_sts, shuffle=shuffle)
    return dataloader




def create_dataloader(data_path, batch_size, shuffle=False):
    data = process_dataset(data_path)
    dataset = StrAEDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    return dataloader


def set_seed(seed=None):
    rseed = seed if seed else torch.initial_seed()
    rseed = rseed & ((1 << 63) - 1)  # protect against uint64 vs int64 issues
    torch.manual_seed(rseed)
    return rseed
