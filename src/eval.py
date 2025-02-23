import torch.nn.functional as f
import torch.optim
import torch.nn as nn
from utils import *
from scipy import stats
import numpy as np
from tokenizers import ByteLevelBPETokenizer



# Todo Implement a intrinsic evaluation class
class IntrinsicEvaluator:
    def __init__(self, device):
        self.device = device
        # lexical 
        self.sl_data = self.load_word_level('../data/simlex_adapted.tsv')
        self.ws_data = self.load_word_level('../data/wordsim_similarity_goldstandard.txt')
        self.wr_data = self.load_word_level('../data/wordsim_relatedness_goldstandard.txt')
        self.cos_words = nn.CosineSimilarity(dim=0)
        # sentence
        self.sts12_dataloader = create_sts_dataloader('../data/sts12_test.csv', 128)
        self.sts13_dataloader = create_sts_dataloader('../data/sts13_test.csv', 128)
        self.sts14_dataloader = create_sts_dataloader('../data/sts14_test.csv', 128)
        self.sts15_dataloader = create_sts_dataloader('../data/sts15_test.csv', 128)
        self.sts16_dataloader = create_sts_dataloader('../data/sts16_test.csv', 128)
        self.stsb_dataloader = create_sts_dataloader('../data/stsb-test.csv', 128)
        self.sick_dataloader = create_sts_dataloader('../data/sick_test.csv', 128)
        self.sem_dataloader = create_sts_dataloader('../data/semrel.csv', 128)
        self.cos_sents = nn.CosineSimilarity(dim=1)

    def load_word_level(self, path):
        dataset = []
        bpemb_en = BPEmb(lang='en', vs=25000, dim=100)
        with open(path, 'r') as f:
            for line in f.readlines():
                dataset.append((bpemb_en.encode_ids(line.split()[0].lower()), bpemb_en.encode_ids(line.split()[1].lower()), line.split()[2]))
        return dataset

    @torch.no_grad()
    def embed_word(self, word, model, embeddings):
        model.eval()
        if len(word) > 1:
            embed = model(torch.tensor(word, dtype=torch.long).to(self.device), words=True)
        else:
            embed = embeddings[word[0]]
        return embed.cpu()

    def evaluate_word_level(self, model):
        embeddings = model.embedding.weight.detach().cpu()
        sl_predictions = [self.cos_words(self.embed_word(x[0], model, embeddings), self.embed_word(x[1], model, embeddings)).item()
                          for x in self.sl_data]
        sl_score = stats.spearmanr(np.array(sl_predictions), np.array([x[2] for x in self.sl_data]))
        print('SimLex Score: {}'.format(sl_score.correlation.round(3)), flush=True)

        ws_predictions = [self.cos_words(self.embed_word(x[0], model, embeddings), self.embed_word(x[1], model,
                                                                                                   embeddings)).item()
                          for x in self.ws_data]
        ws_score = stats.spearmanr(np.array(ws_predictions), np.array([x[2] for x in self.ws_data]))
        print('Wordsim S Score: {}'.format(ws_score.correlation.round(3)), flush=True)

        wr_predictions = [self.cos_words(self.embed_word(x[0], model, embeddings), self.embed_word(x[1], model,
                                                                                                   embeddings)).item()
                          for x in self.wr_data]
        wr_score = stats.spearmanr(np.array(wr_predictions), np.array([x[2] for x in self.wr_data]))
        print('Wordsim R Score: {}'.format(wr_score.correlation.round(3)), flush=True)

        return sl_score.correlation, ws_score.correlation, wr_score.correlation

    @torch.no_grad()
    def embed_sts(self, model, dataloader):
        model.eval()
        predicted_sims = []
        all_scores = []
        for tokens_1, tokens_2, scores in dataloader:
            out = model(tokens_1.to(self.device), seqs2=tokens_2.to(self.device))
            predicted_sims.append(self.cos_sents(out[0], out[1]))
            all_scores.append(scores)
        return predicted_sims, all_scores

    def evaluate_sts(self, model, device):
        predicted_sims, all_scores = self.embed_sts(model, self.sts12_dataloader)
        sts12_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-12: {}'.format(sts12_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sts13_dataloader)
        sts13_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-13: {}'.format(sts13_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sts14_dataloader)
        sts14_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-14: {}'.format(sts14_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sts15_dataloader)
        sts15_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-15: {}'.format(sts15_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sts16_dataloader)
        sts16_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-16: {}'.format(sts16_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.stsb_dataloader)
        stsb_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-B: {}'.format(stsb_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sick_dataloader)
        sick_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                        np.array(torch.cat(all_scores, dim=0).cpu()))
        print('SICK-R: {}'.format(sick_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sem_dataloader)
        sem_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                        np.array(torch.cat(all_scores, dim=0).cpu()))
        print('SemRel: {}'.format(sem_score.correlation.round(3)), flush=True)

        return sts12_score.correlation, sts13_score.correlation, sts14_score.correlation, sts15_score.correlation, sts16_score.correlation, stsb_score.correlation, sick_score.correlation, sem_score.correlation