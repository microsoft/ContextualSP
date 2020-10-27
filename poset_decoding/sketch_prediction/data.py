import re
import os
from torchtext import data
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import GloVe

from torch.utils.data import Dataset, DataLoader
import torch
from collections import defaultdict
import string
from utils import Trie, Tree



class Dictionary:
	def __init__(self, words):
		self.SPECIAL_SYMBOLS = 4
		self.PAD, self.OOV, self.EOS, self.SOS = 0, 1, 2, 3
		self.id2word = [None] + words
		self.word2id = {word: 1 + i for i, word in enumerate(words)}

	def tokenize(self, sentence):
		
		return sentence.split()

	def sentence2ids(self, sentence, eos=False, sos=False):
		tokens = self.tokenize(sentence)
		ids = [self.SPECIAL_SYMBOLS + self.word2id[word] - 1 if word in self.word2id else self.OOV for word in tokens]
		if eos:
			ids = ids + [self.EOS]
		if sos:
			ids = [self.SOS] + ids
		return ids

	def sentences2ids(self, sentences, eos=False, sos=False):
		ids = [self.sentence2ids(sentence, eos=eos, sos=sos) for sentence in sentences]
		lengths = [len(s) for s in ids]
		ids = [s + [self.PAD]*(max(lengths)-len(s)) for s in ids]  # Padding
		# ids = [[ids[i][j] for i in range(len(ids))] for j in range(max(lengths))]  # batch*len -> len*batch
		return ids, lengths

	def sentences2ids_for_multilabel(self, sentences, eos=False, sos = False):
		ids = [[self.sentence2ids(sentence, eos=False, sos=False) for sentence in group] for group in sentences]
		if sos:
			ids = [[[self.SOS]] + s for s in ids]
		if eos:
			ids = [s + [[self.EOS]]  for s in ids]

		
		lengths = [len(s) for s in ids]
		ids = [s + [[self.PAD]]*(max(lengths)-len(s)) for s in ids]
		max_label_length = 0
		for id in ids:
			for iid in id:
				max_label_length = max(len(iid), max_label_length) 
		ids = torch.LongTensor([[s + [self.PAD] * (max_label_length - len(s)) for s in id] for id in ids])# Padding

		targets = torch.zeros(len(ids), max(lengths) , self.size())
		temp = torch.ones(ids.shape)

		targets = targets.scatter_(2 , ids, 1)

		targets[:,:,0] = 0
		return targets, torch.tensor(lengths)

	def ids2sentence(self, ids):

		return ' '.join(['<OOV>' if i == self.OOV else self.id2word[i - self.SPECIAL_SYMBOLS + 1] for i in ids if i != self.EOS and i != self.PAD and i != self.SOS])

	def ids2sentences(self, ids):
		return [self.ids2sentence(i) for i in ids]

	def size(self):
		return len(self.id2word) + self.SPECIAL_SYMBOLS - 1



class treeDataset(Dataset):
	def __init__(self, src_path, trg_path):
		
	   	self.src_sents = []
	   	self.trg_sents = []
	   	self.candidate_token_sents = []

	   	with open(src_path) as fs, open(trg_path) as ft:
	   		for src_sent, trg_sent in zip(fs, ft):
	   			src_sent = src_sent.strip()
	   			trg_sent = trg_sent.strip()
	   			self.src_sents.append(src_sent)
	   			self.trg_sents.append(trg_sent)

	def __getitem__(self, index):

		src_sent = self.src_sents[index]
		trg_sent = self.trg_sents[index].split(" ### ")


		label_list = []

		sparql_tree = Trie(sos_token='<SOS>')
		for sub_path in trg_sent:
			sparql_tree.add_path(sub_path)

		for sub_path in trg_sent:
			label_list.append(sparql_tree.get_label(sub_path))
		 
		return [src_sent] * len(trg_sent), trg_sent, label_list, len(trg_sent)

	@classmethod
	def transform(cls, src_dictionary, trg_dictionary, src_sents, trg_sents, label_sents, device):


		src_ids, src_lengths = src_dictionary.sentences2ids(src_sents, sos=False, eos=True)
		src_lengths, idx_sort = torch.sort(torch.tensor(src_lengths), dim=0, descending=True)
		src_lengths = src_lengths.to(device)

		_, original_idx = torch.sort(idx_sort.data, dim=0)
		original_idx = original_idx.to(device)

		src_ids = torch.LongTensor(src_ids).index_select(0, idx_sort).transpose(0,1).to(device)


		
		trg_ids, trg_lengths = trg_dictionary.sentences2ids(trg_sents, eos=True, sos=True)
		trg_ids = torch.LongTensor(trg_ids).index_select(0, idx_sort).transpose(0,1).to(device)
		trg_lengths = torch.tensor(trg_lengths).index_select(0, idx_sort).to(device)
		label_ids, label_lengths = trg_dictionary.sentences2ids_for_multilabel(label_sents, eos = True, sos =False)

		label_lengths = label_lengths.index_select(0, idx_sort).to(device)
		label_ids = label_ids.index_select(0, idx_sort).transpose(0,1).to(device)
			
		return (src_ids, src_lengths), (trg_ids, trg_lengths), (label_ids, label_lengths) , original_idx


	def __len__(self):
		return len(self.src_sents)

class customDataset(Dataset):
	def __init__(self, src_path, trg_path):
		
	   	self.src_sents = []
	   	self.trg_sents = []
	   	self.candidate_token_sents = []
	   	
	  
	   	with open(src_path) as fs, open(trg_path) as ft:
	   		for src_sent, trg_sent in zip(fs, ft):
	   			src_sent = src_sent.strip()
	   			trg_sent = trg_sent.strip()
	   			self.src_sents.append(src_sent)
	   			self.trg_sents.append(trg_sent)
	   			

	def __getitem__(self, index):

		src_sent = self.src_sents[index]
		trg_sent = self.trg_sents[index]

		return src_sent, trg_sent

	@classmethod
	def transform(cls, src_dictionary, trg_dictionary, src_sents, device='cpu'):	 
		
		src_ids, src_lengths = src_dictionary.sentences2ids(src_sents, sos=False, eos=True)
		src_lengths, idx_sort = torch.sort(torch.tensor(src_lengths), dim=0, descending=True)
		src_ids = torch.LongTensor(src_ids).index_select(0, idx_sort).transpose(0,1).to(device)
		src_lengths = src_lengths.to(device)
		
		return (src_ids, src_lengths)
				


	def __len__(self):
		return len(self.src_sents)

