import random

import torch
from torch import nn
from torch.nn.functional import softmax
from utils import Trie, Tree
MAX_LEN = 256


class Parser(nn.Module):
    def __init__(self, src_dictionary, trg_dictionary, model, device):
        super().__init__()
        self.src_dictionary = src_dictionary
        self.trg_dictionary = trg_dictionary
        self.model =  model
        self.device = device


    def forward(self, src_info, trg_info, teacher_force_rate = 1):
        
        # src_info, trg_info, label_info, ori_idx = self.transform(src, trg, label)
        src_ids, src_lengths = src_info
        trg_ids, trg_lengths = trg_info
      

        output, _  = self.model(src_info, trg_ids, teacher_force_rate)
        # print("output shape:", output.shape)

        return output

    def inference(self, src_info):
        src_ids, src_lengths = src_info
        encoder_outputs, hidden = self.model.encoder(src_ids, src_lengths)
        # print(self.device)
        root = Tree(torch.LongTensor([self.trg_dictionary.SOS]).to(self.device))
        mask = self.model.create_mask(src_ids)


        with torch.no_grad():
            alpha = torch.ones(self.trg_dictionary.size()).to(src_ids)
        self.build_tree(root, hidden, encoder_outputs, mask, alpha, 0)

        return Trie(node = root).get_path()
        
        

    def build_tree(self, tree_node, hidden, encoder_outputs, mask, alpha, depth):
        if depth > 20:
            return

        input = tree_node.value
        if input == self.trg_dictionary.EOS:
            return

        output, hidden, _ = self.model.decoder(input, hidden, encoder_outputs, mask)
        output = alpha * torch.sigmoid(output.squeeze(0))

        for i in range(output.shape[0]):
            if output[i] > 0.5:

                child = Tree(torch.LongTensor([i]).to(self.device))
                tree_node.children[i] = child
        for k, child in tree_node.children.items():
            self.build_tree(child, hidden, encoder_outputs, mask, alpha, depth+1)



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx



    def forward(self, src_, trg = None, teacher_forcing_ratio=1):

        src, src_len = src_
        batch_size = src.shape[1]
        max_len = trg.shape[0] if trg is not None else MAX_LEN
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len-1, batch_size, trg_vocab_size).to(self.device)
        attn_scores = []
        encoder_outputs, hidden = self.encoder(src, src_len)
        input = trg[0, :] if trg is not None else src[0, :]
        mask = self.create_mask(src)
        for t in range(1, max_len):
            output, hidden, attn_score = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t-1] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            attn_scores.append(attn_score)
        return outputs, torch.cat(attn_scores, dim = 1).to(self.device)

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask


class Encoder(nn.Module):
    def __init__(self, input_dim, src_emb_dim, enc_hid_dim, dec_hid_dim, dropout, pad_idx, embed =None):
        super().__init__()
        self.nl_embedding = nn.Embedding(input_dim, src_emb_dim)
        if embed is not None:
            print('using the glove embedding')
            self.nl_embedding.weight.data.copy_(embed)
        else:
            print("not using glove embedding")

        self.vocab_size = input_dim
        self.emb_dim = src_emb_dim 

        self.rnn = nn.GRU(self.emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout_src = nn.Dropout(dropout)


    def forward(self, src, src_len):
        embedded = self.dropout_src(self.nl_embedding(src)* (self.vocab_size ** 0.5))
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.data, batch_first=False)
        outputs, hidden = self.rnn(embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden



class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        attention = attention.masked_fill(mask == 0, -1e10)
        return softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch_size]
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [src_sent_len, batch_size, enc_hid_dim * 2]
        # print(input.shape, hidden.shape, encoder_outputs.shape)
        input = input.unsqueeze(0)  # [1, batch_size]

        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]
        # a = self.attention(hidden, encoder_outputs)  # [batch_size, src_len]
        a = self.attention(hidden, encoder_outputs, mask)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
       
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # print("rnn_input shape:", rnn_input.shape)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output = self.out(torch.cat((output, weighted, embedded), dim=1))
        return output, hidden.squeeze(0), a
