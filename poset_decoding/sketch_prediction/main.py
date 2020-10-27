import torch
import sys
from torch import nn, optim
import os
from data import treeDataset, Dictionary, customDataset
from torch.utils.data import DataLoader
from model import Seq2Seq,  Encoder, Decoder, Attention, Parser
from utils import collate_fn
import argparse
import numpy as np
import sys
import time
import random
import json


SPOUSE_PRED="people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses"
SIBLING_PRED="people.person.sibling_s/ns:people.sibling_relationship.sibling|ns:fictional_universe.fictional_character.siblings/ns:fictional_universe.sibling_relationship_of_fictional_characters.siblings"
N_EPOCHS = 50
TRAIN='train'
TEST='test'
DEBUG='debug'
INFERENCE='inference'
def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device :
        torch.cuda.manual_seed_all(seed)


def train(src_dictionary, trg_dictionary, train_data_loader, dev_data_loader, model,  device, args, clip = 1):

    """
    Train a base simple neural end-to-end semantic parser from (utterance, logic form) dataset.
    :param en_path: utterance file path, see 'data/geo/geoquery.abstract.en' as an example.
    :param funql_path: logic form file path, see 'data/geo/geoquery.abstract.funql' as an example.
    :param train_ids_path: train data sampling file path, see 'data/geo/split880_train_ids.txt' as an example.
    :param model_output_path: where the trained simple parser model should be stored.
    :param model_init_path: if not None, load an initial model from this path and fine-tune it.
    :param refine_data: additional (utterance, logic form) data for training.
    :return:
    """
    # set_seed(args.seed, device)
    # model = model.to(device)
    
    optimizer = optim.Adam(model.parameters())
    
    # print("class_Weight shape:", class_weight.shape)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.weight]).to(device))
    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(args.alpha, args.gamma)


    hit, total, acc = evaluate_iter_loss2(model, dev_data_loader, src_dictionary ,trg_dictionary, device)
    best_hit = hit
    torch.save(model, args.save_path +'parser_model_best.pt')
    print(f'Best Dev hit: {best_hit: d} | Dev total: {total: d} | Dev acc: {acc: f}', flush=True)
    
    for epoch in range(args.iterations):
        train_loss = train_iter(model, train_data_loader, src_dictionary, trg_dictionary, optimizer, criterion, clip, device)
        # if loss < best_loss:
        #     best_loss = loss
        #     torch.save(model, model_output_path)
        print(f'EPOCH: {epoch: d} |  Train Loss: {train_loss: f}', flush=True)
        if epoch %5 == 0:
            torch.save(model, args.save_path +f'parser_model_{epoch}.pt')
        # hit, count, _ = evaluate_iter_loss(model, dev_iterator, trg_dictionary)
        hit, total, acc = evaluate_iter_loss2(model, dev_data_loader,  src_dictionary, trg_dictionary, device)
        if hit >  best_hit:
            best_hit = hit
            # best_loss = loss
            torch.save(model, args.save_path +'parser_model_best.pt')
        
        print(f'Epoch: {epoch: d}  | Best Dev hit: {best_hit: d}| Dev hit: {hit: d} |Dev total: {total: d} | Dev acc: {acc: f}', flush=True)


def create_index_mask(modified_labels):

    pad_mask = modified_labels.sum(dim = -1).gt(0)

    # pad_mask [src_len, bsz, vocab_size]
    pad_mask = pad_mask.unsqueeze(2).repeat(1, 1, modified_labels.shape[-1])
    
    # index_matrix = torch.tensor([i for i in range(modified_labels.shape[0])] * modified_labels.shape[1])
    # indices = torch.arange(0,pad_mask.size(0)).expand(pad_mask.shape[1], -1).transpose(0,1)[pad_mask].long()
    # print("indices:", indices)
    return pad_mask
    # pad_mask = sum(modified_labels[0,:,:]).eq(0)


def train_iter(model, iterator, src_dictionary, trg_dictionary, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        nl, trg, label, _= batch
        
        src_info, trg_info, label_info, _ = treeDataset.transform(src_dictionary, trg_dictionary, nl, trg, label, device)
        modified_labels, _ = label_info

        optimizer.zero_grad()
        output = model(src_info, trg_info, 1)
        # print(output.shape, modified_labels.shape)
        ## output_labels, modified_labels :[bsz, trg_length, vocab_size]
        output = output.transpose(0,1)
        modified_labels = modified_labels.transpose(0, 1)
        pad_mask = create_index_mask(modified_labels)

        loss = criterion(output[pad_mask], modified_labels[pad_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() 
    return epoch_loss / len(iterator)

def debug_iter(model, iterator, src_dictionary,trg_dictionary, device):
    model.eval()
   
    for i, batch in enumerate(iterator):
        nl, trg, candidate_tokens, label, ori_idx= batch
        # print("nl:", nl)
        # print("trg:", trg)
        # print("label:", label)

        src_info, trg_info, label_info, ori_idx = treeDataset.transform(src_dictionary, trg_dictionary, nl, trg, candidate_tokens, label, device)
        
        modified_labels, _ = label_info
      

        # optimizer.zero_grad()
        output = model(src_info, trg_info,   1)

        # print("fff",ori_idx.dtype, modified_labels.dtype)

        output_labels = torch.sigmoid(output.transpose(0,1))
        modified_labels = modified_labels.transpose(0,1)

        

        # ori_idx = ori_idx.to(modified_labels).long()
        # # print("fff",ori_idx.dtype, modified_labels.dtype)
        # # print(ori_idx.dtype, modified_labels.dtype)
        modified_labels = modified_labels.index_select(0, ori_idx)
        output_labels = output_labels.index_select(0, ori_idx)
        pad_mask = create_index_mask(modified_labels)


        # loss = criterion(output, modified_labels)

        for batch_nl, batch_trg, batch_raw_label, batch_pred, batch_label in zip(nl, trg, label, output_labels, modified_labels):
            # print(f"nl:{batch_nl}\ntrg:{batch_trg}\ncandidate:{batch_candidate}")
            for step in range(len(batch_raw_label)):
                print(f"nl:{batch_nl}\ntrg:{batch_trg}\n")
                print("pos labels:", [(idx, model.trg_dictionary.ids2sentence([idx])) for idx, i in enumerate(batch_label[step]) if i==1])
                print(f"\nstep:{step}, step_label:{batch_raw_label[step]}")
                print([(idx, model.trg_dictionary.ids2sentence([idx])) for idx, i in enumerate(batch_pred[step]) if i> 0.5])
                for idx in range(93):
                    print(f"idx:{idx},token:{model.trg_dictionary.ids2sentence([idx])}, batch_label:{batch_label[step][idx]}, batch_pred_loss:{batch_pred[step][idx]}")

        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # optimizer.step()
        # epoch_loss += loss.item() 
    return epoch_loss / len(iterator)


def evaluate_iter_loss(model, iterator, criterion, trg_dictionary):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            nl, trg, label, _= batch
            output, modified_labels, _ = model(nl, trg, label, 1)  # turn off teacher forcing       
            pad_mask = create_index_mask(modified_labels)
            loss = criterion(output[pad_mask], modified_labels[pad_mask])
            epoch_loss += loss.item()
    # print("len iterator:", len(iterator))
                
    return epoch_loss  / len(iterator)

def evaluate_iter_loss2(model, iterator, src_dictionary, trg_dictionary, device):
    model.eval()
    epoch_loss = 0
    hit = 0
    total = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):

            nl, trg, label, dup_len= batch
         
            src_info, trg_info, label_info, ori_idx = treeDataset.transform(src_dictionary, trg_dictionary, nl, trg, label, device)
            modified_labels, _ = label_info

            output = model(src_info, trg_info, 1)  # turn off teacher forcing
            # print("output:", output)
            output_labels = torch.sigmoid(output).transpose(0, 1).index_select(0, ori_idx)
            modified_labels = modified_labels.transpose(0,1).long().index_select(0, ori_idx)
            alpha = torch.ones(output_labels.shape[0], output_labels.shape[2]).to(output_labels)
            alpha[:, 0] = 0
            alpha = alpha.unsqueeze(1).repeat(1, output_labels.shape[1], 1)
            
            output_labels = (alpha * output_labels).gt(0.5).long()
            # output_labels = (output_labels).gt(0.5).long()

            pad_mask = create_index_mask(modified_labels)

            start_pos = 0
            for idx, length in enumerate(dup_len):
                predict_labels_idx = output_labels[start_pos:start_pos+length, :, :]
                modified_labels_idx = modified_labels[start_pos:start_pos+length, :, :]
                pad_mask_idx = pad_mask[start_pos:start_pos+length, :, :]

                if predict_labels_idx[pad_mask_idx].equal(modified_labels_idx[pad_mask_idx]):
                    hit += 1
                
                total += 1

                start_pos += length


    return hit, total ,hit / total



def evaluate_iter_acc(model, iterator, src_dictionary, trg_dictionary, device, output_path):
    model.eval()
    hit = 0
    total = 0
    errors = {}
    ans, golden = [], []
    error_ids = []

    def calculate_acc(predict, golden):
        hit, total, errors = 0, 0, 0

        for p, g in zip(predict, golden):
            if p.strip() == g.strip():
                hit+=1
            total += 1

        return hit, total, hit / total
    # golden = open(GOLDEN_PATH).readlines()
    
    with torch.no_grad():
        for _, batch in enumerate(iterator):

            nl, trg = batch
            src_info = customDataset.transform(src_dictionary, trg_dictionary, nl, device)
            output = model.inference(src_info)  # turn off teacher forcing
            output = trg_dictionary.ids2sentences(output)
           

            sparql_set = set()

            for s in output:
                triples = s.split(" . ")
                for triple in triples:
                    sparql_set.add(triple)

            predict_sent = sorted(list(sparql_set))

            num = len(predict_sent)
            for i in range(num):
                pred = predict_sent[i]
                if SPOUSE_PRED in pred or SIBLING_PRED in pred:
                    try:
                        a1, r, a2 = pred.split()
                        predict_sent.append(f"FILTER ( {a1} != {a2} )")
                    except Exception as e:
                        pass

            predict_sent = ' . '.join(predict_sent)
            # print("trg:", trg)
            # if predict_sent.strip() == trg[0].strip():
            #     flag = True
            # else:
            #     flag = False
            #     error_ids.append(str(_))
            # print(f"\n{'='*60}")
            # print("idx:", _, "original_output:", output)
            # print(f"!!!!!!!!!!!!!!{flag}!!!!!!!!!!!!!\nNL:{nl}\nTRG:{trg[0]}\nPREDICT:{predict_sent}", flush=True)
            ans.append(predict_sent.strip())
            golden.append(trg[0].strip())
    
    open(output_path, "w").write('\n'.join(ans))
            
    return calculate_acc(ans, golden)


def evaluate(src_dictionary,trg_dictionary, dev_data_loader, model, device):
 
    hit, total, acc = evaluate_iter_loss2(model, dev_data_dataloader, trg_dictionary)

    print(f'hit: {hit: d} |  total: {total: d} | acc: {acc: f}', flush=True)



def main():
    parser = argparse.ArgumentParser(description='Train a neural machine translation model')

    # Training corpus
    corpora_group = parser.add_argument_group('training corpora', 'Corpora related arguments; specify either monolingual or parallel training corpora (or both)')
    corpora_group.add_argument('--src_path', help='the source language monolingual corpus')
    corpora_group.add_argument('--trg_path', help='the target language monolingual corpus')
    corpora_group.add_argument('--max_sentence_length', type=int, default=90, help='the maximum sentence length for training (defaults to 50)')
    
    # Embeddings/vocabulary
    embedding_group = parser.add_argument_group('embeddings', 'Embedding related arguments; either give pre-trained cross-lingual embeddings, or a vocabulary and embedding dimensionality to randomly initialize them')
    embedding_group.add_argument('--src_vocabulary', help='the source language vocabulary')
    embedding_group.add_argument('--trg_vocabulary', help='the target language vocabulary')
    embedding_group.add_argument('--embedding_size', type=int, default=0, help='the word embedding size')
      
    # Architecture
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--layers', type=int, default=2, help='the number of encoder/decoder layers (defaults to 2)')
    architecture_group.add_argument('--enc_hid_dim', type=int, default=512, help='the number of dimensions for the hidden layer (defaults to 600)')
    architecture_group.add_argument('--dec_hid_dim', type=int, default=512, help='the number of dimensions for the hidden layer (defaults to 600)')

    # Optimization
    optimization_group = parser.add_argument_group('optimization', 'Optimization related arguments')
    optimization_group.add_argument('--batch_size', type=int, default=128, help='the batch size (defaults to 50)')
    optimization_group.add_argument('--learning_rate', type=float, default=0.0002, help='the global learning rate (defaults to 0.0002)')
    optimization_group.add_argument('--dropout', metavar='PROB', type=float, default=0.4, help='dropout probability for the encoder/decoder (defaults to 0.3)')
    optimization_group.add_argument('--param_init', metavar='RANGE', type=float, default=0.1, help='uniform initialization in the specified range (defaults to 0.1,  0 for module specific default initialization)')
    optimization_group.add_argument('--iterations', type=int, default=50, help='the number of training iterations (defaults to 300000)')
    # Model saving
    saving_group = parser.add_argument_group('model saving', 'Arguments for saving the trained model')
    saving_group.add_argument('--save_path', metavar='PREFIX', help='save models with the given prefix')
    saving_group.add_argument('--save_interval', type=int, default=0, help='save intermediate models at this interval')
    saving_group.add_argument('--model_init_path', help='model init path')

    
    # Logging/validation
    logging_group = parser.add_argument_group('logging', 'Logging and validation arguments')
    logging_group.add_argument('--log_interval', type=int, default=1000, help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--validate_batch_size', type=int, default=1, help='the batch size (defaults to 50)')
    corpora_group.add_argument('--inference_output', help='the source language monolingual corpus')
    corpora_group.add_argument('--validation_src_path', help='the source language monolingual corpus')
    corpora_group.add_argument('--validation_trg_path', help='the source language monolingual corpus')

    # Other
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--type", type=str, default='train', help="type: train/inference/debug")

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    src_dictionary = Dictionary([word.strip() for word in open(args.src_vocabulary).readlines()])
    trg_dictionary = Dictionary([word.strip() for word in open(args.trg_vocabulary).readlines()])

    
    
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    

    if not args.model_init_path:
        attn = Attention(args.enc_hid_dim, args.dec_hid_dim)
        enc = Encoder(src_dictionary.size(), args.embedding_size, args.enc_hid_dim, args.dec_hid_dim, args.dropout, src_dictionary.PAD)
        dec = Decoder(trg_dictionary.size(), args.embedding_size, args.enc_hid_dim, args.dec_hid_dim, args.dropout, attn)
        s2s = Seq2Seq(enc, dec, src_dictionary.PAD, device)
        parallel_model = Parser(src_dictionary, trg_dictionary, s2s, device)
        parallel_model.apply(init_weights)

    else:
        print(f"load init model from {args.model_init_path}")
        parallel_model = torch.load(args.model_init_path)

    parallel_model = parallel_model.to(device)

        

    if args.type ==TEST:
        test_dataset = treeDataset(args.validation_src_path, args.validation_trg_path)
        test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = args.validate_batch_size,collate_fn = collate_fn)
        hit, total, acc = evaluate_iter_loss2(parallel_model, test_dataloader, src_dictionary, trg_dictionary, device)
        print(f'hit: {hit: d} |  total: {total: d} | acc: {acc: f}', flush=True)

    elif args.type==INFERENCE:
        test_dataset = customDataset(args.validation_src_path, args.validation_trg_path)
        test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = args.validate_batch_size)
        hit, total, acc = evaluate_iter_acc(parallel_model, test_dataloader, src_dictionary, trg_dictionary, device, args.inference_output)
        print(f'hit: {hit: d} |  total: {total: d} | acc: {acc: f}', flush=True)
    elif args.type == DEBUG:
        test_dataset = treeDataset(args.validation_src_path, args.validation_trg_path)
        test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = args.validate_batch_size,collate_fn = collate_fn)
        hit, total, acc = debug_iter(parallel_model, test_dataloader, src_dictionary, trg_dictionary, device)
        print(f'hit: {hit: d} |  total: {total: d} | acc: {acc: f}', flush=True)


    else:
        train_dataset = treeDataset(args.src_path, args.trg_path)
        train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = args.batch_size, collate_fn = collate_fn)
        test_dataset = treeDataset(args.validation_src_path, args.validation_trg_path)
        test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = args.validate_batch_size,collate_fn = collate_fn)
      
        train(src_dictionary,trg_dictionary, train_dataloader, test_dataloader, parallel_model, device, args)

if __name__ == '__main__':
    main()

    # evaluate(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))












