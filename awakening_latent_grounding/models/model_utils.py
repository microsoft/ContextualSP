import torch.nn as nn

from baseline.wtq_s2s.seq2seq import WTQSeq2SeqModel
from utils import *
from .spider_align import SpiderAlignmentModel
from .wtq_align import WTQAlignmentModel

_Model_mappings = {
    'SpiderAlignmentModel':
        {
            'model': SpiderAlignmentModel,
            'data_iter': load_spider_data_iterator,
            'evaluator': SpiderEvaluator
        },
    'WTQAlignmentModel':
        {
            'model': WTQAlignmentModel,
            'data_iter': load_wtq_data_iterator,
            'evaluator': WTQEvaluator
        },
    'WTQSeq2SeqModel':
        {
            'model': WTQSeq2SeqModel,
            'data_iter': load_wtq_data_iterator,
            'evaluator': WTQEvaluator
        }
}


def get_data_iterator_func(model: str):
    return _Model_mappings[model]['data_iter']


def get_evaluator_class(model: str):
    return _Model_mappings[model]['evaluator']


def load_model_from_checkpoint(model: str, device: torch.device, checkpoint: str = None, **args) -> nn.Module:
    if model in ['WTQSeq2SeqModel']:
        keyword_vocab_path = os.path.join(args['data_dir'], 'keyword.vocab.txt')
        keyword_vocab = Vocab.from_file(keyword_vocab_path,
                                        special_tokens=[SOS_Token, EOS_Token, UNK_Token, TBL_Token, VAL_Token],
                                        min_freq=5)
        info('load SQL keyword vocab from {} over, size = {}'.format(keyword_vocab_path, len(keyword_vocab)))

        suffix_type_vocab_path = os.path.join(args['data_dir'], 'suffix_type.vocab.txt')
        suffix_type_vocab = Vocab.from_file(suffix_type_vocab_path, special_tokens=[], min_freq=5)
        info('load Column suffix type vocab from {} over, size = {}'.format(suffix_type_vocab_path,
                                                                            len(suffix_type_vocab)))
        model_args = {'keyword_vocab': keyword_vocab, 'suffix_type_vocab': suffix_type_vocab}

        model = WTQSeq2SeqModel(bert_version=args['bert_version'], hidden_size=300, dropout_prob=args['dropout'],
                                **model_args)
        model.to(device)
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            info('Initialize {} from checkpoint {} over.'.format(model, checkpoint))
            return model

        if 'encoder_checkpoint' in args and args['encoder_checkpoint'] is not None:
            model.encoder.load_state_dict(torch.load(args['encoder_ckpt'], map_location=device))
            info('Initialize {} encoder from checkpoint {} over.'.format(model, args.encoder_ckpt))
            return model

        return model
    elif model in ['WTQAlignmentModel']:
        model = WTQAlignmentModel(args['bert_version'], dropout_prob=args['dropout'])
        model.to(device=device)
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            info('Initialize model from checkpoint {} over.'.format(checkpoint))
            return model
        return model

    elif model in ['SpiderAlignmentModel']:
        model = SpiderAlignmentModel(args['bert_version'], dropout_prob=args['dropout'])
        model.to(device)
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            info('Initialize model from checkpoint {} over.'.format(checkpoint))
            return model

        return model
    else:
        raise NotImplementedError("Not supported model: {}".format(model))
