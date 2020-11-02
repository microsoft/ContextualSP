# coding: utf-8

# from pattern.en import lemma
import spacy

sp_english = spacy.load('en_core_web_sm')

STOP_WORD_LIST = [_.strip() for _ in open('data/common/stop_words.txt', 'r', encoding='utf-8').readlines() if _[0] != '#']
TEMPLATE_KEYWORDS = ['find', 'out', 'the', 'common', 'part', 'of', 'set', 'and', 'everyone', 'in', 'but', 'not',
                     'satisfying', 'between', 'like'] + STOP_WORD_LIST


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lemma_token(token):
    token = token.lower()
    # token_lemma = lemma(token)
    token_lemma = sp_english(token)[0].lemma_
    if token_lemma:
        return token_lemma.lower()
    else:
        return token.lower()
