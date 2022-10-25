# -*- coding: utf-8 -*-
import re
import unicodedata


def tsv_unescape(x):
    """Unescape strings in the TSV file.
    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash

    Args:
        x (str or unicode)
    Returns:
        a unicode
    """
    x = x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')
    if not isinstance(x, str):
        x = x.decode('utf-8', errors='ignore')
    return x


def tsv_unescape_list(x):
    """Unescape a list in the TSV file.
    List items are joined with vertical bars (0x5C)

    Args:
        x (str or unicode)
    Returns:
        a list of unicodes
    """
    return [tsv_unescape(y) for y in x.split('|')]


# From the official evaluator

def normalize(x):
    if not isinstance(x, str):
      x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x



PTB_BRACKETS = {
        '-lrb-': '(', '-rrb-': ')',
        '-lsb-': '[', '-rsb-': ']',
        '-lcb-': '{', '-rcb-': '}',
        }
def resolve_ptb_brackets(tokens):
    """Convert Penn Tree Bank escaped brackets to actual brackets."""
    if isinstance(tokens, str):
        tokens = tokens.split()
        if len(tokens) == 1:
            tokens = tsv_unescape_list(tokens[0])
    return [PTB_BRACKETS.get(x, x) for x in tokens]
