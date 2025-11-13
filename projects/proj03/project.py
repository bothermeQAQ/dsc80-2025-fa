# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    if not hasattr(get_book, "_delay"):
        get_book._delay = {}
    if not hasattr(get_book, "_last"):
        get_book._last = {}
    m = re.match(r'^(https?://[^/]+)', url)
    base = m.group(1) if m else ""
    if base not in get_book._delay:
        d = 0.5
        try:
            r = requests.get(f"{base}/robots.txt", timeout=10)
            if r.status_code == 200:
                block = re.search(r'(?ims)^user-agent:\s*\*\s*$([\s\S]*?)(?=^\s*user-agent:|\Z)', r.text)
                if block:
                    cd = re.search(r'^crawl-delay:\s*([0-9]+(?:\.[0-9]+)?)', block.group(1), flags=re.MULTILINE | re.IGNORECASE)
                    if cd:
                        d = float(cd.group(1))
        except Exception:
            pass
        get_book._delay[base] = d
    last = get_book._last.get(base)
    if last is not None:
        w = get_book._delay[base] - (time.monotonic() - last)
        if w > 0:
            time.sleep(w)
    resp = requests.get(url, headers={"User-Agent": "dsc80-project3"}, timeout=30)
    get_book._last[base] = time.monotonic()
    t = resp.text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    s = re.search(r'\*\*\*\s*START OF .*?PROJECT GUTENBERG .*?\*\*\*', t, flags=re.IGNORECASE)
    e_iter = list(re.finditer(r'\*\*\*\s*END OF .*?PROJECT GUTENBERG .*?\*\*\*', t, flags=re.IGNORECASE))
    if s and e_iter:
        e = e_iter[-1]
        if e.start() >= s.end():
            return t[s.end():e.start()]
    return t


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    s = book_string.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p for p in re.split(r"\n{2,}", s) if p.strip() != ""]
    pat = re.compile(r"\w+|[^\s\w]", re.UNICODE)
    out = []
    for p in paras:
        out.append("\x02")
        out.extend(pat.findall(p))
        out.append("\x03")
    return out if out else ["\x02", "\x03"]


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):

    def __init__(self, tokens):
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        u = pd.unique(list(tokens))
        n = len(u)
        return pd.Series(np.full(n, 1.0 / n, dtype=float), index=u)
    
    def probability(self, words):
        if not words:
            return 0.0
        probs = self.mdl.reindex(list(words))
        if probs.isna().any():
            return 0.0
        return float(np.prod(probs.values))
        
    def sample(self, M):
        M = int(M)
        if M <= 0:
            return ""
        draws = np.random.choice(self.mdl.index.values, size=M, replace=True, p=self.mdl.values)
        return " ".join(map(str, draws))


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        s = pd.Series(list(tokens))
        return s.value_counts(normalize=True).astype(float)
    
    def probability(self, words):
        if not words:
            return 0.0
        probs = self.mdl.reindex(list(words))
        if probs.isna().any():
            return 0.0
        return float(np.prod(probs.values))
        
    def sample(self, M):
        M = int(M)
        if M <= 0:
            return ""
        draws = np.random.choice(self.mdl.index.values, size=M, replace=True, p=self.mdl.values)
        return " ".join(map(str, draws))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        N = self.N
        L = len(tokens)
        return [tuple(tokens[i:i+N]) for i in range(L - N + 1)]

        
    def train(self, ngrams):
        ng_counts = {}
        n1_counts = {}
        for ng in ngrams:
            ng_counts[ng] = ng_counts.get(ng, 0) + 1
            n1 = ng[:-1]
            n1_counts[n1] = n1_counts.get(n1, 0) + 1
        rows = []
        for ng, c in ng_counts.items():
            n1 = ng[:-1]
            p = c / float(n1_counts[n1])
            rows.append((ng, n1, p))
        mdl = pd.DataFrame(rows, columns=['ngram', 'n1gram', 'prob'])

        self._prob_map = dict(zip(mdl['ngram'], mdl['prob']))
        cond = {}
        for ng, n1, p in rows:
            nxt = ng[-1]
            if n1 not in cond:
                cond[n1] = ([], [])
            cond[n1][0].append(nxt)
            cond[n1][1].append(p)
        self._cond_map = cond
        return mdl

    
    def probability(self, words):
        k = len(words)
        if k == 0:
            return 1.0

        m = min(k, self.N - 1)
        prob = 1.0
        if m > 0:
            prob *= self.prev_mdl.probability(tuple(words[:m]))

        for i in range(self.N - 1, k):
            ng = tuple(words[i - self.N + 1: i + 1])
            p = self._prob_map.get(ng)
            if p is None:
                return 0.0
            prob *= p
        return prob

    

    def sample(self, M):
        START, STOP = '\x02', '\x03'
        out = [START]  
        for _ in range(M - 1):
            model = self
            while hasattr(model, 'N') and len(out) < model.N:
                model = model.prev_mdl

            if hasattr(model, 'N') and model.N >= 2:
                ctx_len = model.N - 1
                ctx = tuple(out[-ctx_len:]) if ctx_len > 0 else tuple()
                choices = model._cond_map.get(ctx)
                if not choices:
                    nxt = STOP
                else:
                    tokens, probs = choices
                    nxt = np.random.choice(tokens, p=np.array(probs, dtype=float))
            else:
                tokens = model.mdl.index.values
                probs = model.mdl.values.astype(float)
                nxt = np.random.choice(tokens, p=probs)
            out.append(nxt)
        out.append(STOP)
        return ' '.join(out)

