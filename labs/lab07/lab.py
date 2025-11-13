# lab.py


import pandas as pd
import numpy as np
import os
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def match_1(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    pattern = r'^..\[..\].*$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_2(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    pattern = r'^\(858\) \d{3}-\d{4}$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_3(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    pattern = r'^[A-Za-z0-9\s?]{5,9}\?$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False
    >>> match_4("$iiuABc")
    False
    >>> match_4("123$$$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    >>> match_4("$!@$")
    False
    """
    pattern = r'^\$[^abc$]*\$[aA]+[bB]+[cC]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """
    pattern = r'^[A-Za-z0-9_]+\.py$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """
    pattern = r'^[a-z]+_[a-z]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_7(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """
    pattern = r'^_.*_$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_8(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo!!!!!!! !!!!!!!!!")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """
    pattern = r'^[^Ol1]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_9(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    >>> match_9('NY-32-LAX-0000') # If the state is NY, the city can be any 3 letter code, including LAX or SAN!
    True
    >>> match_9('TX-32-SAN-4491')
    False
    '''
    pattern = r'^(?:NY-\d{2}-[A-Z]{3}-\d{4}|CA-\d{2}-(?:SAN|LAX)-\d{4})$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_10('ABCdef')
    ['bcd']
    >>> match_10(' DEFaabc !g ')
    ['def', 'bcg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10('Ab..DEF')
    ['bde']
    
    '''
    s = re.sub(r'\W', '', string.lower()).replace('a', '')
    return [s[i:i+3] for i in range(0, len(s), 3) if len(s[i:i+3]) == 3]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def extract_personal(s):
    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    ssn_pattern = r'ssn:(\d{3}-\d{2}-\d{4})'
    bitcoin_pattern = r'bitcoin:([A-Za-z0-9]{26,35})'
    address_pattern = r'\b\d{1,6} (?:\d{1,2}(?:st|nd|rd|th) )?(?:[A-Z][A-Za-z]*)(?: [A-Z][A-Za-z]*)*'
    emails = re.findall(email_pattern, s)
    ssns = re.findall(ssn_pattern, s)
    bitcoins = re.findall(bitcoin_pattern, s)
    addresses = re.findall(address_pattern, s)
    return (emails, ssns, bitcoins, addresses)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def tfidf_data(reviews_ser, review):
    pattern = r'\b\w+\b'
    words = re.findall(pattern, review.lower())
    total = len(words)
    cnt_ser = pd.Series(words).value_counts()

    target_words = set(cnt_ser.index)
    n_docs = len(reviews_ser)
    df_counts = {w: 0 for w in target_words}

    for text in reviews_ser:
        doc_words = set(re.findall(pattern, str(text).lower()))
        for w in target_words.intersection(doc_words):
            df_counts[w] += 1

    idf_ser = pd.Series({w: np.log(n_docs / df_counts[w]) for w in target_words})

    out = pd.DataFrame({"cnt": cnt_ser, "tf": cnt_ser / total})
    out["idf"] = idf_ser
    out["tfidf"] = out["tf"] * out["idf"]
    return out


def relevant_word(out):
    if out.empty:
        return None
    col = "tfidf"
    tfidf_col = out[col]
    if tfidf_col.isna().all():
        return out.index[0]
    max_mask = tfidf_col == tfidf_col.max()
    return tfidf_col.index[max_mask][0]


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def hashtag_list(tweet_text):
    pattern = r'#(\S+)'
    return tweet_text.apply(lambda x: re.findall(pattern, str(x)))


def most_common_hashtag(tweet_lists):
    flat = [h for lst in tweet_lists for h in lst]
    if len(flat) == 0:
        return pd.Series([np.nan] * len(tweet_lists), index=tweet_lists.index)
    counts = pd.Series(flat).value_counts()

    def pick(lst):
        if len(lst) == 0:
            return np.nan
        uniq = []
        for h in lst:
            if h not in uniq:
                uniq.append(h)
        best = uniq[0]
        bestc = counts.get(best, 0)
        for h in uniq[1:]:
            c = counts.get(h, 0)
            if c > bestc:
                best, bestc = h, c
        return best

    return tweet_lists.apply(pick)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------




    


def create_features(ira):
    texts = ira['text']

    hashtags = hashtag_list(texts)
    num_hashtags = hashtags.apply(len)
    mc_hashtags = most_common_hashtag(hashtags)

    tag_pattern = r'@[A-Za-z0-9]+'
    num_tags = texts.apply(lambda x: len(re.findall(tag_pattern, str(x))))

    link_pattern = r'https?://\S+'
    num_links = texts.apply(lambda x: len(re.findall(link_pattern, str(x))))

    is_retweet = texts.str.match(r'\s*RT\b')

    def clean(t):
        t = str(t)
        t = re.sub(r'https?://\S+', ' ', t)
        t = re.sub(r'@[A-Za-z0-9]+', ' ', t)
        t = re.sub(r'#\S+', ' ', t)
        t = re.sub(r'\bRT\b', ' ', t)
        t = re.sub(r'[^A-Za-z0-9 ]+', ' ', t)
        t = t.lower()
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    cleaned_text = texts.apply(clean)

    out = pd.DataFrame({
        'text': cleaned_text,
        'num_hashtags': num_hashtags,
        'mc_hashtags': mc_hashtags,
        'num_tags': num_tags,
        'num_links': num_links,
        'is_retweet': is_retweet
    }, index=ira.index)

    return out[['text', 'num_hashtags', 'mc_hashtags',
                'num_tags', 'num_links', 'is_retweet']]

