# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):
    out = loans.copy()
    out['issue_d'] = pd.to_datetime(out['issue_d'], format='%b-%Y', errors='coerce')
    out['term'] = out['term'].astype(str).str.extract(r'(\d+)')[0].astype(int)
    emp = out['emp_title'].astype('string').str.lower().str.strip().replace({'rn': 'registered nurse'})
    out['emp_title'] = emp
    offsets = out['term'].map(lambda m: pd.DateOffset(months=int(m)))
    try:
        out['term_end'] = out['issue_d'] + offsets
    except Exception:
        out['term_end'] = out.apply(
            lambda r: r['issue_d'] + pd.DateOffset(months=int(r['term']))
            if pd.notna(r['issue_d']) and pd.notna(r['term']) else pd.NaT,
            axis=1
        )
    return out



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def correlations(df, pairs):
    idx = []
    vals = []
    for col1, col2 in pairs:
        s1, s2 = df[col1], df[col2]
        def to_num(s):
            if pd.api.types.is_numeric_dtype(s):
                return pd.to_numeric(s, errors='coerce')
            s_num = pd.to_numeric(s, errors='coerce')
            if s_num.notna().any():
                return s_num
            ex = s.astype(str).str.extract(r'([+-]?\d+(?:\.\d+)?)')[0]
            return pd.to_numeric(ex, errors='coerce')
        x, y = to_num(s1), to_num(s2)
        m = x.notna() & y.notna()
        r = x[m].corr(y[m])
        idx.append(f"r_{col1}_{col2}")
        vals.append(r if pd.notna(r) else np.nan)
    return pd.Series(vals, index=idx, dtype=float)




# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    bins = [580, 670, 740, 800, 850]
    labels = ['[580, 670)', '[670, 740)', '[740, 800)', '[800, 850)']
    df = loans.copy()
    df = df[df['term'].isin([36, 60])]
    df['fico_bin'] = pd.cut(df['fico_range_low'], bins=bins, right=False, include_lowest=True, labels=labels).astype(str)
    fig = px.box(
        df,
        x='fico_bin',
        y='int_rate',
        color='term',
        category_orders={'fico_bin': labels, 'term': [36, 60]},
        labels={'fico_bin': 'Credit Score Range', 'int_rate': 'Interest Rate (%)', 'term': 'Loan Length (Months)'},
        title='Interest Rate vs. Credit Score',
        color_discrete_map={36: 'purple', 60: 'gold'}
    )
    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    has_ps = loans['desc'].notna()
    rates = loans['int_rate']
    m = rates.notna() & has_ps.notna()
    has_ps = has_ps[m].to_numpy()
    rates = rates[m]
    if has_ps.sum() == 0 or (~has_ps).sum() == 0:
        return np.nan
    obs = rates[has_ps].mean() - rates[~has_ps].mean()
    count = 0
    for _ in range(int(N)):
        perm = np.random.permutation(has_ps)
        diff = rates[perm].mean() - rates[~perm].mean()
        if diff >= obs:
            count += 1
    return (count + 1) / (int(N) + 1)

def missingness_mechanism():
    return 2

def argument_for_nmar():
    return "Applicants may add a statement only when they believe their personal story is compelling, so inclusion depends on the (unobserved) content of the statement itself, making NMAR plausible."



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    owed = 0.0
    inc = float(income)
    bs = sorted(brackets, key=lambda x: x[1])
    n = len(bs)
    for i, (rate, lower) in enumerate(bs):
        if inc <= lower:
            break
        upper = bs[i+1][1] if i+1 < n else None
        amount = (inc if upper is None else min(inc, upper)) - lower
        if amount > 0:
            owed += rate * amount
    return owed

# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw):
    df = state_taxes_raw.copy()
    df = df.dropna(how='all')
    s = df['State'].astype('string').str.strip()
    s = s.mask(s.str.contains(r'\(', na=False) | s.eq(''))
    df['State'] = s.ffill()
    r = df['Rate'].astype('string').str.strip().str.lower().replace({'none': None})
    r = r.str.replace('%', '', regex=False)
    r = pd.to_numeric(r, errors='coerce').fillna(0).div(100).round(2)
    df['Rate'] = r
    L = df['Lower Limit'].astype('string').str.replace(r'[\$,]', '', regex=True)
    L = pd.to_numeric(L, errors='coerce').fillna(0).astype(int)
    df['Lower Limit'] = L
    return df[['State', 'Rate', 'Lower Limit']]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    df = state_taxes.sort_values(['State', 'Lower Limit'])
    ser = df.groupby('State').apply(lambda g: list(zip(g['Rate'].astype(float), g['Lower Limit'].astype(int))))
    return ser.to_frame('bracket_list')


def combine_loans_and_state_taxes(loans, state_taxes):
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
    brackets = state_brackets(state_taxes).reset_index()
    brackets['State'] = brackets['State'].map(state_mapping)
    out = loans.rename(columns={'addr_state': 'State'}).merge(brackets, on='State', how='left')
    return out


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
        (0.1, 0),
        (0.12, 11000),
        (0.22, 44725),
        (0.24, 95375),
        (0.32, 182100),
        (0.35, 231251),
        (0.37, 578125),
    ]
    df = loans_with_state_taxes.copy()
    df['federal_tax_owed'] = df['annual_inc'].apply(lambda x: np.nan if pd.isna(x) else tax_owed(float(x), FEDERAL_BRACKETS))
    df['state_tax_owed'] = df.apply(lambda r: np.nan if pd.isna(r['annual_inc']) else (0.0 if not isinstance(r.get('bracket_list', None), list) or len(r['bracket_list']) == 0 else tax_owed(float(r['annual_inc']), r['bracket_list'])), axis=1)
    df['disposable_income'] = df['annual_inc'] - df['federal_tax_owed'] - df['state_tax_owed']
    return df


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    cols = []
    for kw in keywords:
        mask = loans['emp_title'].fillna('').str.contains(kw, regex=False)
        by_cat = loans.loc[mask].groupby(categorical_column)[quantitative_column].mean().sort_index()
        overall = pd.Series({'Overall': loans.loc[mask, quantitative_column].mean()})
        s = pd.concat([by_cat, overall])
        s.name = f"{kw}_mean_{quantitative_column}"
        cols.append(s)
    return pd.concat(cols, axis=1)

# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    tbl = aggregate_and_combine(loans, keywords, quantitative_column, categorical_column)
    a = f"{keywords[0]}_mean_{quantitative_column}"
    b = f"{keywords[1]}_mean_{quantitative_column}"
    by_cat = tbl.loc[tbl.index != "Overall", [a, b]].dropna()
    if by_cat.empty or "Overall" not in tbl.index:
        return False
    return bool((by_cat[a] > by_cat[b]).all() and (tbl.loc["Overall", a] < tbl.loc["Overall", b]))


def paradox_example(loans):
    titles = loans["emp_title"].dropna().astype(str)
    tokens = titles.str.findall(r"[a-z]+").explode().dropna()
    vocab = tokens[tokens.str.len() >= 3].value_counts().index.tolist()
    quantitatives = ["loan_amnt", "int_rate", "annual_inc", "dti"]
    categoricals = ["home_ownership", "grade", "term", "purpose"]
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            k1, k2 = vocab[i], vocab[j]
            if {"engineer", "nurse"} == {k1, k2}:
                continue
            for q in quantitatives:
                for c in categoricals:
                    if exists_paradox(loans, [k1, k2], q, c):
                        return {"loans": loans, "keywords": [k1, k2], "quantitative_column": q, "categorical_column": c}
                    if exists_paradox(loans, [k2, k1], q, c):
                        return {"loans": loans, "keywords": [k2, k1], "quantitative_column": q, "categorical_column": c}
    return {"loans": loans, "keywords": ["manager", "teacher"], "quantitative_column": "loan_amnt", "categorical_column": "grade"}
