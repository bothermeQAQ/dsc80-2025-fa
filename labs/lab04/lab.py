# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login):
    t = pd.to_datetime(login['Time'], errors='coerce')
    mask = (t.dt.hour >= 16) & (t.dt.hour < 20)
    counts = mask.groupby(login['Login Id']).sum().astype(int)
    return counts.to_frame('Time')


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    end = pd.Timestamp('2024-01-31 23:59:00')
    t = pd.to_datetime(login['Time'], errors='coerce')
    g = login.assign(Time=t).groupby('Login Id')
    total = g.size()
    days = (end - g['Time'].min()).dt.days.clip(lower=1)
    return (total / days)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return [1, 2]

def cookies_p_value(N):
    n = 250
    observed = 15
    p_burnt = 0.04
    sims = np.random.binomial(n, p_burnt, size=N)
    return (sims >= observed).mean()


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return [1, 4]

def car_alt_hypothesis():
    return [2, 6]

def car_test_statistic():
    return [1, 4]

def car_p_value():
    return 4



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    return [1, 2]

def bhbe_col(heroes):
    hair = heroes['Hair color'].str.contains('blond', case=False, na=False)
    eyes = heroes['Eye color'].str.contains('blue', case=False, na=False)
    return hair & eyes

def superheroes_observed_statistic(heroes):
    mask = bhbe_col(heroes)
    a = heroes.loc[mask, 'Alignment'].str.lower()
    a = a[a.notna()]
    return a.eq('good').mean()

def simulate_bhbe_null(heroes, N):
    mask = bhbe_col(heroes)
    k = int(mask.sum())
    a = heroes['Alignment'].str.lower()
    a = a[a.notna()]
    g = int(a.eq('good').sum())
    t = int(a.size)
    return np.random.hypergeometric(g, t - g, k, size=N) / k

def superheroes_p_value(heroes):
    obs = superheroes_observed_statistic(heroes)
    sims = simulate_bhbe_null(heroes, 100_000)
    p = (sims >= obs).mean()
    return [p, 'Reject' if p < 0.01 else 'Fail to reject']



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    colors = ['red', 'orange', 'yellow', 'green', 'purple']
    prop = data[col] / data[colors].sum(axis=1)
    g = prop.groupby(data['Factory']).mean()
    return abs(g['Yorkville'] - g['Waco'])

def simulate_null(data, col='orange'):
    colors = ['red', 'orange', 'yellow', 'green', 'purple']
    prop = data[col] / data[colors].sum(axis=1)
    shuffled = np.random.permutation(data['Factory'].values)
    g = prop.groupby(shuffled).mean()
    return abs(g['Yorkville'] - g['Waco'])

def color_p_value(data, col='orange'):
    obs = diff_of_means(data, col)
    sims = np.array([simulate_null(data, col) for _ in range(10_000)])
    return (sims >= obs).mean()



# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    fp = Path('data') / 'skittles.tsv'
    df = pd.read_csv(fp, sep='\t')
    colors = ['red', 'orange', 'yellow', 'green', 'purple']
    totals = df[colors].sum(axis=1)
    labels = df['Factory'].to_numpy()
    n_y = (labels == 'Yorkville').sum()
    rng = np.random.default_rng(0)
    def pval(col):
        prop = (df[col] / totals).to_numpy()
        obs = abs(prop[labels == 'Yorkville'].mean() - prop[labels == 'Waco'].mean())
        sims = np.empty(10_000)
        n = len(labels)
        for i in range(10_000):
            idx = rng.permutation(n)
            mask = np.zeros(n, dtype=bool)
            mask[idx[:n_y]] = True
            sims[i] = abs(prop[mask].mean() - prop[~mask].mean())
        return round((sims >= obs).mean(), 3)
    out = [(c, pval(c)) for c in colors]
    out.sort(key=lambda x: x[1])
    return out



# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


    
def same_color_distribution():
    return (0.0, 'Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P', 'P', 'H', 'H', 'P']
