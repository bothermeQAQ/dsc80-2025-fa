# lab.py


from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():
    return ['NMAR', 'MD', 'MAR', 'MAR', 'MAR']



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():
    return ['MAR', 'NMAR', 'MAR', 'NMAR', 'MCAR']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def first_round():
    payments_fp = Path('data') / 'payment.csv'
    payments = pd.read_csv(payments_fp)
    payments['date_of_birth'] = pd.to_datetime(
        payments['date_of_birth'],
        format='%d-%b-%Y',
        errors='coerce'
    )
    payments['age'] = 2024 - payments['date_of_birth'].dt.year
    df = payments.dropna(subset=['age'])
    ages = df['age'].to_numpy()
    missing = df['credit_card_number'].isna().to_numpy()

    obs = abs(ages[missing].mean() - ages[~missing].mean())

    np.random.seed(0)
    reps = 1000
    more_extreme = 0
    for _ in range(reps):
        shuffled = np.random.permutation(missing)
        stat = abs(ages[shuffled].mean() - ages[~shuffled].mean())
        if stat >= obs:
            more_extreme += 1

    p_val = more_extreme / reps
    decision = 'R' if p_val < 0.05 else 'NR'
    return [p_val, decision]


def second_round():
    payments_fp = Path('data') / 'payment.csv'
    payments = pd.read_csv(payments_fp)
    payments['date_of_birth'] = pd.to_datetime(
        payments['date_of_birth'],
        format='%d-%b-%Y',
        errors='coerce'
    )
    payments['age'] = 2024 - payments['date_of_birth'].dt.year
    df = payments.dropna(subset=['age'])
    ages = df['age'].to_numpy()
    missing = df['credit_card_number'].isna().to_numpy()

    obs = stats.ks_2samp(ages[missing], ages[~missing]).statistic

    np.random.seed(0)
    reps = 1000
    more_extreme = 0
    for _ in range(reps):
        shuffled = np.random.permutation(missing)
        stat = stats.ks_2samp(ages[shuffled], ages[~shuffled]).statistic
        if stat >= obs:
            more_extreme += 1

    p_val = more_extreme / reps
    decision = 'R' if p_val < 0.05 else 'NR'
    conclusion = 'D' if decision == 'R' else 'ND'
    return [p_val, decision, conclusion]



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def verify_child(heights):
    child_cols = [c for c in heights.columns if c.startswith('child_')]
    pvals = {}
    for col in child_cols:
        miss = heights[col].isna()
        g1 = heights.loc[miss, 'father']
        g2 = heights.loc[~miss, 'father']
        _, pval = stats.ks_2samp(g1, g2)
        pvals[col] = pval
    return pd.Series(pvals)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):
    bins = pd.qcut(new_heights['father'], 4)
    grp_means = new_heights.groupby(bins)['child'].transform('mean')
    imputed = new_heights['child'].fillna(grp_means)
    return imputed


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    vals = child.dropna().to_numpy()
    hist, edges = np.histogram(vals, bins=10)
    probs = hist / hist.sum()
    np.random.seed(0)
    idx = np.random.choice(len(hist), size=N, p=probs)
    return np.random.uniform(edges[idx], edges[idx + 1])


def impute_height_quant(child):
    s = child.copy()
    missing = s.isna()
    k = missing.sum()
    np.random.seed(0)
    fills = quantitative_distribution(s, k)
    s[missing] = fills
    return s



# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def answers():
    mc_answers = [1, 2, 2, 1]
    websites = ['example.com', 'instagram.com']
    return mc_answers, websites
