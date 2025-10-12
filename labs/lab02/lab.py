# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def trick_me():
    tricky_1 = pd.DataFrame(
        [
            ['Faker',  'T1',   29],
            ['Chovy',  'Gen.G',24],
            ['Rookie', 'iG',   28],
            ['Smlz',   'OMG',  28],
            ['Uzi',    'RNG',  28],
        ],
        columns=['Name', 'Name', 'Age']
    )

    tricky_1.to_csv('tricky_1.csv', index=False)
    tricky_2 = pd.read_csv('tricky_1.csv')
    return 2 if tricky_1.columns.equals(tricky_2.columns) else 3


def trick_bool():
    return [4, 10, 13]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(df):
    num_nonnull = df.notna().sum()
    prop_nonnull = num_nonnull / len(df)
    num_distinct = df.nunique(dropna=True)
    prop_distinct = num_distinct / num_nonnull
    return pd.DataFrame({
        'num_nonnull': num_nonnull,
        'prop_nonnull': prop_nonnull,
        'num_distinct': num_distinct,
        'prop_distinct': prop_distinct
    })


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(df, N=10):
    result = pd.DataFrame(index=range(N))
    for col in df.columns:
        counts = df[col].value_counts()
        values = counts.index[:N].to_list()
        freqs = counts.values[:N].astype(float)
        if len(values) < N:
            values += [np.nan] * (N - len(values))
            freqs = np.append(freqs, [np.nan] * (N - len(freqs)))
        result[f'{col}_values'] = values
        result[f'{col}_counts'] = freqs
    return result


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    hero_names = powers['hero_names']
    power_data = powers.drop(columns='hero_names')

    num_powers = power_data.sum(axis=1)
    most_powerful = hero_names.iloc[num_powers.idxmax()]

    flyers = power_data[power_data['Flight']]
    flight_common = flyers.drop(columns='Flight').sum().idxmax()

    single_power_heroes = power_data[num_powers == 1]
    single_common = single_power_heroes.sum().idxmax()

    return [most_powerful, flight_common, single_common]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    return heroes.replace(['-', '--', 'Unknown', 'unknown', 'None', 'none', '', -99.0], np.nan)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    return [
        'Apocalypse',           
        'Dark Horse Comics',    
        'good',                 
        'Marvel Comics',        
        'NBC - Heroes',         
        'Giganta'              
    ]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_universities(df):
    df = df.copy()
    df['institution'] = df['institution'].str.replace('\n', ', ', regex=False)
    df['broad_impact'] = df['broad_impact'].astype(int)
    split = df['national_rank'].str.split(', ', expand=True)
    df['nation'] = split[0]
    df['national_rank_cleaned'] = split[1].astype(int)
    df = df.drop(columns='national_rank')
    df['nation'] = df['nation'].replace({
        'Czechia': 'Czech Republic',
        'Russia': 'Russian Federation',
        'UK': 'United Kingdom',
        'USA': 'United States'
    })
    df['is_r1_public'] = (
        df['control'].notna() &
        df['city'].notna() &
        df['state'].notna() &
        (df['control'].str.lower() == 'public')
    )
    return df



def university_info(cleaned):
    state_counts = cleaned['state'].value_counts()
    valid_states = state_counts[state_counts >= 3].index
    lowest_state = (
        cleaned[cleaned['state'].isin(valid_states)]
        .groupby('state')['score']
        .mean()
        .idxmin()
    )
    top100 = cleaned[cleaned['world_rank'] <= 100]
    prop_quality_faculty = (top100['quality_of_faculty'] <= 100).mean()
    state_private_ratio = (
        cleaned.groupby('state')['is_r1_public']
        .apply(lambda x: (x == False).mean())
    )
    num_states_majority_private = (state_private_ratio >= 0.5).sum()
    best_in_each = (
        cleaned[cleaned['national_rank_cleaned'] == 1]
        .sort_values('world_rank', ascending=False)
    )
    worst_best_institution = best_in_each.iloc[0]['institution']
    return [
        lowest_state,
        float(prop_quality_faculty),
        int(num_states_majority_private),
        worst_best_institution
    ]