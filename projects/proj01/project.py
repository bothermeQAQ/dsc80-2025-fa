# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    categories = {
        'lab': [],
        'project': [],
        'midterm': [],
        'final': [],
        'disc': [],
        'checkpoint': []
    }

    for col in grades.columns:
        for key in categories:
            if key == 'disc':
                prefix = 'discussion'
            else:
                prefix = key
            if col.lower().startswith(prefix):
                name = col.split(' ')[0].split('-')[0]
                if name not in categories[key]:
                    categories[key].append(name)

    return categories


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    project_names = get_assignment_names(grades)['project']
    project_scores = []

    for p in project_names:
        score_cols = [c for c in grades.columns if c.startswith(p) and 'Max Points' not in c and 'Lateness' not in c]
        max_cols = [c for c in grades.columns if c.startswith(p) and 'Max Points' in c]

        total_earned = grades[score_cols].fillna(0).sum(axis=1)
        total_possible = grades[max_cols].fillna(0).sum(axis=1)

        project_scores.append(total_earned / total_possible)

    return pd.concat(project_scores, axis=1).mean(axis=1)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    parts = col.str.split(":", expand=True).astype(float)
    hours = parts[0] + parts[1] / 60 + parts[2] / 3600
    hours -= 2
    hours = hours.clip(lower=0)
    penalty = pd.Series(1.0, index=col.index)
    penalty.loc[(hours > 0) & (hours <= 168)] = 0.9
    penalty.loc[(hours > 168) & (hours <= 336)] = 0.7
    penalty.loc[hours > 336] = 0.4

    return penalty


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    lab_names = get_assignment_names(grades)['lab']
    labs = pd.DataFrame(index=grades.index)

    for lab in lab_names:
        score_col = lab
        max_col = f"{lab} - Max Points"
        late_col = f"{lab} - Lateness (H:M:S)"

        score = grades[score_col].fillna(0)
        max_points = grades[max_col].fillna(0)

        normalized = (score / max_points).replace([np.inf, -np.inf], 0).fillna(0)

        penalty = lateness_penalty(grades[late_col].fillna("00:00:00"))

        labs[lab] = normalized * penalty

    return labs



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    n_labs = processed.shape[1]

    if n_labs <= 1:
        return processed.mean(axis=1).fillna(0)

    total = (processed.sum(axis=1) - processed.min(axis=1)) / (n_labs - 1)

    total = total.clip(lower=0, upper=1).fillna(0)

    return total


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):
    labs = process_labs(grades)
    lab = lab_total(labs)
    project = projects_total(grades)

    names = get_assignment_names(grades)
    checkpoint = names['checkpoint']
    disc = names['disc']
    midterm = names['midterm']
    final = names['final']

    def safe_ratio(n):
        num = grades[n].fillna(0)
        den = grades[[f"{c} - Max Points" for c in n]].fillna(0)
        return (num.sum(axis=1) / den.sum(axis=1)).replace([np.inf, -np.inf], 0).fillna(0)

    checkpoints = safe_ratio(checkpoint)
    discussions = safe_ratio(disc)
    midterm_score = safe_ratio(midterm)
    final_score = safe_ratio(final)

    total = (
        0.20 * lab +
        0.30 * project +
        0.025 * checkpoints +
        0.025 * discussions +
        0.15 * midterm_score +
        0.30 * final_score
    )

    return total.clip(0, 1)


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    grade = pd.Series(index=total.index, dtype=str)
    grade[total >= 0.9] = 'A'
    grade[(total >= 0.8) & (total < 0.9)] = 'B'
    grade[(total >= 0.7) & (total < 0.8)] = 'C'
    grade[(total >= 0.6) & (total < 0.7)] = 'D'
    grade[total < 0.6] = 'F'
    return grade

def letter_proportions(total):
    letters = final_grades(total)
    proportions = letters.value_counts(normalize=True)
    proportions = proportions.reindex(['B', 'C', 'A', 'D', 'F'], fill_value=0)
    return proportions


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    redemption_cols = [final_breakdown.columns[i] for i in question_numbers]
    earned = final_breakdown[redemption_cols].sum(axis=1, skipna=True)
    max_points = sum([
        float(col.split('(')[1].split()[0])
        for col in redemption_cols
    ])

    scores = earned / max_points
    scores = scores.fillna(0)
    return pd.DataFrame({
        'PID': final_breakdown['PID'],
        'Raw Redemption Score': scores
    })
    
def combine_grades(grades, raw_redemption_scores):
    return grades.merge(raw_redemption_scores, on='PID', how='left')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    mean = ser.mean()
    std = ser.std(ddof=0)
    return (ser - mean) / std
    
def add_post_redemption(grades_combined):
    df = grades_combined.copy()

    df['Midterm Score Pre-Redemption'] = df['Midterm'] / df['Midterm - Max Points']

    z_pre = z_score(df['Midterm Score Pre-Redemption'])
    z_red = z_score(df['Raw Redemption Score'])

    mean_pre = df['Midterm Score Pre-Redemption'].mean()
    std_pre = df['Midterm Score Pre-Redemption'].std(ddof=0)

    df['Midterm Score Post-Redemption'] = np.where(
        z_red > z_pre,
        z_red * std_pre + mean_pre,
        df['Midterm Score Pre-Redemption']
    )

    df['Midterm Score Post-Redemption'] = df['Midterm Score Post-Redemption'].clip(upper=1)
    
    return df

# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    if 'Midterm Score Pre-Redemption' not in grades_combined.columns:
        grades_combined = add_post_redemption(grades_combined)

    total_before = total_points(grades_combined)
    midterm_before = grades_combined['Midterm Score Pre-Redemption']
    midterm_after = grades_combined['Midterm Score Post-Redemption']
    
    total_after = total_before - 0.15 * midterm_before + 0.15 * midterm_after
    return total_after
        
def proportion_improved(grades_combined):
    if 'Midterm Score Pre-Redemption' not in grades_combined.columns:
        grades_combined = add_post_redemption(grades_combined)

    before = total_points(grades_combined)
    after = total_points_post_redemption(grades_combined)
    
    before_letters = final_grades(before)
    after_letters = final_grades(after)
    
    order = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    improved = after_letters.map(order) > before_letters.map(order)
    
    return improved.mean()

# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    before = total_points(grades_analysis)
    after = total_points_post_redemption(grades_analysis)
    before_letters = final_grades(before)
    after_letters = final_grades(after)
    improved = after_letters != before_letters
    return (grades_analysis.loc[improved]
            .groupby('Section')
            .size()
            .div(grades_analysis.groupby('Section').size())
            .idxmax())
    
def top_sections(grades_analysis, t, n):
    sections = (grades_analysis[grades_analysis['Final'] / grades_analysis['Final - Max Points'] >= t]
                .groupby('Section').size())
    return np.array(sorted(sections[sections >= n].index))


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    scores = total_points_post_redemption(grades_analysis)
    df = grades_analysis.assign(Total=scores)
    ranks = (df.sort_values(['Section', 'Total'], ascending=[True, False])
               .groupby('Section')['PID']
               .apply(list))
    n = ranks.apply(len).max()
    sections = [f'A{i:02d}' for i in range(1, 31)]
    data = {s: (ranks[s] + [''] * (n - len(ranks[s]))) if s in ranks else [''] * n for s in sections}
    return pd.DataFrame(data, index=range(1, n + 1)).rename_axis('Section Rank')







# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    import plotly.express as px
    post = final_grades(total_points_post_redemption(grades_analysis))
    df = grades_analysis.assign(Post=post)
    props = (df.groupby(['Section', 'Post']).size() / df.groupby('Section').size()).unstack(fill_value=0)
    props = props.reindex(['A', 'B', 'C', 'D', 'F']).T.reindex([f'A{i:02d}' for i in range(1, 31)], fill_value=0).T
    fig = px.imshow(props, color_continuous_scale='YlGnBu',
                    labels={'x': 'Section', 'y': 'Letter Grade Post-Redemption'},
                    title='Distribution of Letter Grades by Section')
    return fig
