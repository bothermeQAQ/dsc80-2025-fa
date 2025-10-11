# lab.py


from pathlib import Path
import io
import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.21')


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    nums_sorted = sorted(nums)
    n = len(nums_sorted)

    mean = sum(nums_sorted) / n

    if n % 2 == 1:
        median = nums_sorted[n // 2]

    else:
        mid = n // 2
        median = (nums_sorted[mid-1] + nums_sorted[mid]) / 2

    return median <= mean
    


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    prefixes = []
    for i in range(1, n+1):
        prefix = s[:i]
        prefixes.append(prefix)

    prefixes = prefixes[::-1]
    result = "".join(prefixes)

    return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    result = []
    max_num = max(ints) + n
    width = len(str(max_num))

    for x in ints:
        expanded = list(range(x-n, x+n+1))
        str_list = [str(num).zfill(width) for num in expanded]
        exploded_str = " ".join(str_list)
        result.append(exploded_str)

    return result
    

    


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def last_chars(fh):
    result = ""
    for line in fh:
        last_char = line.rstrip("\n")[-1]
        result += last_char
    return result


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def add_root(A):
    idx = np.arange(A.size)
    return A + np.sqrt(idx)

def where_square(A):
    roots = np.floor(np.sqrt(A)).astype(int)
    return roots * roots == A


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_loop(matrix, cutoff):
    rows, cols = matrix.shape
    col_means = []
    for j in range(cols):
        s = 0
        for i in range(rows):
            s += matrix[i,j]
        mean_j = s / rows
        col_means.append(mean_j)
    keep = []
    for index,value in enumerate(col_means):
        if value > cutoff:
            keep.append(index)
    
    out_rows = []
    for i in range(rows):
        new_row = []
        for j in keep:
            new_row.append(matrix[i,j])
        out_rows.append(new_row)

    return np.array(out_rows)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_np(matrix, cutoff):
    col_means = np.mean(matrix, axis = 0)
    keep = col_means > cutoff
    return matrix[:, keep]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    diffs = A[1:] - A[:-1]
    rates = diffs / A[:-1]
    return np.round(rates, 2)

def with_leftover(A):
    remainders = 20 % A
    cum = np.cumsum(remainders)
    mask = cum >= A
    return int(np.argmax(mask)) if mask.any() else -1


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
    num_players = len(salary) 
    num_teams   = salary['Team'].nunique() 
    total_salary = salary['Salary'].sum() 

    max_idx = salary['Salary'].idxmax()
    highest_salary_name = salary.loc[max_idx, 'Player']
    
    avg_los = round(
        salary.loc[salary['Team'] == 'Los Angeles Lakers', 'Salary'].mean(),
        2
    )

    fifth_row = salary.sort_values('Salary', ascending=True).iloc[4]
    fifth_lowest = f"{fifth_row['Player']}, {fifth_row['Team']}"

    cleaned_names = salary['Player'].str.replace(
        r'\s+(Jr\.|Sr\.|II|III|IV|V)$', '', regex=True
    )
    last_names = cleaned_names.str.split().str[-1]
    duplicates = last_names.duplicated().any()

    highest_team = salary.loc[max_idx, 'Team']
    total_highest = salary.loc[salary['Team'] == highest_team, 'Salary'].sum()

    out = pd.Series({
        'num_players':  num_players,
        'num_teams':    num_teams,
        'total_salary': total_salary,
        'highest_salary': highest_salary_name,
        'avg_los':      avg_los,
        'fifth_lowest': fifth_lowest,
        'duplicates':   bool(duplicates),
        'total_highest': total_highest,
    })

    return out

# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    rows = []
    with open(fp, "r") as f:
        next(f)
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = s.replace('"', '').rstrip(',')
            parts = s.split(',', 2)
            if len(parts) < 3:
                continue
            first = parts[0].strip()
            last  = parts[1].strip()
            rest  = parts[2].strip()

            toks = [t.strip() for t in rest.split(',')]
            nums = [t for t in toks if t != '' and is_float(t)]
            if len(nums) < 4:
                continue

            weight = float(nums[0])
            height = float(nums[1])
            geo = f"{nums[2]},{nums[3]}"

            rows.append([first, last, weight, height, geo])

    return pd.DataFrame(rows, columns=['first', 'last', 'weight', 'height', 'geo'])
