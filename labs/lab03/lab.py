# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):
    dir_path = Path(dirname)
    if not dir_path.exists():
        raise FileNotFoundError(f"{dirname} not found")

    dfs = []
    for file in dir_path.iterdir():
        if file.name.startswith("survey") and file.suffix == ".csv":
            df = pd.read_csv(file)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No survey CSVs found in directory.")

    combined = pd.concat(dfs, ignore_index=True)
    cols = ['first name', 'last name', 'current company', 'job title', 'email', 'university']
    combined = combined[cols]

    return combined.reset_index(drop=True)


def com_stats(df):
    ohio = df['university'].str.contains('Ohio', case=False, na=False)
    programmer = df['job title'].str.contains('Programmer', case=False, na=False)
    prop = (ohio & programmer).sum() / ohio.sum() if ohio.sum() > 0 else 0

    ends_with_engineer = int(df['job title'].str.endswith('Engineer', na=False).sum())
    longest_title = df['job title'].iloc[df['job title'].str.len().idxmax()]
    manager_count = df['job title'].str.contains('manager', case=False, na=False).sum()

    return [prop, ends_with_engineer, longest_title, manager_count]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    dir_path = Path(dirname)
    if not dir_path.exists():
        raise FileNotFoundError(f"{dirname} not found")

    dfs = []
    for file in dir_path.iterdir():
        if file.name.startswith("favorite") and file.suffix == ".csv":
            df = pd.read_csv(file)
            dfs.append(df)

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.merge(df, on="id", how="left")

    combined = combined.set_index("id")
    return combined


def check_credit(df):
    valid = df.copy()
    if "genres" in valid.columns:
        valid["genres"] = valid["genres"].replace("(no genres listed)", np.nan)

    answered = valid.notna().sum(axis=1)
    total = valid.shape[1] - 1
    base_ec = np.where(answered >= total / 2, 5, 0)

    class_bonus = 0
    for col in valid.columns:
        if col == "name":
            continue
        prop_answered = valid[col].notna().mean()
        if prop_answered >= 0.9:
            class_bonus += 1
            if class_bonus == 2:
                break

    total_ec = base_ec + class_bonus
    out = pd.DataFrame({"name": valid["name"], "ec": total_ec})
    out.index = valid.index
    return out



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    merged = pets.merge(procedure_history, on="PetID", how="inner")
    return merged["ProcedureType"].mode()[0]


def pet_name_by_owner(owners, pets):
    pets = pets.rename(columns={"Name": "PetName"})
    merged = owners.merge(pets, on="OwnerID", how="left")

    grouped = merged.groupby(["OwnerID", "Name"])["PetName"].agg(lambda s: [x for x in s.dropna()])
    grouped = grouped.apply(lambda lst: lst[0] if len(lst) == 1 else lst)

    grouped.index = grouped.index.get_level_values("Name")
    return grouped




def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    merged = (
        owners.merge(pets, on="OwnerID", how="inner")
        .merge(procedure_history, on="PetID", how="inner")
        .merge(procedure_detail, on="ProcedureType", how="inner")
    )
    result = merged.groupby("City")["Price"].sum()
    return result



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

def average_seller(sales):
    return sales.groupby("Name")["Total"].mean().to_frame("Average Sales")


def product_name(sales):
    return sales.pivot_table(index="Name", columns="Product", values="Total", aggfunc="sum")


def count_product(sales):
    out = sales.pivot_table(index=["Product", "Name"], columns="Date", values="Total", aggfunc="count")
    return out.fillna(0).astype(int)


def total_by_month(sales):
    s = sales.copy()
    s["Month"] = pd.to_datetime(s["Date"], format="%m.%d.%Y").dt.month_name()
    out = s.pivot_table(index=["Name", "Product"], columns="Month", values="Total", aggfunc="sum")
    return out.fillna(0).astype(int)

