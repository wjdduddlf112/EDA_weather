import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decorator import append

BASE_DIR = Path(__file__).resolve().parent

def import_raw(start_year:int, end_year:int) -> pd.DataFrame:
    # check existing file
    tar_name = "pollution_raw.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        return pd.read_csv(BASE_DIR / "data" / tar_name)

    # preprocess without year data
    df = preprocess(start_year, end_year)

    # export result
    df.to_csv((BASE_DIR / "data" / tar_name), index=False)
    return df

def import_csv() -> pd.DataFrame:
    tar_name = "pollution_preprocessed.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        return pd.read_csv(BASE_DIR / "data" / tar_name)
    df = preprocess(2017, 2019)
    df.to_csv((BASE_DIR / "data" / tar_name), index=False)
    return df

def preprocess(s=-1, e=-1) -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "data" / "seoul_air_1988_2021.csv")

    # add year column & cut data to optimize
    df.loc[:, "year"] = df["dt"] // 1_000_000
    if s == -1:
        s = df["year"].min
    if e == -1:
        e = df["year"].max
    df = df[(df["year"] >= s) & (df["year"] <= e)]

    # add month, day
    df.loc[:, "month"] = df["dt"] % 1_000_000 // 10_000
    df.loc[:, "day"] = df["dt"] % 10_000 // 100

    # drop unnecessary columns
    group_cols = ["year", "month", "day"]
    target_cols = ["so2", "no2", "co", "o3", "pm10", "pm2.5"]
    df = df.drop([k for k in df.columns if k not in group_cols+target_cols], axis=1)

    # fill na with mean data
    df[target_cols] = (
        df.groupby(group_cols)[target_cols]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # mean pollutant amount for each day
    df = df.groupby(group_cols, as_index=False).mean().round(2)

    # define pollutant level data
    pollutant_label = target_cols
    level_label = [k+"_level" for k in pollutant_label]
    bins = [
        [-np.inf, 0.02, 0.05, 0.15, np.inf],
        [-np.inf, 0.03, 0.06, 0.2, np.inf],
        [-np.inf, 2, 9, 15, np.inf],
        [-np.inf, 0.03, 0.09, 0.15, np.inf],
        [-np.inf, 30, 80, 150, np.inf],
        [-np.inf, 15, 35, 75, np.inf]
    ]

    # add each pollutant level column
    for i in range(len(pollutant_label)):
        df.loc[:, level_label[i]] = pd.cut(df[pollutant_label[i]], bins=bins[i], labels=range(4))

    # sort columns
    front = group_cols
    others = target_cols
    arranged = front + others
    target_list = []
    for i in range(len(arranged)):
        target_list.append(arranged[i])
        if arranged[i] in pollutant_label:
            target_list.append(arranged[i] + "_level")

    df = df[target_list]
    return df

def reset_file():
    tar_name = "pollution_raw.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        os.remove(BASE_DIR / "data" / tar_name)
    tar_name = "pollution_preprocessed.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        os.remove(BASE_DIR / "data" / tar_name)