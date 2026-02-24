import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent

def import_raw(filenames: list[str]) -> pd.DataFrame:
    tar_name = "weather_raw.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        return pd.read_csv(BASE_DIR / "data" / tar_name)
    df = import_csv_names(filenames)
    df.to_csv((BASE_DIR / "data" / tar_name), index=False)
    return df

def import_csv() -> pd.DataFrame:
    # check existing file
    tar_name = "weather_preprocessed.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        return pd.read_csv(BASE_DIR / "data" / tar_name)

    # 2017~2019 file names
    f1 = "seoul 2016-01-01 to 2018-01-01.csv"
    f2 = "seoul 2018-01-01 to 2020-01-01.csv"

    df = import_csv_names([f1, f2])

    # filter needed years only
    buff = (2017 <= df['year']) & (df['year'] <= 2019)
    df = df[buff]
    df = df.reset_index(drop=True)

    # save file
    df.to_csv((BASE_DIR / "data" / tar_name), index=False)
    return df

def import_csv_names(filenames: list[str]) -> pd.DataFrame:
    # concat csvs
    df_list = []
    for l in filenames:
        if l[-4:] != '.csv':
            l = l + '.csv'
        buff = pd.read_csv(BASE_DIR / 'data' / l)
        df_list.append(buff)

    if len(df_list) == 0:
        raise Exception('No weather data to import.')

    df = pd.concat(df_list, axis=0, ignore_index=True)

    # preprocess and return
    df = preprocess(df)

    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # preprocess day
    df.loc[:, "year"] = df["datetime"].str[:4].astype(int)
    df.loc[:, "month"] = df["datetime"].str[5:7].astype(int)
    df.loc[:, "day"] = df["datetime"].str[8:].astype(int)

    # read cloudy information
    df.loc[:, "cloudy"] = df['conditions'].apply(read_condition)

    # fill float columns with 0
    float_cols = ['temp', 'humidity',
            'precip', 'snow', 'windspeed', 'winddir',
            'sealevelpressure', 'visibility', 'solarradiation']
    df[float_cols] = df[float_cols].fillna(0).astype(float)

    # drop unnecessary columns
    sorted_cols = ['year', 'month', 'day', 'temp', 'humidity', 'cloudy',
            'precip', 'snow', 'windspeed', 'winddir',
            'sealevelpressure', 'visibility', 'solarradiation']
    df = df[sorted_cols]

    df = df.drop_duplicates(
        subset=['year', 'month', 'day'],
        keep='first'
    )

    return df

# cloudy conditions
def read_condition(cond:str) -> int:
    if "Overcast" in cond:
        return 3
    elif "Partially cloudy" in cond:
        return 1
    elif "cloudy" in cond:
        return 2
    return 0

def reset_file():
    tar_name = "weather_raw.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        os.remove(BASE_DIR / "data" / tar_name)
    tar_name = "weather_preprocessed.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        os.remove(BASE_DIR / "data" / tar_name)
