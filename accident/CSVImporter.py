import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent

def import_csv() -> pd.DataFrame:
    tar_name = "accident_preprocessed.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        return pd.read_csv(BASE_DIR / "data" / tar_name)

    data = pd.read_csv(BASE_DIR / "data" / "Seoul_Traffic_Accident_20172019.csv")
    data = data.drop(["Unnamed: 0"], axis=1)

    data.loc[:, "year"] = data["발생일"].str[:4].astype(int)
    data.loc[:, "month"] = data["발생일"].str[5:7].astype(int)
    data.loc[:, "day"] = data["발생일"].str[8:].astype(int)

    front = ["year", "month", "day"]
    others = [c for c in data.columns if c not in front]
    data = data[front + others]

    data = data.drop(["발생일"], axis=1)

    agg_dict = {
        "사고건수": "sum",
        "사망자수": "sum",
        "중상자수": "sum",
        "경상자수": "sum"
    }
    datas_by_month = data.groupby(["year", "month", "day"], as_index=False).agg(agg_dict)

    datas_by_month.loc[:, "death_rate(%)"] = (datas_by_month["사망자수"] / datas_by_month["사고건수"] * 100).round(2)
    datas_by_month.loc[:, "fatal_rate(%)"] = (datas_by_month["중상자수"] / datas_by_month["사고건수"] * 100).round(2)

    rename_dict = {
        "사고건수": "accident_cnt",
        "사망자수": "death",
        "중상자수": "fatal",
        "경상자수": "injured"
    }
    datas_by_month = datas_by_month.rename(columns=rename_dict)

    datas_by_month.to_csv((BASE_DIR / "data" / tar_name), index=False)

    return datas_by_month

def reset_file():
    tar_name = "accident_preprocessed.csv"
    if (BASE_DIR / "data" / tar_name).exists():
        os.remove(BASE_DIR / "data" / tar_name)