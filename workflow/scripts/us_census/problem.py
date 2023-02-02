import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

main_conf_levels = [0.8, 0.9, 0.95]
all_conf_levels = list(np.linspace(0.05, 0.95, 19)) # = [0.05, 0.1, 0.15, ..., 0.9, 0.95]
max_ent_n_syn_datasets = [10, 25, 50, 100]

def df_transform(df, orig_oh_columns):
    cdf = df.copy()
    cdf["dPoverty"] = (cdf.dPoverty.astype(int) < 2).astype(int)
    syn_oh_df = pd.get_dummies(cdf.drop(columns=["dYrsserv", "iEnglish", "iMarital", "iMobillim"]))
    syn_oh_df.drop(columns=[
        "iSex_0", 
        "iVietnam_0",
        "iKorean_0",
        "iMilitary_2",
    ], inplace=True)
    syn_oh_df = syn_oh_df.reindex(columns=orig_oh_columns, fill_value=0)
    return syn_oh_df

def interval_middle(interval):
    e = re.compile(r"\((\d*\.\d*), (\d*.\d*)]")
    m = e.match(interval)
    return (float(m[1]) + float(m[2])) / 2
