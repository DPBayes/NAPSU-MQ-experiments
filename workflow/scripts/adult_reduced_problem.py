import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

main_conf_levels = [0.8, 0.9, 0.95]
all_conf_levels = list(np.linspace(0.05, 0.95, 19)) # = [0.05, 0.1, 0.15, ..., 0.9, 0.95]
cols_in_lr = ["income", "age", "race", "gender"]
max_ent_n_syn_datasets = [10, 25, 50, 100]

def df_transform(df, orig_columns):
    c_df = df.copy()[cols_in_lr]
    c_df["income"] = (c_df["income"] == "True").astype(int)
    c_df["age_continuous"] = c_df.age.apply(lambda age: interval_middle(age)).astype(float)
    c_df.drop(columns=["age"], inplace=True)
    syn_oh_df = pd.get_dummies(c_df)
    if "race_White" in syn_oh_df.columns:
        syn_oh_df.drop(columns=["race_White"], inplace=True)
    if "gender_Female" in syn_oh_df.columns:
        syn_oh_df.drop(columns=["gender_Female"], inplace=True)
    syn_oh_df = syn_oh_df.reindex(columns=orig_columns, fill_value=0)
    return syn_oh_df

def interval_middle(interval):
    e = re.compile(r"\((\d*\.\d*), (\d*.\d*)]")
    m = e.match(interval)
    return (float(m[1]) + float(m[2])) / 2

def logistic_regression_accuracy(syn_oh_df, orig_oh_df):
    try:
        model = sm.GLM(syn_oh_df["income"], sm.add_constant(syn_oh_df.drop(columns=["income"]), has_constant="add"), family=sm.families.Binomial())
        result = model.fit()
        probs = result.predict(sm.add_constant(orig_oh_df.drop(columns=["income"]), has_constant="add"))
        preds = probs > 0.5
        correct = (preds == orig_oh_df["income"])
        return correct.sum() / orig_oh_df.shape[0]
    except PerfectSeparationError:
        return np.nan
