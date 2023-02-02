import statsmodels.api as sm
import pandas as pd
import pickle
import re
import os
import adult_reduced_problem as problem

if __name__ == "__main__":

    cols_in_lr = ["income", "age", "race", "gender"]
    c_df = pd.read_csv(str(snakemake.input), dtype="category")[cols_in_lr]
    c_df["age_continuous"] = c_df.age.apply(lambda age: problem.interval_middle(age)).astype(float)
    c_df.drop(columns=["age"], inplace=True)
    c_df["income"] = (c_df["income"] == "True").astype(int)
    oh_df = pd.get_dummies(c_df)
    oh_df.drop(columns=["race_White", "gender_Female"], inplace=True)

    model = sm.GLM(oh_df["income"], sm.add_constant(oh_df.drop(columns=["income"])), family=sm.families.Binomial())
    orig_result = model.fit()

    output_object = {
        "orig_oh_df": oh_df,
        "orig_result": orig_result
    }

    output = str(snakemake.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as file:
        pickle.dump(output_object, file)