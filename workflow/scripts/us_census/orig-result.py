import statsmodels.api as sm
import pandas as pd
import pickle
import os

if __name__ == "__main__":
    sdf = pd.read_csv(str(snakemake.input), dtype="category")
    sdf["dPoverty"] = (sdf.dPoverty.astype(int) < 2)
    oh_df = pd.get_dummies(sdf.drop(columns=["dYrsserv", "iEnglish", "iMarital", "iMobillim"]))
    oh_df.drop(columns=[
        "iSex_0", 
        "iVietnam_0",
        "iKorean_0",
        "iMilitary_2",
    ], inplace=True)

    model = sm.GLM(oh_df["dPoverty"], sm.add_constant(oh_df.drop(columns=["dPoverty"])), family=sm.families.Binomial())
    orig_result = model.fit()

    output_object = {
        "orig_oh_df": oh_df,
        "orig_result": orig_result
    }

    output = str(snakemake.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as file:
        pickle.dump(output_object, file)