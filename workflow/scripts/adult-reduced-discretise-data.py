import pandas as pd

if __name__ == "__main__":
    dtypes = {
        "age": int, "workclass": "category", "education": "category", "marital-status": "category",
        "occupation": "category", "relationship": "category", "race": "category", "gender": "category",
        "capital-gain": int, "capital-loss": int, "hours-per-week": int, 
        "native-country": "category", "income": bool
    }
    df = pd.read_csv(str(snakemake.input), na_values=["?"], dtype=dtypes, true_values=[">50K"], false_values=["<=50K"])
    df.drop(columns=[
        "fnlwgt", "educational-num", "native-country", 
        "occupation", "relationship",
    ], inplace=True)
    df.dropna(inplace=True)
    df["age"] = pd.cut(df["age"], 5)
    df["hours-per-week"] = pd.cut(df["hours-per-week"], 5)
    df["capital-gain"] = (df["capital-gain"] > 0).astype("category")
    df["capital-loss"] = (df["capital-loss"] > 0).astype("category")

    df.to_csv(str(snakemake.output), index=False)
