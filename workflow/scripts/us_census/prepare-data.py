import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(str(snakemake.input), index_col=0)
    df = df[(df.iMilitary != 0) & (df.iMilitary != 4)].copy()
    columns = [
        "dYrsserv", "iSex", "iVietnam", "iKorean", "iMilitary", "dPoverty", 
        "iMobillim", "iEnglish", "iMarital"
        ]
    df = df[columns]
    df.to_csv(str(snakemake.output), index=False)
