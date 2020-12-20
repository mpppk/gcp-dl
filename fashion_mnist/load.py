import pandas as pd


def load(csv_path):
    df = pd.read_csv(csv_path)
    train_df = select_subset(df, "train")
    test_df = select_subset(df, "test")
    return train_df, test_df

def select_first_label(tags):
    for label in tags.split(","):
        if label != "train" and label != "test":
            return label


def select_subset(df, subset):
    subset_df = df[df["tags"].str.contains(subset, na=False)].assign(subset=subset)
    return subset_df.assign(label=subset_df["tags"].apply(select_first_label)).dropna()

