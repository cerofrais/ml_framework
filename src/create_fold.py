import pandas as pd
from sklearn import model_selection

def make_kfolds(input_path, output_path):
    """
    Function takes in input csv file and writes a csv file with kfold data for cross-validation
    """
    df = pd.read_csv(input_path)
    df["Kfold"] = -1

    df = df.sample(frac=1).reset_index(drop = True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle= False, random_state = 123)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx),len(val_idx))
        df.loc[val_idx, "kfold"] = fold
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = "input/train.csv"
    output_path = "input/train_folds.csv"
    make_kfolds(input_path, output_path)