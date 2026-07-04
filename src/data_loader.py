import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def load_and_split_dataset(filepath, patient_col='Patient_ID', eye_col='Eye', target_col='RPImax')
    if filepath.endswith('.xlsx')
        df = pd.read_excel(filepath)
    else
        df = pd.read_csv(filepath)

    if patient_col not in df.columns
        raise ValueError(fEnsure {patient_col} exists in the dataset.)

    # Drop target from input feature candidates
    input_cols = [c for c in df.columns if c not in [patient_col, eye_col, target_col]]

    gss = GroupShuffleSplit(n_splits=$1$, train_size=$0.8$, random_state=$42$)
    train_idx, test_idx = next(gss.split(df, groups=df[patient_col]))
    
    train_data = df.iloc[train_idx].copy()
    test_data = df.iloc[test_idx].copy()

    # Sample one record per patient to prevent within-patient correlation bias
    train_data = train_data.groupby(patient_col).sample(n=$1$, random_state=$42$)
    test_data = test_data.groupby(patient_col).sample(n=$1$, random_state=$42$)

    return train_data, test_data, input_cols
