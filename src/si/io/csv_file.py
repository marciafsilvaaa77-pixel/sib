import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ",", features: bool = False, label: bool = False) -> Dataset:
    """
    Reads a csv file and returns a Dataset object.

    Parameters
    ----------
    filename: str
        Path of the file
    sep: str
        Value separator
    features: bool
        Whether the file has a header with the feature names
    label: bool
        Whether the file has a label (assumed to be the last column)

    Returns
    -------
    Dataset
    """
    df = pd.read_csv(filename, sep=sep, header=0 if features else None)

    if label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        feature_names = list(df.columns[:-1]) if features else None
        label_name = df.columns[-1] if features else None
    else:
        X = df.to_numpy()
        y = None
        feature_names = list(df.columns) if features else None
        label_name = None

    return Dataset(X, y, features=feature_names, label=label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ",", features: bool = False, label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file.

    Parameters
    ----------
    filename: str
        Path of the file
    dataset: Dataset
        Dataset to write
    sep: str
        Value separator
    features: bool
        Whether to write a header with the feature names
    label: bool
        Whether to write the label (as the last column)
    """
    df = pd.DataFrame(dataset.X, columns=dataset.features)

    if label and dataset.has_label():
        df[dataset.label] = dataset.y

    df.to_csv(filename, sep=sep, header=features, index=False)
    