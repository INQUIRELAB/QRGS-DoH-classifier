import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def downsample_data(df):
    total = int(df.shape[0])
    total_1 = int(sum(df["Class"]))
    total_0 = int(total - total_1)

    if total_1 < total_0:
        df_0 = df[df["Class"] == 0].sample(n=total_1) # Return a random sample of items from an axis of object. Default no replacement
        df_comb = pd.concat([df[df["Class"] == 1], df_0])
        df = df_comb.sample(frac=1).reset_index(drop=True)
    elif total_0 < total_1:
        df_1 = df[df["Class"] == 1].sample(n=total_0)
        df_comb = pd.concat([df[df["Class"] == 0], df_1])
        df = df_comb.sample(frac=1).reset_index(drop=True)

    return df


def get_scaler(scaler_type):
    """
    Returns the appropriate scaler object based on user input.
    
    Parameters:
    scaler_type (str): Type of scaler to use. Options:
        - "standard": StandardScaler()
        - "min0max1": MinMaxScaler(0,1)
        - "min0maxpi": MinMaxScaler(0,π)
        - "min0max2pi": MinMaxScaler(0,2π)
        - "min-1max1": MinMaxScaler(-1,1)
        - "min-pimaxpi": MinMaxScaler(-π,π)
    
    Returns:
    Scaler object from sklearn.preprocessing
    
    Raises:
    ValueError: If an invalid scaler_type is provided.
    """
    scalers = {
        "standard": StandardScaler(),
        "min0max1": MinMaxScaler((0, 1)),
        "min0maxpi": MinMaxScaler((0, np.pi)),
        "min0max2pi": MinMaxScaler((0, 2 * np.pi)),
        "min-1max1": MinMaxScaler(feature_range=(-1, 1)),
        "min-pimaxpi": MinMaxScaler(feature_range=(-np.pi, np.pi)),
        "min-pi-maxpi": MinMaxScaler(feature_range=(-np.pi, np.pi)),
    }
    
    if scaler_type not in scalers:
        raise ValueError(f"Invalid scaler_type '{scaler_type}'. Must be one of {list(scalers.keys())}.")
    
    return scalers[scaler_type]
