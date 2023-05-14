import pandas as pd
import numpy as np

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# pd.options.mode.dtype_backend = "pyarrow"
# Extract data from the source file with only close price
def extract_and_process_data(missing: float) -> DataFrame:
    '''
    1. Extract adj. close data from the S&P 500 Stocks 
    2. Process the data with missing percentage > User-Defined Ratio

    Parameters:
    -------------------
    missing: float 64 - Missing ratio Ranging from 0 to 1

    Returns:
    ------------------
    data: DataFrame storing all the adjusted close data
    '''
    data = pd.read_csv("S&P500_Stock.csv", index_col = 0)

    # Drop all the data that with missing data > 0.2
    print('Data Shape before cleaning =', data.shape)

    missing_percentage = data.isnull().mean().sort_values(ascending=False)
    dropped_list = sorted(list(missing_percentage[missing_percentage > missing].index))
    data.drop(labels=dropped_list, axis=1, inplace=True)
    data = data.fillna(method='ffill')

    print('Data Shape after cleaning =', data.shape)
    
    return data

def metrics_process(data: DataFrame, metrics: str = 'cumret') -> DataFrame:
    '''
    Choose the metrics for testing .

    Parameters:
    -------------------
    data: pd.DataFrame - time series data
    metrics: Metrics to choose from our pool of cases

    Returns:
    ------------------
    data_new: pd.DataFrame with metrics calculated
    '''

    if metrics == 'cumret':
        cumret = np.log(data).diff().cumsum() + 1
        # cumret = cumret.dropna()
    if metrics == 'logret':
        cumret = data.apply(lambda x: np.log(x) - np.log(x.shift(1)))
    return cumret

def test_PCA_Components(df: DataFrame, components: int = 10) -> DataFrame:
    '''
    Test PCA Component And show its explanatory variance.

    Parameters:
    -------------------
    df: pd.DataFrame - time series data (columns = date, index = stock)
    max_components: Number of Dimensionality after reduction

    Returns:
    ------------------
    df_pca: pd.DataFrame DataFrame storing data after PCA
    '''

    metrics = df.dropna()
    scale = StandardScaler().fit(metrics.T)
    
    # Fit the scaler
    scaled_data = pd.DataFrame(scale.fit_transform(metrics.T),columns = metrics.T.columns, index = metrics.T.index)
    X = scaled_data
    
    # PCA Fit
    pca = PCA(n_components= components)
    pca.fit(X.T)
    
    print("Components", len(pca.explained_variance_ratio_), "\nExplained Variance", pca.explained_variance_ratio_.sum())
    
    df_pca = pd.DataFrame(data = pca.components_.T, index = X.index)
    
    return df_pca
