import numpy as np
import pandas as pd 

from pandas import DataFrame
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from hurst import compute_Hc
from statsmodels.tsa.stattools import adfuller
    
def calculate_halflife(spread: list) -> float:
    '''
    calculate half-life of mean reversion of the spread
    '''
    
    ylag = spread.shift()
    deltay = spread - ylag
    ylag.dropna(inplace=True)
    deltay.dropna(inplace=True)

    res = OLS(deltay, add_constant(ylag)).fit()
    halflife = -np.log(2)/res.params[0]
    
    return halflife

def compute_stationary_long(df: DataFrame) -> DataFrame:
    '''
    Calculate the stationary metrics according to price series.
    '''
    tmp_df = pd.DataFrame(index = df.columns)
    df = df.dropna(how = 'all')
    for col in df.columns:
        tmp_df.loc[col, "ADF_value"] = adfuller(df[col])[1]
        tmp_df.loc[col, 'Hurst Exponent'] = compute_Hc(df[col])[0]
        tmp_df.loc[col, 'Half-life mean reversion'] = calculate_halflife(df[col])
    return tmp_df

def choose_stocks(df: DataFrame, metrics: list) -> DataFrame:
    '''
    Choose the stocks according to some chosen metrics
    '''
    tmp_df = df
    
    for metric in metrics:
        if metric == 'Hurst Exponent':
            tmp_df.loc[tmp_df['Hurst Exponent'] < 0.5, "Chosen"] = 1
        if metric == 'ADF_value':
            tmp_df.loc[tmp_df['ADF_value'] < 0.05, "Chosen"] = 1   
        if metric == 'Half-life mean reversion':
            tmp_df.loc[abs(tmp_df['Half-life mean reversion']) < 40, "Chosen"] = 1

    return tmp_df[tmp_df["Chosen"] == 1]

