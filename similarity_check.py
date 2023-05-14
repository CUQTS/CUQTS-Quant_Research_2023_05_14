import pandas as pd
import numpy as np
from pandas import DataFrame

def cadf_pvalue(s1, s2, cumret):
    '''
    perform CADF cointegration tests
    since it is sensitive to the order of stocks in the pair, perform both tests (s1-2 and s2-s1)
    return the smallest p-value of two tests
    '''
    from statsmodels.tsa.stattools import coint
    p1 = coint(cumret[s1], cumret[s2])[1]
    p2 = coint(cumret[s2], cumret[s1])[1]
    
    return min(p1,p2)

def calculate_halflife(spread):
    '''
    calculate half-life of mean reversion of the spread
    '''
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    
    ylag = spread.shift()
    deltay = spread - ylag
    ylag.dropna(inplace=True)
    deltay.dropna(inplace=True)

    res = OLS(deltay, add_constant(ylag)).fit()
    halflife = -np.log(2)/res.params[0]
    
    return halflife

def calculate_metrics(pair_list:list , cumret: DataFrame, start: str, end:str):
    '''
    calculate metrics for N pairs with the smallest Euclidean distance
    return dataframe of results
    '''
    from hurst import compute_Hc
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import grangercausalitytests
    

    cols = ['Euclidean distance', 'ADF p-value', 'Granger Casuality p-value', 'Hurst Exponent', 'Half-life of mean reversion', 'Spread SD', 
        'Num zero-crossings', '% days within 2-SD band']
    index_list = [f'{a}-{b}'  for a, b in pair_list]
    results = pd.DataFrame(index = index_list, columns = cols)
    cumret = cumret.loc[start: end].dropna()
    
    for pair in pair_list:
        s1, s2 = pair[0], pair[1]
        spread = cumret[s1] - cumret[s2]
        # Distance of the spread
        results.loc[f'{s1}-{s2}']['Euclidean distance'] = np.sqrt(np.sum((spread)**2))
        
        # Testing with P-Value
        results.loc[f'{s1}-{s2}']['ADF p-value'] = adfuller(spread)[1]
        
        # Check if there'll be a lot of noise between s1 f'{s1}-{s2}' and s2 f'{s1}-{s2}' (with lag 1) -> explain some causualisation with their lag
        # Check time series s1 is really affected by s2
        tmp_df = pd.DataFrame(data = zip(cumret[s1][1:], cumret[s2][:-1]))
        results.loc[f'{s1}-{s2}']['Granger Casuality p-value'] = grangercausalitytests(tmp_df, maxlag=2, verbose = False)[1][0]["ssr_ftest"][1]
       
        # Nature of Stocks Movement
        results.loc[f'{s1}-{s2}']['Hurst Exponent'] = compute_Hc(spread)[0]
        results.loc[f'{s1}-{s2}']['Half-life of mean reversion'] = calculate_halflife(spread)
        
        # Basics of the spread information
        results.loc[f'{s1}-{s2}']['Spread SD'] = spread.std()
        results.loc[f'{s1}-{s2}']['Num zero-crossings'] = ((spread[1:].values * spread[:-1].values) < 0).sum()
        results.loc[f'{s1}-{s2}']['% days within 2-SD band'] = (abs(spread) < 2*spread.std()).sum() / len(spread) * 100
    
        
    return results