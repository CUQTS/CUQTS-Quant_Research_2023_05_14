from pykalman import KalmanFilter
from pandas import DataFrame

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

def calculate_kalman_spread(metrics: DataFrame, obs_stock: str, hidden_stock: str, start: str, end: str, plot: bool = False):
    # Define some metricc
    metrics = metrics.loc[start: end]
    metrics = metrics.dropna(how = 'all').apply(lambda x: x / x.iloc[0])
    
    # Define Observation Matrix
    obs_mat = sm.add_constant(metrics[obs_stock].dropna(how = 'all').values, prepend=False)[:, np.newaxis]
    
    # Define Transition Covariance 
    trans_cov = 1e-5 / (1 - 1e-5) * np.eye(2)
    
    kf = KalmanFilter(
        n_dim_obs= 1,   # Observation Matrix Dimension
        n_dim_state = 2, # State Matrix Dimension
        initial_state_mean= np.array([1, 0]), # (beta, y-intercept) -> initialized as 1 
        initial_state_covariance= np.ones((2, 2)),
        transition_matrices= np.eye(2), # State Transition matrix - which is default as [[1, 0], [0,1]]
        observation_matrices= obs_mat,  # Observation Matrices 
        observation_covariance= 1,
        transition_covariance= trans_cov, 
        em_vars = ['transition_matrices', 'observation_matrices', 'observation_covariance', 'transition_covariance']
    )

    # Result of the Kalman Filter
    state_means, state_covs = kf.filter(metrics[hidden_stock].dropna(how = 'all').values)

    # Spread calculated using Kalman Filter
    kl_spread = metrics[obs_stock].dropna(how = 'all') - (metrics[hidden_stock].dropna(how = 'all') * state_means[:,0] + state_means[:,1])
    
    if plot:
        fig, ax = plt.subplots(figsize = (12, 6))
        kl_spread.plot(ax=ax, label ='KM Spread')
        new = metrics[obs_stock] - metrics[hidden_stock]
        new.plot(ax = ax, label = 'Normal Spread')
        ax.legend()

    return kl_spread

def backtest_pairs(data: DataFrame, km_spread: DataFrame, s1: str, s2: str, start: str, end: str):
 
    # Initialize backtest Dataframe
    bt_df = pd.DataFrame()
    data = data.loc[start:end]
    km_spread = km_spread.loc[start:end]
    
    # Get the return from one period to another
    bt_df["return - pair 1"] = data[s1].pct_change()
    bt_df["return - pair 2"] = data[s2].pct_change()
    
    # Calculate the original spread
    bt_df["spread"] = km_spread 
    bt_df["spread_mean"] = bt_df["spread"].rolling(30).mean()
    bt_df["spread_std"] = bt_df["spread"].rolling(30).std()
    
    # Demos for the upper bound and lower bound
    bt_df["upper_bound"] = bt_df["spread_mean"] + 1.8 * bt_df["spread_std"]
    bt_df["lower_bound"] = bt_df["spread_mean"] - 1.8 * bt_df["spread_std"]
    
    # Check Crossover & buy and sell signals
    
    bt_df["buy_signal_2"] = np.where((bt_df["spread"] < bt_df["spread_mean"] - 1.8 * bt_df["spread_std"]), 1, 0)
    bt_df["sell_signal_1"] = np.where((bt_df["spread"] < bt_df["spread_mean"] - 1.8 * bt_df["spread_std"]), 1, 0)
    
    bt_df["buy_signal_1"] = np.where((bt_df["spread"] > bt_df["spread_mean"] + 1.8 * bt_df["spread_std"]), 1, 0)
    bt_df["sell_signal_2"] = np.where((bt_df["spread"] > bt_df["spread_mean"] + 1.8 * bt_df["spread_std"]), 1, 0)
    
    # Take Profit Signal
    bt_df["tp_signal"] = np.where(((bt_df['spread'] > bt_df["spread_mean"]) | (bt_df['spread'] < bt_df["spread_mean"])), 1, 0)
    bt_df["tp_signal"] = np.where(bt_df["tp_signal"].diff() == 1, 1, 0) 
  
    bt_df.loc[bt_df["tp_signal"] == 1, ['buy_signal_1', 'buy_signal_2', 'sell_signal_1', 'sell_signal_2']] = 0
    bt_df[['buy_signal_1', 'buy_signal_2', 'sell_signal_1', 'sell_signal_2']] = bt_df[['buy_signal_1', 'buy_signal_2', 'sell_signal_1', 'sell_signal_2']].fillna(method='ffill')
    
    # Compute Cumulative Returns
    bt_df["Cumulative Return"] = bt_df["buy_signal_1"] * bt_df["return - pair 1"] + bt_df["buy_signal_2"] * bt_df["return - pair 2"] - bt_df["sell_signal_1"] * bt_df["return - pair 1"] - bt_df["sell_signal_2"] * bt_df["return - pair 2"]
    bt_df["Cumulative Return"] = bt_df["Cumulative Return"].fillna(0)
    
    bt_df["check"] = (bt_df["Cumulative Return"] + 1).cumprod()
    
    # Get the result of buy and hold strategies V.S. Pair Trading Strategies
    base_result_1 = (data.iloc[-1][s1] - data.iloc[0][s1]) / data.iloc[0][s1]
    base_result_2 = (data.iloc[-1][s2] - data.iloc[0][s2]) / data.iloc[0][s2]
    enhanced_result = (bt_df["Cumulative Return"] + 1).cumprod().iloc[-1] - 1

    pct_rep_1 = "{:.2%}".format(base_result_1)
    pct_rep_2 = "{:.2%}".format(base_result_2)
    enhanced_rep = "{:.2%}".format(enhanced_result)

    print(f"Calculation for {s1} - {s2} Pairs")
    print(f"Buy-And-Hold Stratgies for {s1} generates {pct_rep_1}")
    print(f"Buy-And-Hold Stratgies for {s2} generates {pct_rep_2}")
    print(f"Pair Trading Stratgies for kalman spread generates {enhanced_rep}\n")

    return base_result_1, base_result_2, enhanced_result

