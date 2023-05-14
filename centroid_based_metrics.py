from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from pandas import DataFrame

import plotly.express as px
import pandas as pd
import numpy as np

def gmm_js(gmm_p, gmm_q, n_samples=10**2):
    """
    Calculates the Jensen-Shannon (JS) distance between two Gaussian Mixture Models (GMMs).
    Parameters:
        gmm_p: The first Gaussian Mixture Model.
        gmm_q: The second Gaussian Mixture Model.
        n_samples: Number of samples to generate from each GMM. Default is 10^2.
    Returns:
        The JS distance between the two GMMs.
    """
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2)) + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2

def GMM_test(X: DataFrame, max_iter: int) -> DataFrame:
    """
    Runs the Gaussian Mixture Model algorithm on the given data X and returns evaluation metrics for each cluster size.
    Parameters:
        X: The input data in DataFrame format.
        max_iter: The maximum number of clusters to test.
    Returns:
        A DataFrame containing the evaluation metrics for each cluster size.
    """
    eval_list = []
    print("The Score is listed in this order: BIC, Silhouette_score, JS Distance")
    for n_components in range(2, max_iter + 1):
        
        # Fit the GMM to the data
        gmm = GaussianMixture(n_components= n_components, random_state=0)
        labels = gmm.fit_predict(X)
        # Calculate the log-likelihood of the data under the GMM
        log_likelihood = gmm.score(X)
        dist = {}
        # Calculate the BIC score
        n_samples, n_features = X.shape
        
        p = n_components * (n_features + n_features*(n_features + 1)/ 2 + 1) -1   # Number of parameters in the GMM
        # bic = -2 * log_likelihood + p * np.log(n_samples)
        bic = gmm.bic(X)
        # Calculate the JS Distance 

        # Randomly split half of the data in X
        indices = np.random.permutation(len(X))
        split = int(0.5 * len(X))

        X_train, X_test = X.iloc[indices[:split], :], X.iloc[indices[split:], :]
        
        gmm_p = GaussianMixture(n_components= n_components, random_state=0)
        gmm_p.fit(X_train)

        gmm_q = GaussianMixture(n_components= n_components, random_state=0)
        gmm_q.fit(X_test)

        # Compute all the metrics
        JS_distance = gmm_js(gmm_p, gmm_q)
        score = silhouette_score(X, labels)
        result = (bic, score, JS_distance)
        
        score_list = []
        score_list.append(bic)
        score_list.append(score)
        score_list.append(JS_distance)
        eval_list.append(score_list)
        print("N-Cluster:", n_components, " ", result)

    df = pd.DataFrame(eval_list, columns = ["BIC", "silhouette_score", "JS Distance"])
    df.index.name = 'Evaluation Metrics'
    return df

def GMM_select(X: DataFrame, cluster_no: int) -> DataFrame:
    '''
    Select the cluster number and return a dataframe storing all the labels correspondence
    '''
    gmm = GaussianMixture(n_components= cluster_no, random_state=0)
    labels = gmm.fit_predict(X)
    
    new = pd.DataFrame()
    new["stocks"] = X.index
    new['labels'] = labels

    return new.groupby('labels').agg(list)

def plot_all_metrics(df: DataFrame) -> None:
    '''
    Plot three metrics 
    '''
    fig = px.line(df['BIC'])
    fig.show()

    fig = px.line(df['silhouette_score'])
    fig.show()

    fig = px.line(df['JS Distance'])
    fig.show()