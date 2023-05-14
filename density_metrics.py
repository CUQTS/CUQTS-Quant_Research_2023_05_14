import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AffinityPropagation, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from pandas import DataFrame
from kneed import KneeLocator

def plot_distance(df: DataFrame) -> None:
    '''
    This function takes a pandas DataFrame and returns None. It computes the distance between the k-th neighbors of each point in the DataFrame and plots the results on a line chart. It also returns the computed distance at the elbow point of the chart.
    '''
    neighbors = np.shape(df)[1] * 2 - 1
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(df) # Fit the DataFrame into k-th neighbros
    distances, indices = nbrs.kneighbors(df) # Find distances 
    distance_desc = sorted(distances[:, neighbors - 1], reverse=True) # Extract the largest distance in K-th neighbors

    kneedle = KneeLocator(range(1,len(distance_desc)+1),  # x values
                      distance_desc, # y values
                      S=1.0,   #parameter suggested from paper
                      curve="convex", #parameter from figure
                      direction="decreasing") #parameter from figure
    
    print("The computed distance(eps)", kneedle.knee_y)

    fig = px.line(x=list(range(1,len(distance_desc )+1)), y= distance_desc, title = 'Compute elbow point with KNN') # Draw a line
    fig.update_xaxes(title_text='Index')
    fig.update_yaxes(title_text='Distance')

    fig.show()

    return kneedle.knee_y

def DBSCAN_labels(df_pca: DataFrame, eps: int, min_samples: int) -> DataFrame:
    '''
    This function takes a pandas DataFrame, an integer value for the epsilon (eps) parameter, and an integer value for the minimum number of samples (min_samples). It performs DBSCAN clustering on the DataFrame using the given eps and min_samples values. It returns a pandas DataFrame that groups the labels and stocks by cluster.
    '''
    try:
        db = DBSCAN(eps = eps, min_samples = min_samples)
        labels = db.fit_predict(df_pca)
        
        # Silhouette Score determines how distant one cluster is classified from another one
        stock_label = pd.DataFrame()
        stock_label["label"] = labels
        stock_label["stock"] = df_pca.index

        print("eps: ", eps, "min_samples", min_samples)
        print("Score:", silhouette_score(df_pca, labels))
        
        return stock_label.groupby('label').agg(list)
    except:
        print("No Score Got")
        pass



def new_approach(df: DataFrame, add: int) -> list:
    '''
    This function implemented the strategy proposed in the research paper
    while add parameters is the parameters user-specified to determined how
    many steps to go when elbow point could not change cluster result
    '''
    def find_best_eps(prev: int, df: DataFrame, addition: int) -> float:
        # Find the knee point which is chosen for eps
        neighbors = df.shape[1] * 2
        nbrs = NearestNeighbors(n_neighbors=neighbors - 1).fit(df)
        distances, indices = nbrs.kneighbors(df) 
        distance_desc = sorted(distances[:, neighbors - 2], reverse=True)
        
        kneedle = KneeLocator(range(1,len(distance_desc)+1),  # x values
                        distance_desc,                  # y values
                        S = 1.0,                        # parameter suggested from paper
                        curve="convex",                 # parameter from figure
                        direction="decreasing")         # parameter from figure
        
        # If there's no change in knee point after updates, then add the index to "add" values specified by users
        if prev == kneedle.knee:
            previous = kneedle.knee + addition
            return previous, distance_desc[kneedle.knee + addition]

        previous = kneedle.knee 
        return previous, kneedle.knee_y

    def DBSCAN_label(eps: float, df_pca: DataFrame, min_samples: int, roll_df: DataFrame, addition: int) -> DataFrame:
        db = DBSCAN(eps = eps, min_samples = min_samples)
        labels = db.fit_predict(df_pca)
        
        # Silhouette Score determines how distant one cluster is classified from another one
        if len(np.unique(labels)) != 1:
            print("SIL Score", silhouette_score(df_pca, labels))
        stock_label = pd.DataFrame()
        stock_label["label"] = labels
        stock_label["stock"] = df_pca.index
        
        DB_list = stock_label.groupby('label').agg(list)
        DB_Result = DB_list.loc[0].stock

        if list(df_pca.index) == list(DB_Result):
            addition += add

        return df_pca.T[DB_Result].T, addition
    
    new_df = df 
    prev = 0
    addition = add
    tmp_df = pd.read_csv('roll_df.csv', index_col=0)

    for i in range(1000):
        try:
            # Find eps for latest DBSCAN
            prev, eps = find_best_eps(prev, new_df, addition)
            
            # Update the addition value
            prev_addition = addition

            # Return a new dataframe after updates with the DBSCAN based on latest eps value
            new_df, addition = DBSCAN_label(eps, new_df, 10, tmp_df, addition)

            # If clusters remains the same, we make an addition to the distance index
            if addition > add and addition == prev_addition: 
                addition = add
        except:
            return list(new_df.index)
        
def GMM_test(X: DataFrame, max_iter: int) -> None:
    print("The Score is listed in this order: Silhouette_score, bic, aic")
    for n_components in range(2, max_iter):
        # Fit the GMM to the data
        gmm = GaussianMixture(n_components= n_components, random_state=0)
        labels = gmm.fit_predict(X)
        
        # Calculate the log-likelihood of the data under the GMM
        log_likelihood = gmm.score(X)

        # Calculate the BIC score
        n_samples, n_features = X.shape
        
        p = n_components * (n_features + n_features*(n_features + 1)/ 2 + 1) -1   # Number of parameters in the GMM
        bic = -2 * log_likelihood + p * np.log(n_samples)
        aic = -2 * log_likelihood + 2 * p
        score = silhouette_score(X, labels)
        result = (bic, aic, score)
        print("N-Cluster:", n_components, " ", result)


        
    