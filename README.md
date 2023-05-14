# Time Series Analysis: Pair Trading

Time series data analysis is a critical task in finance, and its complexities are
often compounded by factors such as seasonality, trends, and irregularities.
Traditional statistical methods may not be sufficient to analyze such data, and
this is where machine learning techniques can provide a valuable toolset for
identifying patterns and relationships.

In this paper, we propose a novel approach to selecting the best pairs for
trading using a combination of machine learning and statistical methods.
Specifically, we explore the use of principal component analysis (PCA),
density-based spatial clustering of applications with noise (DBSCAN),
Gaussian mixture models (GMM), and soft dynamic time warping (Soft-DTW)
to analyze time series data and identify pairs that are most suitable for
trading. We also introduce the use of the non-dominated sorting genetic
algorithm II (NSGA-II) as a method for pair selection that has just recently
started the application.
Following the selection of the best pairs for trading, we conducted
backtesting against our strategy using traditional statistical approaches such
as standard deviation. Additionally, we introduced the use of Kalman Filter to
dynamically adjust certain factors within the market and between the two
selected stocks. This approach allowed us to further refine our trading
strategy and improve its effectiveness, and determine which could be a
better candidate for choices.

The reference paper can be found in https://www.cuqts.com/
