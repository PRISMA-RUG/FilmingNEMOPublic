import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from featboost.feat_boost import FeatBoostClassifier
import tsfresh

def distance_thumb_index(data, index=20, thumb=22, window=10):
    duration = len(data["Data X"])

    index_mov = np.array([np.reshape(data["Data X"], (33, duration))[index], np.reshape(data["Data Y"], (33, duration))[index],
                          np.reshape(data["Data Z"], (33, duration))[index]])
    thumb_mov = np.array([np.reshape(data["Data X"], (33, duration))[thumb], np.reshape(data["Data Y"], (33, duration))[thumb],
                          np.reshape(data["Data Z"], (33, duration))[thumb]])

    result = np.linalg.norm(index_mov - thumb_mov, axis=0)
    series = pd.Series(result)
    result = series.rolling(window=window).mean()
    result = result[window:]
    result = result - min(result)
    result = result*(1/max(result))

    return result


def make_time_series(raw_data, keypoint):
    ts_list = []
    for i in range(0, len(raw_data)):
        ts = raw_data.iloc[i]
        ts = np.reshape(ts, (33, -1))
        ts_list.append(ts[keypoint])

    return ts_list


def extract_features_from_time_series(time_series, series):
    long_df = time_series.explode(series).reset_index()
    long_df = long_df.rename(columns={"Patient ID": "id", series: "value"})[["id", "value"]]
    long_df["time"] = long_df.groupby("id").cumcount()
    long_df['value'] = pd.to_numeric(long_df['value'], errors='raise')
    long_df['id'] = pd.to_numeric(long_df['id'], errors='raise')

    features = tsfresh.extract_features(long_df, column_id="id", column_sort="time", column_value="value", n_jobs=0,
                                        disable_progressbar=True,)
    features = features.dropna(axis=1)
    X = np.array(features.iloc[:, :-1].reset_index().drop(columns="index"))

    return X

def PCA_transform(X, variance = 0.75):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Perform PCA without specifying the number of components
    pca_90 = PCA()
    X_pca_90 = pca_90.fit_transform(X_std)

    # Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(pca_90.explained_variance_ratio_)

    # Find the number of components that explain at least 90% of the variance
    n_components_90 = np.argmax(cumulative_variance >= variance) + 1

    # Perform PCA again with the optimal number of components
    pca_90_optimal = PCA(n_components=n_components_90)
    X_90 = pca_90_optimal.fit_transform(X_std)
    return X_90, scaler, pca_90_optimal

def rfe_select(X_train, X_test, y_train, clf, n_features=5):
    # Initialize RFE with the model and the number of features to select
    rfe = RFE(estimator=clf, n_features_to_select=n_features)

    # Fit RFE to the training data
    rfe.fit(X_train, y_train)

    return rfe.transform(X_train), rfe.transform(X_test), rfe.support_


def featboost_select(X_train, X_test, y_train, clf, n_features=5):

    featboost = FeatBoostClassifier(
        estimator=[clf, clf],
        number_of_folds=5,
        siso_ranking_size=n_features,
        max_number_of_features=n_features,
        siso_order=1,
        epsilon=1e-18,
        verbose=0,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    featboost.fit(X_train, y_train)

    return X_train[:, featboost.selected_subset_], X_test[:, featboost.selected_subset_], featboost.selected_subset_