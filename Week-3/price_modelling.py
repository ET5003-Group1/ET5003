import os
import sys
import pandas as pd
import numpy as np
import pymc3 as pm
import theano as tt
import matplotlib.pyplot as plt
import logging

# Sklearn
import sklearn.datasets as dt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# MLflow to track metrics
from datetime import datetime
import mlflow
from mlflow import log_metric, log_param

runtime = datetime.now().strftime("%Y%m%d_%H%M%S")
mlflow.set_experiment(runtime)

# Setup logging
log_format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(
    # filename = f"logfile_{runtime}.log",
    stream = sys.stdout,
    filemode = "w",
    format = log_format, 
    level = logging.INFO)
logger = logging.getLogger()
logger.info(f"Running experiment: {runtime}")

NUM_COLS = ["bathrooms", "beds", "latitude", "longitude", "surface"]
ONE_HOT_COLS = ["property_type"]
ORDINAL_COLS = ["ber_classification"]


def load_datasets(dir_name="house_prices"):
    """Loads house prices train and test set"""
    fpath = os.path.join(os.getcwd(), dir_name)
    logger.info(f"Loading files from {fpath}")
    train_fname = "house_train.csv"
    test_fname = "house_test.csv"
    cost_fname = "true_price.csv"

    train_set = pd.read_csv(os.path.join(fpath, train_fname)).drop(columns="ad_id")
    X_test = pd.read_csv(os.path.join(fpath, test_fname)).drop(columns="ad_id")
    y_test = pd.read_csv(os.path.join(fpath, cost_fname)).drop(columns="Id")
    test_set = pd.concat([X_test, y_test], axis=1).rename(columns={"Expected": "price"})
    logger.info(f"\tLoaded train_set of shape {train_set.shape}")
    logger.info(f"\tLoaded test_set of shape {train_set.shape}")
    return train_set, test_set


def prepare_train_set(
    train_set,
    drop_cols=[
        "area",
        "county",
        "description_block",
        "environment",
        "facility",
        "features",
        # "has_parking",
        "no_of_units",
        "property_category",
        "price",
    ],
    target="log_price",
):
    column_order = NUM_COLS + ORDINAL_COLS + ONE_HOT_COLS
    processed = train_set[train_set["price"].notna()]
    processed = processed[processed["latitude"] > 53]
    processed[target] = np.log(processed["price"])
    processed = processed.drop(columns=drop_cols).loc[:, column_order + [target]]

    X, y = (
        processed.drop(columns=target),
        processed[target].copy().values.reshape(-1, 1),
    )
    logger.info("Processed train_set:")
    logger.info(f"\tX_train shape: {X.shape}")
    logger.info(f"\ty_train shape: {y.shape}")
    return X, y


def prepare_test_set(
    test_set,
    drop_cols=[
        "area",
        "county",
        "description_block",
        "environment",
        "facility",
        "features",
        # "has_parking",
        "no_of_units",
        "property_category",
        "price",
    ],
    target="log_price",
):
    column_order = NUM_COLS + ORDINAL_COLS + ONE_HOT_COLS
    test_set[target] = np.log(test_set["price"])
    test_set = test_set.drop(columns=drop_cols).loc[:, column_order + [target]]

    X, y = test_set.drop(columns=target), test_set[target].copy().values.reshape(-1, 1)
    logger.info("Processed test_set:")
    logger.info(f"\tX_test shape: {X.shape}")
    logger.info(f"\ty_test shape: {y.shape}")
    return X, y


def feature_engineering(X_train, y_train):
    scaler = StandardScaler()
    y_scaler = StandardScaler()
    ordinal_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    one_hot_enc = OneHotEncoder(sparse=False)

    X_train_scaled = X_train.reset_index(drop=True).copy()

    # Fit and transform X
    X_train_scaled[NUM_COLS] = scaler.fit_transform(X_train_scaled[NUM_COLS])
    X_train_scaled[ORDINAL_COLS] = ordinal_enc.fit_transform(
        X_train_scaled[ORDINAL_COLS]
    )
    X_one_hot = one_hot_enc.fit_transform(X_train_scaled[ONE_HOT_COLS])
    X_one_hot = pd.DataFrame(X_one_hot, columns=one_hot_enc.categories_)
    X_train_scaled = pd.concat(
        [X_train_scaled.drop(columns=ONE_HOT_COLS), X_one_hot],
        axis=1,
        ignore_index=True,
    ).to_numpy(dtype=np.float32)

    # Fit and transform y
    y_train_scaled = y_scaler.fit_transform(y_train)

    logger.info("Fit and transformed train_set:")
    logger.info(f"\tX_train shape: {X_train_scaled.shape}")
    logger.info(f"\ty_train shape: {y_train_scaled.shape}")
    return (
        X_train_scaled,
        y_train_scaled,
        [scaler, ordinal_enc, one_hot_enc],
        y_scaler,
    )


def transform_test_set(X_test, y_test, X_transforms, y_scaler):
    scaler, ordinal_enc, one_hot_enc = X_transforms

    X_test_scaled = X_test.copy()

    # Transform test set
    X_test_scaled[NUM_COLS] = scaler.transform(X_test_scaled[NUM_COLS])
    X_test_scaled[ORDINAL_COLS] = ordinal_enc.transform(X_test[ORDINAL_COLS])
    X_one_hot = one_hot_enc.transform(X_test_scaled[ONE_HOT_COLS])
    X_one_hot = pd.DataFrame(X_one_hot, columns=one_hot_enc.categories_)
    X_test_scaled = pd.concat(
        [X_test_scaled.drop(columns=ONE_HOT_COLS), X_one_hot], axis=1, ignore_index=True
    ).to_numpy(dtype=np.float32)

    # Transform test target
    y_test_scaled = y_scaler.transform(y_test)
    logger.info("Transformed test:")
    logger.info(f"\tX_test shape: {X_test_scaled.shape}")
    logger.info(f"\ty_test shape: {y_test_scaled.shape}")
    return X_test_scaled, y_test_scaled


def feature_sampling(X_train, X_test, features="num_only"):
    if features == "num_only":
        # Numerical features are the first ones in the dataset
        features = range(len(NUM_COLS))

    param_feature_names = ("feat_indexes", list(features))
    log_param(*param_feature_names)
    logging.info(f"Selecting feature indexes: {', '.join([str(idx) for idx in features])}")


    X_train = X_train[:, features]
    X_test = X_test[:, features]

    return X_train, X_test


def define_lin_reg(
    predictors,
    observed,
    model_name,
    n_iterations=30_000,
    n_samples=5_000,
    alpha=("Normal", 0, 10),
    beta=("Normal", 0, 10),
    sigma=("HalfCauchy", 5),
    plot_loss=False,
):
    """Defines and trains a Bayesian linear regression model:
            mu ~ alpha + beta * predictors
    With likelihood
            likelihood ~ N(mu, sigma)
    Where alpha, beta and sigma are pymc distributions defined by the user.

    Parameters
    ----------
    predictors : np.ndarray
        Numpy array with model features.
    observed : np.ndarray
        Numpy array with observed values of the target feature. Preferably as a 1-D array
        to speed up fitting time.
    model_name : str
        Identifier for the model being defined
    n_iterations : int
        The number of iterations for fitting.
    n_samples : int
        The number of samples to draw for the posterior.
    alpha : tuple(string, int, [int, ])
        Prior distribution of alpha. The first argument should be a string with
        a pymc3 model followed by 1 or more integer arguments for the parameters of that distribution.
    beta : tuple(string, int, [int, ])
        Prior distribution of beta. The first argument should be a string with
        a pymc3 model followed by 1 or more integer arguments for the parameters of that distribution.
    sigma : tuple(string, int, [int, ])
        Prior distribution of sigma. The first argument should be a string with
        a pymc3 model followed by 1 or more integer arguments for the parameters of that distribution.

    Returns
    -------
    posterior : pymc3.backends.base.MultiTrace
        Posterior distribution estimated by pymc model.
    """
    log_param(f"{model_name}_alpha", alpha)
    log_param(f"{model_name}_beta", beta)
    log_param(f"{model_name}_sigma", sigma)
    log_param(f"{model_name}_n_iterations", n_iterations)
    log_param(f"{model_name}_n_samples", n_samples)

    with pm.Model() as model:
        alpha = getattr(pm, alpha[0])("alpha", *alpha[1:])
        beta = getattr(pm, beta[0])("beta", *beta[1:], shape=predictors.shape[1])

        mu = alpha + pm.math.dot(beta, predictors.T)

        sigma = getattr(pm, sigma[0])("sigma", *sigma[1:])
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=observed)
        approximation = pm.fit(n_iterations, method="advi")
        posterior = approximation.sample(n_samples)
    if plot_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(approximation.hist, alpha=0.7)
        plt.title("Full model loss")
        plt.xlabel("n_iterations")
        plt.ylabel("log(loss)")
        plt.yscale("log")
        plt.grid(True, which="both", axis="y", linestyle="--")
        plt.show()
    return posterior


def mean_absolute_error(y_true, y_pred):
    """Calculates the Mean Absolute Error between y_true and y_pred
    
    Parameters
    ----------
    y_true : 

    y_pred : 

    Returns
    -------
    np.ndarray
        The mean absolute error of the predictions
    """
    return np.mean(abs(y_true - y_pred))


def mape(y_true, y_pred):
    """Calculates the MAPE between y_true and y_pred
    
    Parameters
    ----------
    y_true : 

    y_pred : 

    Returns
    -------
    np.ndarray
        The MAPE of the predictions
    """
    return np.mean(abs(y_true - y_pred) / y_true)


def predict(posterior, X, y_scaler):
    """Calculates the predictions for a given X based on a learned posterior
    
    Parameters
    ----------
    posterior : pymc3.backends.base.MultiTrace.
        Posterior distribution estimated by pymc model.
    X : np.ndarray
        Input features of data to estimate.
    y_scaler : sklearn.preprocessing._data.StandardScaler
        Scaler used to transform the predictor variable.
    Returns
    -------
    np.ndarray
        The model predictions.
    """
    log_likelihood = np.mean(posterior["alpha"]) + np.dot(
        np.mean(posterior["beta"], axis=0), X.T
    )
    y_pred = np.exp(y_scaler.inverse_transform(log_likelihood.reshape(-1, 1)))
    return y_pred


def evaluate(posterior, X, y, y_scaler, model_name, dataset_name):
    """Generates predictions for a dataset and evaluates MAE and MAPE.
    
    Parameters
    ----------
    posterior : pymc3.backends.base.MultiTrace.
        Posterior distribution estimated by pymc model.
    X : np.ndarray
        Input features of data to estimate.
    y : np.ndarray
        Observed values of target feature.
    y_scaler : sklearn.preprocessing._data.StandardScaler
        Scaler used to transform the predictor variable.
    model_name : str
        Identifier for the model being used.
    model_name : str
        Identifier for the dataset being used.
    Returns
    -------
    None
    """
    y_pred = predict(posterior, X, y_scaler)
    mae, mape_ = mean_absolute_error(y, y_pred), mape(y, y_pred)

    log_metric(f"{model_name}_{dataset_name}_mae", mae)
    log_metric(f"{model_name}_{dataset_name}_mape", mape_)
    logger.info(f"{model_name} metrics on {dataset_name}")
    logger.info("\tMAE = ", mae)
    logger.info("\tMAPE = ", mape_)
    return y_pred


def run_full_model(
    X_train,
    y_train,
    X_test,
    y_test,
    y_scaler,
    alpha=("Normal", 0, 30),
    beta=("Normal", 0, 30),
    sigma=("HalfCauchy", 5),
):
    logger.info("Defining and fitting full model...")
    full_posterior = define_lin_reg(
        X_train, y_train.ravel(), "full_model", alpha=alpha, beta=beta, sigma=sigma,
    )
    logger.info("Evaluating full model")
    evaluate(full_posterior, X_train, y_train, y_scaler, "full_model", "train")
    evaluate(full_posterior, X_test, y_test, y_scaler, "full_model", "test")


def gmm_clustering(X_train, lat_lon_idx=[2, 3], n_clusters=4):
    logger.info(f"Running GMM clustering with {n_clusters} clusters")
    log_param("n_clusters", n_clusters)
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(X_train)
    return gmm


def prepare_clustered_data(
    X, y, cluster_labels, n_clusters, y_scalers=None, method="transform"
):
    """
    y_scalers is defaulted to none, meaning it will use method="fit_transform" on the y dataset
    if y_scalers is not none, then it only transforms the targets
    """
    # Prepare clustered data
    X_clusters = [X[cluster_labels == idx] for idx in range(n_clusters)]
    if not y_scalers:
        y_scalers = [StandardScaler() for _ in range(n_clusters)]
        method = "fit_transform"

    y_clusters = [
        getattr(y_scaler, method)(y[cluster_labels == idx].reshape(-1, 1))
        for idx, y_scaler in enumerate(y_scalers)
    ]
    return X_clusters, y_clusters, y_scalers


def run_piecewise_models(
    X_train, y_train, X_test, y_test, lat_lon_idx=[2, 3], n_clusters=4
):
    gmm = gmm_clustering(X_train, lat_lon_idx, n_clusters)
    train_cluster_labels = gmm.predict(X_train)
    test_cluster_labels = gmm.predict(X_test)

    logger.info("Preparing piecewise train data")
    X_train_clusters, y_train_clusters, y_scalers = prepare_clustered_data(
        X_train, y_train, train_cluster_labels, n_clusters
    )
    logger.info("Preparing piecewise test data")
    X_test_clusters, y_test_clusters, y_scalers = prepare_clustered_data(
        X_test, y_test, test_cluster_labels, n_clusters, y_scalers
    )

    logger.info("Calculating piecewise posteriors...")
    posteriors = [
        define_lin_reg(X_cluster, y_cluster, f"piece_{idx}")
        for idx, (X_cluster, y_cluster) in enumerate(
            zip(X_train_clusters, y_train_clusters)
        )
    ]
    # Run evaluations
    for idx in enumerate(range(n_clusters)):
        logger.info(f"Cluster{idx}, size: {len(y_train_clusters[idx])}")
        evaluate(
            posteriors[idx],
            X_train_clusters[idx],
            y_train_clusters[idx],
            y_scalers[idx],
            f"piece_{idx}",
            "train",
        )
        evaluate(
            posteriors[idx],
            X_test_clusters[idx],
            y_test_clusters[idx],
            y_scalers[idx],
            f"piece_{idx}",
            "test",
        )


def main():
    """Main function to run"""
    train_set, test_set = load_datasets()
    X_train, y_train = prepare_train_set(train_set)
    X_test, y_test = prepare_test_set(test_set)

    X_train_scaled, y_train_scaled, X_transforms, y_scaler = feature_engineering(
        X_train, y_train
    )
    X_test_scaled, y_test_scaled = transform_test_set(
        X_test, y_test, X_transforms, y_scaler
    )
    X_train_scaled, X_test_scaled = feature_sampling(X_train_scaled, X_test_scaled)

    run_full_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler)
    run_piecewise_models(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    mlflow.end_run()

if __name__ == "__main__":
    main()
