import sys
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from inference.stan.ARK.utility import construct_stan_input_data


def construct_pymc_input_data_standard(num_time_steps_retained=None):
    dict_stan_data = construct_stan_input_data(num_time_steps_retained)

    dict_pymc_data = {
        "K": dict_stan_data["K"],
        "T": dict_stan_data["T"],
        "y": np.asarray(dict_stan_data["y"], dtype=np.float64),
    }

    return dict_pymc_data


def construct_pymc_model_standard(num_time_steps_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_time_steps_retained)

    K = dict_pymc_data["K"]
    T = dict_pymc_data["T"]
    y = dict_pymc_data["y"]

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=K)
        sigma = pm.HalfCauchy("sigma", beta=2.5)

        for t in range(K, T):
            mu = alpha
            for k in range(K):
                mu += beta[k] * y[t - k - 1]

            pm.Normal(
                "y_{}".format(t),
                mu=mu,
                sigma=sigma,
                observed=y[t],
            )

    return model, dict_pymc_data


def construct_lagged_observations(y, K):
    list_lagged_observations = []
    for k in range(K):
        lag = k + 1
        list_lagged_observations.append(y[K - lag : -lag])

    return np.stack(list_lagged_observations, axis=1)


def construct_pymc_input_data_vectorized(num_time_steps_retained=None):
    dict_stan_data = construct_stan_input_data(num_time_steps_retained)

    K = dict_stan_data["K"]
    y = np.asarray(dict_stan_data["y"], dtype=np.float64)
    y_observed = y[K:]
    y_lagged_observed = construct_lagged_observations(y, K)

    dict_pymc_data = {
        "K": K,
        "T": dict_stan_data["T"],
        "y": y,
        "y_observed": y_observed,
        "y_lagged_observed": y_lagged_observed,
    }

    return dict_pymc_data


def construct_pymc_model_vectorized(num_time_steps_retained=None):
    dict_pymc_data = construct_pymc_input_data_vectorized(num_time_steps_retained)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=dict_pymc_data["K"])
        sigma = pm.HalfCauchy("sigma", beta=2.5)

        y_lagged_observed = pt.as_tensor_variable(dict_pymc_data["y_lagged_observed"])
        mu = alpha + pt.sum(beta * y_lagged_observed, axis=1)

        pm.Normal(
            "y",
            mu=mu,
            sigma=sigma,
            observed=dict_pymc_data["y_observed"],
        )

    return model, dict_pymc_data


def print_model_sanity_check(num_time_steps_retained=None):
    model, dict_pymc_data = construct_pymc_model_vectorized(num_time_steps_retained)

    print("Constructed PyMC model")
    print("  K: {}".format(dict_pymc_data["K"]))
    print("  T: {}".format(dict_pymc_data["T"]))
    print("  observed likelihood terms: {}".format(dict_pymc_data["T"] - dict_pymc_data["K"]))
    print("  y shape: {}".format(dict_pymc_data["y"].shape))
    print("  y_observed shape: {}".format(dict_pymc_data["y_observed"].shape))
    print("  y_lagged_observed shape: {}".format(dict_pymc_data["y_lagged_observed"].shape))
    print("  beta shape: ({},)".format(dict_pymc_data["K"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]

    if mode == "sanity_check":
        num_time_steps_retained = None
        if len(args) >= 2:
            num_time_steps_retained = int(args[1])

        print_model_sanity_check(num_time_steps_retained)
    else:
        raise ValueError("Mode {} is not supported".format(mode))
