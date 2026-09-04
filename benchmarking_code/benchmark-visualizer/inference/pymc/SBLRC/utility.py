import sys
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from inference.stan.SBLRC.utility import construct_stan_input_data


def construct_pymc_input_data_standard(num_observations_retained=None):
    dict_stan_data = construct_stan_input_data(num_observations_retained)
    return {
        "N": dict_stan_data["N"],
        "D": dict_stan_data["D"],
        "X": np.asarray(dict_stan_data["X"], dtype=np.float64),
        "y": np.asarray(dict_stan_data["y"], dtype=np.float64),
    }


def construct_pymc_model_standard(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)
    X = dict_pymc_data["X"]
    y = dict_pymc_data["y"]

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=dict_pymc_data["D"])
        sigma = pm.HalfNormal("sigma", sigma=10)

        for n in range(dict_pymc_data["N"]):
            mu = 0.0
            for d in range(dict_pymc_data["D"]):
                mu += X[n, d] * beta[d]
            pm.Normal("y_{}".format(n), mu=mu, sigma=sigma, observed=y[n])

    return model, dict_pymc_data


def construct_pymc_model_vectorized(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=dict_pymc_data["D"])
        sigma = pm.HalfNormal("sigma", sigma=10)

        X = pt.as_tensor_variable(dict_pymc_data["X"])
        mu = pt.dot(X, beta)
        pm.Normal("y", mu=mu, sigma=sigma, observed=dict_pymc_data["y"])

    return model, dict_pymc_data


def print_model_sanity_check_vectorized(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_vectorized(num_observations_retained)
    print("Constructed vectorized PyMC model")
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  D: {}".format(dict_pymc_data["D"]))
    print("  X shape: {}".format(dict_pymc_data["X"].shape))
    print("  y shape: {}".format(dict_pymc_data["y"].shape))
    print("  beta shape: ({},)".format(dict_pymc_data["D"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


def print_model_sanity_check(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_standard(num_observations_retained)
    print("Constructed PyMC model")
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  D: {}".format(dict_pymc_data["D"]))
    print("  X shape: {}".format(dict_pymc_data["X"].shape))
    print("  beta shape: ({},)".format(dict_pymc_data["D"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]
    if mode == "sanity_check":
        num_observations_retained = None
        if len(args) >= 2:
            num_observations_retained = int(args[1])
        print_model_sanity_check(num_observations_retained)
    elif mode == "sanity_check_vectorized":
        num_observations_retained = None
        if len(args) >= 2:
            num_observations_retained = int(args[1])
        print_model_sanity_check_vectorized(num_observations_retained)
    else:
        raise ValueError("Mode {} is not supported".format(mode))
