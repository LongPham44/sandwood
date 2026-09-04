import sys
import numpy as np
import pymc as pm

from inference.stan.EightSchoolsNonCentered.utility import construct_stan_input_data


def construct_pymc_input_data_standard(num_schools_retained=None):
    dict_stan_data = construct_stan_input_data(num_schools_retained)
    return {
        "J": dict_stan_data["J"],
        "y": np.asarray(dict_stan_data["y"], dtype=np.float64),
        "sigma": np.asarray(dict_stan_data["sigma"], dtype=np.float64),
    }


def construct_pymc_model_standard(num_schools_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_schools_retained)
    y = dict_pymc_data["y"]
    sigma = dict_pymc_data["sigma"]

    with pm.Model() as model:
        theta_trans = pm.Normal(
            "theta_trans", mu=0, sigma=1, shape=dict_pymc_data["J"]
        )
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)
        theta = pm.Deterministic("theta", theta_trans * tau + mu)

        for j in range(dict_pymc_data["J"]):
            pm.Normal("y_{}".format(j), mu=theta[j], sigma=sigma[j], observed=y[j])

    return model, dict_pymc_data


def construct_pymc_model_vectorized(num_schools_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_schools_retained)

    with pm.Model() as model:
        theta_trans = pm.Normal(
            "theta_trans", mu=0, sigma=1, shape=dict_pymc_data["J"]
        )
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)
        theta = pm.Deterministic("theta", theta_trans * tau + mu)

        pm.Normal(
            "y",
            mu=theta,
            sigma=dict_pymc_data["sigma"],
            observed=dict_pymc_data["y"],
        )

    return model, dict_pymc_data


def print_model_sanity_check_vectorized(num_schools_retained=None):
    model, dict_pymc_data = construct_pymc_model_vectorized(num_schools_retained)
    print("Constructed vectorized PyMC model")
    print("  J: {}".format(dict_pymc_data["J"]))
    print("  y shape: {}".format(dict_pymc_data["y"].shape))
    print("  sigma shape: {}".format(dict_pymc_data["sigma"].shape))
    print("  theta_trans shape: ({},)".format(dict_pymc_data["J"]))
    print("  theta shape: ({},)".format(dict_pymc_data["J"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


def print_model_sanity_check(num_schools_retained=None):
    model, dict_pymc_data = construct_pymc_model_standard(num_schools_retained)
    print("Constructed PyMC model")
    print("  J: {}".format(dict_pymc_data["J"]))
    print("  theta_trans shape: ({},)".format(dict_pymc_data["J"]))
    print("  theta shape: ({},)".format(dict_pymc_data["J"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]
    if mode == "sanity_check":
        num_schools_retained = None
        if len(args) >= 2:
            num_schools_retained = int(args[1])
        print_model_sanity_check(num_schools_retained)
    elif mode == "sanity_check_vectorized":
        num_schools_retained = None
        if len(args) >= 2:
            num_schools_retained = int(args[1])
        print_model_sanity_check_vectorized(num_schools_retained)
    else:
        raise ValueError("Mode {} is not supported".format(mode))
