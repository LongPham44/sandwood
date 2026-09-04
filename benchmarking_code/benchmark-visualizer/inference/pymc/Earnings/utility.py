import sys
import numpy as np
import pymc as pm

from inference.stan.Earnings.utility import construct_stan_input_data


def construct_pymc_input_data_standard(num_observations_retained=None):
    dict_stan_data = construct_stan_input_data(num_observations_retained)
    return {
        "N": dict_stan_data["N"],
        "earn": np.asarray(dict_stan_data["earn"], dtype=np.float64),
        "height": np.asarray(dict_stan_data["height"], dtype=np.float64),
        "male": np.asarray(dict_stan_data["male"], dtype=np.float64),
    }


def construct_pymc_model_standard(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)

    earn = dict_pymc_data["earn"]
    height = dict_pymc_data["height"]
    male = dict_pymc_data["male"]
    log_earn = np.log(earn)
    inter = height * male

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=4)
        sigma = pm.HalfCauchy("sigma", beta=10)

        for n in range(dict_pymc_data["N"]):
            mu = beta[0] + beta[1] * height[n] + beta[2] * male[n] + beta[3] * inter[n]
            pm.Normal("log_earn_{}".format(n), mu=mu, sigma=sigma, observed=log_earn[n])

    return model, dict_pymc_data


def construct_pymc_input_data_vectorized(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)

    height = dict_pymc_data["height"]
    male = dict_pymc_data["male"]
    dict_pymc_data["log_earn"] = np.log(dict_pymc_data["earn"])
    dict_pymc_data["inter"] = height * male

    return dict_pymc_data


def construct_pymc_model_vectorized(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_vectorized(num_observations_retained)

    height = dict_pymc_data["height"]
    male = dict_pymc_data["male"]
    inter = dict_pymc_data["inter"]

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=4)
        sigma = pm.HalfCauchy("sigma", beta=10)

        mu = beta[0] + beta[1] * height + beta[2] * male + beta[3] * inter
        pm.Normal(
            "log_earn",
            mu=mu,
            sigma=sigma,
            observed=dict_pymc_data["log_earn"],
        )

    return model, dict_pymc_data


def print_model_sanity_check_vectorized(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_vectorized(num_observations_retained)
    print("Constructed vectorized PyMC model")
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  log_earn shape: {}".format(dict_pymc_data["log_earn"].shape))
    print("  inter shape: {}".format(dict_pymc_data["inter"].shape))
    print("  beta shape: (4,)")
    print("  model variables: {}".format(list(model.named_vars.keys())))


def print_model_sanity_check(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_standard(num_observations_retained)
    print("Constructed PyMC model")
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  beta shape: (4,)")
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
