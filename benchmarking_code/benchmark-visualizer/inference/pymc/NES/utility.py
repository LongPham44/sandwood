import sys
import numpy as np
import pymc as pm

from inference.stan.NES.utility import construct_stan_input_data


def construct_pymc_input_data_standard(num_observations_retained=None):
    dict_stan_data = construct_stan_input_data(num_observations_retained)
    dict_pymc_data = {"N": dict_stan_data["N"]}
    for key in [
        "partyid7",
        "real_ideo",
        "race_adj",
        "educ1",
        "gender",
        "income",
    ]:
        dict_pymc_data[key] = np.asarray(dict_stan_data[key], dtype=np.float64)
    dict_pymc_data["age_discrete"] = np.asarray(
        dict_stan_data["age_discrete"], dtype=np.int64
    )
    return dict_pymc_data


def construct_pymc_model_standard(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)

    partyid7 = dict_pymc_data["partyid7"]
    real_ideo = dict_pymc_data["real_ideo"]
    race_adj = dict_pymc_data["race_adj"]
    educ1 = dict_pymc_data["educ1"]
    gender = dict_pymc_data["gender"]
    income = dict_pymc_data["income"]
    age_discrete = dict_pymc_data["age_discrete"]
    age30_44 = (age_discrete == 2).astype(np.float64)
    age45_64 = (age_discrete == 3).astype(np.float64)
    age65up = (age_discrete == 4).astype(np.float64)

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=9)
        sigma = pm.HalfCauchy("sigma", beta=10)

        for n in range(dict_pymc_data["N"]):
            mu = (
                beta[0]
                + beta[1] * real_ideo[n]
                + beta[2] * race_adj[n]
                + beta[3] * age30_44[n]
                + beta[4] * age45_64[n]
                + beta[5] * age65up[n]
                + beta[6] * educ1[n]
                + beta[7] * gender[n]
                + beta[8] * income[n]
            )
            pm.Normal("partyid7_{}".format(n), mu=mu, sigma=sigma, observed=partyid7[n])

    return model, dict_pymc_data


def construct_pymc_model_vectorized(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)

    age_discrete = dict_pymc_data["age_discrete"]
    age30_44 = (age_discrete == 2).astype(np.float64)
    age45_64 = (age_discrete == 3).astype(np.float64)
    age65up = (age_discrete == 4).astype(np.float64)

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=9)
        sigma = pm.HalfCauchy("sigma", beta=10)

        mu = (
            beta[0]
            + beta[1] * dict_pymc_data["real_ideo"]
            + beta[2] * dict_pymc_data["race_adj"]
            + beta[3] * age30_44
            + beta[4] * age45_64
            + beta[5] * age65up
            + beta[6] * dict_pymc_data["educ1"]
            + beta[7] * dict_pymc_data["gender"]
            + beta[8] * dict_pymc_data["income"]
        )
        pm.Normal(
            "partyid7",
            mu=mu,
            sigma=sigma,
            observed=dict_pymc_data["partyid7"],
        )

    return model, dict_pymc_data


def print_model_sanity_check_vectorized(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_vectorized(num_observations_retained)
    print("Constructed vectorized PyMC model")
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  partyid7 shape: {}".format(dict_pymc_data["partyid7"].shape))
    print("  beta shape: (9,)")
    print("  model variables: {}".format(list(model.named_vars.keys())))


def print_model_sanity_check(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_standard(num_observations_retained)
    print("Constructed PyMC model")
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  beta shape: (9,)")
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
