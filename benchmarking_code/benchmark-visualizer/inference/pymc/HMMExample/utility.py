import sys
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from inference.stan.HMMExample.utility import construct_stan_input_data


def construct_pymc_input_data_standard(num_observations_retained=None):
    dict_stan_data = construct_stan_input_data(num_observations_retained)
    return {
        "N": dict_stan_data["N"],
        "K": dict_stan_data["K"],
        "y": np.asarray(dict_stan_data["y"], dtype=np.float64),
    }


def normal_logp(value, mu, sigma):
    return pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), value)


def hmm_forward_logp(y, theta, mu, K, N):
    gamma = []
    for k in range(K):
        gamma.append(normal_logp(y[0], mu[k], 1))

    for t in range(1, N):
        gamma_next = []
        for k in range(K):
            acc = []
            for j in range(K):
                acc.append(gamma[j] + pt.log(theta[j][k]) + normal_logp(y[t], mu[k], 1))
            gamma_next.append(pt.logsumexp(pt.stack(acc)))
        gamma = gamma_next

    return pt.logsumexp(pt.stack(gamma))


def hmm_forward_logp_vectorized(y, theta, mu, N):
    gamma = normal_logp(y[0], mu, 1)
    log_theta = pt.log(theta)

    for t in range(1, N):
        emission_logp = normal_logp(y[t], mu, 1)
        gamma = pt.logsumexp(gamma[:, None] + log_theta, axis=0) + emission_logp

    return pt.logsumexp(gamma)


def construct_positive_ordered_mu(K):
    if K != 2:
        raise ValueError("HMMExample currently expects K = 2, matching the Stan model")

    mu_first = pm.HalfFlat("mu_first")
    mu_gap = pm.HalfFlat("mu_gap")
    mu = pm.Deterministic("mu", pt.stack([mu_first, mu_first + mu_gap]))
    return mu


def construct_pymc_model_standard(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)
    K = dict_pymc_data["K"]
    N = dict_pymc_data["N"]
    y = dict_pymc_data["y"]

    with pm.Model() as model:
        theta1 = pm.Dirichlet("theta1", a=np.ones(K))
        theta2 = pm.Dirichlet("theta2", a=np.ones(K))
        mu = construct_positive_ordered_mu(K)

        theta = [theta1, theta2]
        pm.Potential("mu_1_prior", normal_logp(mu[0], 3, 1))
        pm.Potential("mu_2_prior", normal_logp(mu[1], 10, 1))
        pm.Potential("forward_logp", hmm_forward_logp(y, theta, mu, K, N))

    return model, dict_pymc_data


def construct_pymc_model_vectorized(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)
    K = dict_pymc_data["K"]
    N = dict_pymc_data["N"]
    y = dict_pymc_data["y"]

    with pm.Model() as model:
        theta1 = pm.Dirichlet("theta1", a=np.ones(K))
        theta2 = pm.Dirichlet("theta2", a=np.ones(K))
        mu = construct_positive_ordered_mu(K)

        theta = pt.stack([theta1, theta2])
        pm.Potential("mu_1_prior", normal_logp(mu[0], 3, 1))
        pm.Potential("mu_2_prior", normal_logp(mu[1], 10, 1))
        pm.Potential("forward_logp", hmm_forward_logp_vectorized(y, theta, mu, N))

    return model, dict_pymc_data


def print_model_sanity_check_vectorized(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_vectorized(num_observations_retained)
    print("Constructed vectorized PyMC model")
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  K: {}".format(dict_pymc_data["K"]))
    print("  theta1 shape: ({},)".format(dict_pymc_data["K"]))
    print("  theta2 shape: ({},)".format(dict_pymc_data["K"]))
    print("  mu shape: ({},)".format(dict_pymc_data["K"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


def print_model_sanity_check(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_standard(num_observations_retained)
    print("Constructed PyMC model")
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  K: {}".format(dict_pymc_data["K"]))
    print("  theta1 shape: ({},)".format(dict_pymc_data["K"]))
    print("  theta2 shape: ({},)".format(dict_pymc_data["K"]))
    print("  mu shape: ({},)".format(dict_pymc_data["K"]))
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
