import sys

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from inference.stan.LDAK5.utility import construct_stan_input_data

NUM_TOPICS = 5


def construct_pymc_input_data_standard(num_observations_retained=None):
    dict_stan_data = construct_stan_input_data(num_observations_retained)
    return {
        "V": dict_stan_data["V"],
        "M": dict_stan_data["M"],
        "N": dict_stan_data["N"],
        "w": np.asarray(dict_stan_data["w"], dtype=np.int64) - 1,
        "doc": np.asarray(dict_stan_data["doc"], dtype=np.int64) - 1,
        "alpha": np.asarray(dict_stan_data["alpha"], dtype=np.float64),
        "beta": np.asarray(dict_stan_data["beta"], dtype=np.float64),
    }


def lda_logp(w, doc, theta, phi, N):
    terms = []
    for n in range(N):
        gamma = []
        for k in range(NUM_TOPICS):
            gamma.append(pt.log(theta[doc[n], k]) + pt.log(phi[k, w[n]]))
        terms.append(pt.logsumexp(pt.stack(gamma)))

    return pt.sum(pt.stack(terms))


def lda_logp_vectorized(w, doc, theta, phi):
    word_topic_probabilities = phi[:, w].T
    doc_topic_probabilities = theta[doc]
    gamma = pt.log(doc_topic_probabilities) + pt.log(word_topic_probabilities)
    return pt.sum(pt.logsumexp(gamma, axis=1))


def construct_pymc_model_standard(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)
    M = dict_pymc_data["M"]
    V = dict_pymc_data["V"]
    theta_alpha = np.tile(dict_pymc_data["alpha"], (M, 1))
    phi_beta = np.tile(dict_pymc_data["beta"], (NUM_TOPICS, 1))

    with pm.Model() as model:
        theta = pm.Dirichlet(
            "theta",
            a=theta_alpha,
            shape=(M, NUM_TOPICS),
        )
        phi = pm.Dirichlet(
            "phi",
            a=phi_beta,
            shape=(NUM_TOPICS, V),
        )
        pm.Potential(
            "lda_logp",
            lda_logp(
                dict_pymc_data["w"],
                dict_pymc_data["doc"],
                theta,
                phi,
                dict_pymc_data["N"],
            ),
        )

    return model, dict_pymc_data


def construct_pymc_model_vectorized(num_observations_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_observations_retained)
    M = dict_pymc_data["M"]
    V = dict_pymc_data["V"]
    theta_alpha = np.tile(dict_pymc_data["alpha"], (M, 1))
    phi_beta = np.tile(dict_pymc_data["beta"], (NUM_TOPICS, 1))

    with pm.Model() as model:
        theta = pm.Dirichlet(
            "theta",
            a=theta_alpha,
            shape=(M, NUM_TOPICS),
        )
        phi = pm.Dirichlet(
            "phi",
            a=phi_beta,
            shape=(NUM_TOPICS, V),
        )
        pm.Potential(
            "lda_logp",
            lda_logp_vectorized(
                dict_pymc_data["w"],
                dict_pymc_data["doc"],
                theta,
                phi,
            ),
        )

    return model, dict_pymc_data


def print_model_sanity_check_vectorized(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_vectorized(num_observations_retained)
    print("Constructed vectorized PyMC model")
    print("  V: {}".format(dict_pymc_data["V"]))
    print("  M: {}".format(dict_pymc_data["M"]))
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  theta shape: ({}, {})".format(dict_pymc_data["M"], NUM_TOPICS))
    print("  phi shape: ({}, {})".format(NUM_TOPICS, dict_pymc_data["V"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


def print_model_sanity_check(num_observations_retained=None):
    model, dict_pymc_data = construct_pymc_model_standard(num_observations_retained)
    print("Constructed PyMC model")
    print("  V: {}".format(dict_pymc_data["V"]))
    print("  M: {}".format(dict_pymc_data["M"]))
    print("  N: {}".format(dict_pymc_data["N"]))
    print("  theta shape: ({}, {})".format(dict_pymc_data["M"], NUM_TOPICS))
    print("  phi shape: ({}, {})".format(NUM_TOPICS, dict_pymc_data["V"]))
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
