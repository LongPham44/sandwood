import sys
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from inference.stan.ARMA.utility import construct_stan_input_data


def construct_pymc_input_data_standard(num_time_steps_retained=None):
    dict_stan_data = construct_stan_input_data(num_time_steps_retained)
    return {
        "T": dict_stan_data["T"],
        "y": np.asarray(dict_stan_data["y"], dtype=np.float64),
    }


def construct_pymc_model_standard(num_time_steps_retained=None):
    dict_pymc_data = construct_pymc_input_data_standard(num_time_steps_retained)
    T = dict_pymc_data["T"]
    y = dict_pymc_data["y"]

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        phi = pm.Normal("phi", mu=0, sigma=2)
        theta = pm.Normal("theta", mu=0, sigma=2)
        sigma = pm.HalfCauchy("sigma", beta=2.5)

        nu = mu + phi * mu
        err = y[0] - nu
        pm.Normal("y_0", mu=nu, sigma=sigma, observed=y[0])

        for t in range(1, T):
            nu = mu + phi * y[t - 1] + theta * err
            err = y[t] - nu
            pm.Normal("y_{}".format(t), mu=nu, sigma=sigma, observed=y[t])

    return model, dict_pymc_data


def construct_pymc_input_data_vectorized(num_time_steps_retained=None):
    return construct_pymc_input_data_standard(num_time_steps_retained)


def construct_conditional_mean_vector(y, mu, phi, theta):
    y_tensor = pt.as_tensor_variable(y)

    nu_0 = mu + phi * mu
    err_0 = y_tensor[0] - nu_0

    if y.shape[0] == 1:
        return nu_0[None]

    def step(y_t, y_lagged_t, err_previous, mu, phi, theta):
        nu_t = mu + phi * y_lagged_t + theta * err_previous
        err_t = y_t - nu_t
        return err_t, nu_t

    (_, nu_tail), _ = pytensor.scan(
        fn=step,
        sequences=[y_tensor[1:], y_tensor[:-1]],
        outputs_info=[err_0, None],
        non_sequences=[mu, phi, theta],
        strict=True,
    )

    return pt.concatenate([nu_0[None], nu_tail])


def construct_pymc_model_vectorized(num_time_steps_retained=None):
    dict_pymc_data = construct_pymc_input_data_vectorized(num_time_steps_retained)
    y = dict_pymc_data["y"]

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        phi = pm.Normal("phi", mu=0, sigma=2)
        theta = pm.Normal("theta", mu=0, sigma=2)
        sigma = pm.HalfCauchy("sigma", beta=2.5)

        nu = construct_conditional_mean_vector(y, mu, phi, theta)
        pm.Normal("y", mu=nu, sigma=sigma, observed=y)

    return model, dict_pymc_data


def print_model_sanity_check(num_time_steps_retained=None):
    model, dict_pymc_data = construct_pymc_model_standard(num_time_steps_retained)
    print("Constructed PyMC model")
    print("  T: {}".format(dict_pymc_data["T"]))
    print("  y shape: {}".format(dict_pymc_data["y"].shape))
    print("  observed likelihood terms: {}".format(dict_pymc_data["T"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


def print_model_sanity_check_vectorized(num_time_steps_retained=None):
    model, dict_pymc_data = construct_pymc_model_vectorized(num_time_steps_retained)
    print("Constructed vectorized PyMC model")
    print("  T: {}".format(dict_pymc_data["T"]))
    print("  y shape: {}".format(dict_pymc_data["y"].shape))
    print("  observed likelihood terms: {}".format(dict_pymc_data["T"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]
    if mode == "sanity_check":
        num_time_steps_retained = None
        if len(args) >= 2:
            num_time_steps_retained = int(args[1])
        print_model_sanity_check(num_time_steps_retained)
    elif mode == "sanity_check_vectorized":
        num_time_steps_retained = None
        if len(args) >= 2:
            num_time_steps_retained = int(args[1])
        print_model_sanity_check_vectorized(num_time_steps_retained)
    else:
        raise ValueError("Mode {} is not supported".format(mode))
