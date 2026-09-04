from input_output.file_manipulation import read_observed_data


def construct_stan_input_data(num_time_steps_retained=None):
    # Dictionary storing input data (e.g., hyperparameters and observed data)
    # for Stan.
    benchmark_name = "ARK"
    dict_observed_data = read_observed_data(benchmark_name)

    for required_key in ["K", "y"]:
        if required_key not in dict_observed_data:
            raise ValueError(
                "Observed data for {} is missing required key {}".format(
                    benchmark_name, required_key
                )
            )

    K = int(dict_observed_data["K"])
    y = list(dict_observed_data["y"])

    if num_time_steps_retained is not None:
        T = int(num_time_steps_retained)
        y = y[:T]
    T = len(y)

    if T <= K:
        raise ValueError(
            "The ARK Stan model requires T > K, but received T = {:d} and K = {:d}".format(
                T, K
            )
        )

    dict_stan_data = {
        "K": K,
        "T": T,
        "y": y,
    }

    return dict_stan_data
