from input_output.file_manipulation import read_observed_data


def construct_stan_input_data(num_time_steps_retained=None):
    benchmark_name = "ARMA"
    dict_observed_data = read_observed_data(benchmark_name)

    for required_key in ["y"]:
        if required_key not in dict_observed_data:
            raise ValueError(
                "Observed data for {} is missing required key {}".format(
                    benchmark_name, required_key
                )
            )

    y = list(dict_observed_data["y"])
    if num_time_steps_retained is not None:
        y = y[: int(num_time_steps_retained)]

    if len(y) < 1:
        raise ValueError("The ARMA Stan model requires at least one observation")

    return {
        "T": len(y),
        "y": y,
    }

