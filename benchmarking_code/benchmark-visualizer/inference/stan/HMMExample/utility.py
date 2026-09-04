from input_output.file_manipulation import read_observed_data


def construct_stan_input_data(num_observations_retained=None):
    benchmark_name = "HMMExample"
    dict_observed_data = read_observed_data(benchmark_name)

    for required_key in ["K", "y"]:
        if required_key not in dict_observed_data:
            raise ValueError(
                "Observed data for {} is missing required key {}".format(
                    benchmark_name, required_key
                )
            )

    y = list(dict_observed_data["y"])
    if num_observations_retained is not None:
        y = y[: int(num_observations_retained)]

    return {
        "N": len(y),
        "K": int(dict_observed_data["K"]),
        "y": y,
    }
