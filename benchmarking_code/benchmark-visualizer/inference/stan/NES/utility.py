from input_output.file_manipulation import read_observed_data


def construct_stan_input_data(num_observations_retained=None):
    benchmark_name = "NES"
    dict_observed_data = read_observed_data(benchmark_name)
    required_keys = [
        "partyid7",
        "real_ideo",
        "race_adj",
        "educ1",
        "gender",
        "income",
        "age_discrete",
    ]
    for required_key in required_keys:
        if required_key not in dict_observed_data:
            raise ValueError(
                "Observed data for {} is missing required key {}".format(
                    benchmark_name, required_key
                )
            )

    dict_stan_data = {}
    N = None
    for key in required_keys:
        values = list(dict_observed_data[key])
        if num_observations_retained is not None:
            values = values[: int(num_observations_retained)]
        if N is None:
            N = len(values)
        elif len(values) != N:
            raise ValueError("NES observed variables must have the same length")
        dict_stan_data[key] = values

    dict_stan_data["N"] = N
    return dict_stan_data
