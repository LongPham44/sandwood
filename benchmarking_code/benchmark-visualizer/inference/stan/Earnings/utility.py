from input_output.file_manipulation import read_observed_data


def construct_stan_input_data(num_observations_retained=None):
    benchmark_name = "Earnings"
    dict_observed_data = read_observed_data(benchmark_name)

    for required_key in ["earn", "height", "male"]:
        if required_key not in dict_observed_data:
            raise ValueError(
                "Observed data for {} is missing required key {}".format(
                    benchmark_name, required_key
                )
            )

    earn = list(dict_observed_data["earn"])
    height = list(dict_observed_data["height"])
    male = list(dict_observed_data["male"])

    if num_observations_retained is not None:
        N = int(num_observations_retained)
        earn = earn[:N]
        height = height[:N]
        male = male[:N]

    N = len(earn)
    if len(height) != N or len(male) != N:
        raise ValueError("Earnings observed variables must have the same length")

    return {
        "N": N,
        "earn": earn,
        "height": height,
        "male": male,
    }
