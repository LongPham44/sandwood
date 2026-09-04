from input_output.file_manipulation import read_observed_data


def construct_stan_input_data(num_observations_retained=None):
    benchmark_name = "LDAK5"
    dict_observed_data = read_observed_data(benchmark_name)

    for required_key in ["V", "M", "N", "w", "doc", "alpha", "beta"]:
        if required_key not in dict_observed_data:
            raise ValueError(
                "Observed data for {} is missing required key {}".format(
                    benchmark_name, required_key
                )
            )

    w = list(dict_observed_data["w"])
    doc = list(dict_observed_data["doc"])
    if num_observations_retained is not None:
        N = int(num_observations_retained)
        w = w[:N]
        doc = doc[:N]

    N = len(w)
    if len(doc) != N:
        raise ValueError("LDAK5 observed variables w and doc must have the same length")

    return {
        "V": int(dict_observed_data["V"]),
        "M": int(dict_observed_data["M"]),
        "N": N,
        "w": w,
        "doc": doc,
        "alpha": list(dict_observed_data["alpha"]),
        "beta": list(dict_observed_data["beta"]),
    }
