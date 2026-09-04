from input_output.file_manipulation import read_observed_data


def construct_stan_input_data(num_observations_retained=None):
    benchmark_name = "SBLRC"
    dict_observed_data = read_observed_data(benchmark_name)

    for required_key in ["X", "y"]:
        if required_key not in dict_observed_data:
            raise ValueError(
                "Observed data for {} is missing required key {}".format(
                    benchmark_name, required_key
                )
            )

    X = [list(row) for row in dict_observed_data["X"]]
    y = list(dict_observed_data["y"])
    if num_observations_retained is not None:
        N = int(num_observations_retained)
        X = X[:N]
        y = y[:N]

    N = len(y)
    if len(X) != N:
        raise ValueError("SBLRC X and y must have the same number of rows")

    D = len(X[0]) if N > 0 else 0
    for row in X:
        if len(row) != D:
            raise ValueError("SBLRC X rows must have the same length")

    return {"N": N, "D": D, "X": X, "y": y}
