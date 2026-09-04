from input_output.file_manipulation import read_observed_data


def construct_stan_input_data(num_schools_retained=None):
    benchmark_name = "EightSchoolsNonCentered"
    observed_data_name = "EightSchools"
    dict_observed_data = read_observed_data(observed_data_name)

    for required_key in ["y", "sigma"]:
        if required_key not in dict_observed_data:
            raise ValueError(
                "Observed data for {} is missing required key {}".format(
                    benchmark_name, required_key
                )
            )

    y = list(dict_observed_data["y"])
    sigma = list(dict_observed_data["sigma"])
    if num_schools_retained is not None:
        J = int(num_schools_retained)
        y = y[:J]
        sigma = sigma[:J]

    J = len(y)
    if len(sigma) != J:
        raise ValueError("EightSchoolsNonCentered y and sigma must have the same length")

    return {"J": J, "y": y, "sigma": sigma}
