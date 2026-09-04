from inference.stan.Earnings.utility import construct_stan_input_data
from inference.stan.log_probability import (
    calculate_and_save_log_probabilities as calculate_and_save_log_probabilities_common,
    calculate_log_probability as calculate_log_probability_common,
    main,
)

BENCHMARK_NAME = "Earnings"
PARAMETER_NAMES = ["beta", "sigma"]


def calculate_log_probability(dict_parameters, retained_dimension=None):
    return calculate_log_probability_common(
        dict_parameters,
        BENCHMARK_NAME,
        construct_stan_input_data,
        PARAMETER_NAMES,
        retained_dimension,
    )


def calculate_and_save_log_probabilities(language):
    return calculate_and_save_log_probabilities_common(
        BENCHMARK_NAME,
        language,
        construct_stan_input_data,
        PARAMETER_NAMES,
    )


if __name__ == "__main__":
    main(BENCHMARK_NAME, construct_stan_input_data, PARAMETER_NAMES)
