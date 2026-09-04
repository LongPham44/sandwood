import sys

from inference.stan.SequentialTrajectories.utility import construct_stan_input_data
from inference.stan.log_probability import (
    calculate_and_save_log_probabilities as calculate_and_save_log_probabilities_common,
    calculate_log_probability as calculate_log_probability_common,
    main,
)

BENCHMARK_NAME = "SequentialTrajectories"
PARAMETER_NAMES = ["gamma", "a", "b", "c", "sigma2", "qualities"]


def calculate_log_probability(dict_parameters, benchmark_name=BENCHMARK_NAME):
    return calculate_log_probability_common(
        dict_parameters,
        benchmark_name,
        construct_stan_input_data,
        PARAMETER_NAMES,
        infer_retained_dimension=True,
    )


def calculate_and_save_log_probabilities(benchmark_name, language=None):
    if language is None:
        language = benchmark_name
        benchmark_name = BENCHMARK_NAME

    return calculate_and_save_log_probabilities_common(
        benchmark_name,
        language,
        construct_stan_input_data,
        PARAMETER_NAMES,
        infer_retained_dimension=True,
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "log_probabilities":
        _, max_log_prob = calculate_and_save_log_probabilities(args[1], args[2])
        print(
            "Log probabilities calculated for {} using {}".format(
                args[1], args[2]
            )
        )
        print("MAP estimate's density: {:.7f}".format(max_log_prob))
    else:
        main(
            BENCHMARK_NAME,
            construct_stan_input_data,
            PARAMETER_NAMES,
            infer_retained_dimension=True,
        )
