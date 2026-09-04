import sys

from input_output.file_manipulation import (
    CHAIN_IDS_KEY,
    EXECUTION_TARGET_KEY,
    NUM_CHAINS_KEY,
    SAMPLING_CONFIGURATION_KEY,
    read_posterior_samples,
    read_sandwood_posterior_samples_benchmark_list_chain_ids,
    write_log_probabilities,
)
from inference.stan.ARK.utility import construct_stan_input_data as construct_ark_data
from inference.stan.ARMA.utility import construct_stan_input_data as construct_arma_data
from inference.stan.Earnings.utility import (
    construct_stan_input_data as construct_earnings_data,
)
from inference.stan.EightSchoolsCentered.utility import (
    construct_stan_input_data as construct_eight_schools_centered_data,
)
from inference.stan.EightSchoolsNonCentered.utility import (
    construct_stan_input_data as construct_eight_schools_non_centered_data,
)
from inference.stan.HMMExample.utility import (
    construct_stan_input_data as construct_hmm_example_data,
)
from inference.stan.LDAK5.utility import construct_stan_input_data as construct_ldak5_data
from inference.stan.NES.utility import construct_stan_input_data as construct_nes_data
from inference.stan.SBLRC.utility import construct_stan_input_data as construct_sblrc_data
from inference.stan.SequentialTrajectories.utility import (
    construct_stan_input_data as construct_sequential_trajectories_data,
)
from inference.stan.utility import construct_stan_model

SANDWOOD_EXECUTION_TARGET = "SingleThread"
SANDWOOD_LIST_CHAIN_IDS = list(range(42, 52))
SUPPORTED_LANGUAGES = ["sandwood", "stan", "pymc"]
BENCHMARK_SPECS = [
    # {
    #     "benchmark_name": "ARK",
    #     "construct_stan_input_data": construct_ark_data,
    #     "parameter_names": ["alpha", "beta", "sigma"],
    # },
    {
        "benchmark_name": "ARMA",
        "construct_stan_input_data": construct_arma_data,
        "parameter_names": ["mu", "phi", "theta", "sigma"],
    },
    # {
    #     "benchmark_name": "SequentialTrajectories",
    #     "construct_stan_input_data": construct_sequential_trajectories_data,
    #     "parameter_names": ["gamma", "a", "b", "c", "sigma2", "qualities"],
    #     "infer_retained_dimension": True,
    # },
    {
        "benchmark_name": "Earnings",
        "construct_stan_input_data": construct_earnings_data,
        "parameter_names": ["beta", "sigma"],
    },
    {
        "benchmark_name": "EightSchoolsCentered",
        "construct_stan_input_data": construct_eight_schools_centered_data,
        "parameter_names": ["theta", "mu", "tau"],
    },
    {
        "benchmark_name": "EightSchoolsNonCentered",
        "construct_stan_input_data": construct_eight_schools_non_centered_data,
        "parameter_names": ["theta_trans", "mu", "tau"],
    },
    {
        "benchmark_name": "NES",
        "construct_stan_input_data": construct_nes_data,
        "parameter_names": ["beta", "sigma"],
    },
    {
        "benchmark_name": "SBLRC",
        "construct_stan_input_data": construct_sblrc_data,
        "parameter_names": ["beta", "sigma"],
    },
    {
        "benchmark_name": "HMMExample",
        "construct_stan_input_data": construct_hmm_example_data,
        "parameter_names": ["theta1", "theta2", "mu"],
    },
    {
        "benchmark_name": "LDAK5",
        "construct_stan_input_data": construct_ldak5_data,
        "parameter_names": ["theta", "phi"],
        "model_filename": "LDAK5-log-prob.stan",
    },
]


def construct_log_probability_context(
    benchmark_name,
    construct_stan_input_data,
    retained_dimension=None,
    model_filename=None,
):
    if model_filename is None:
        model_filename = "{}.stan".format(benchmark_name)
    model = construct_stan_model(model_filename=model_filename)

    if retained_dimension is None:
        dict_stan_data = construct_stan_input_data()
    else:
        dict_stan_data = construct_stan_input_data(retained_dimension)

    return model, dict_stan_data


def filter_parameters(dict_parameters, parameter_names):
    dict_filtered_parameters = {}
    for parameter_name in parameter_names:
        if parameter_name not in dict_parameters:
            raise ValueError("Posterior samples are missing {}".format(parameter_name))
        dict_filtered_parameters[parameter_name] = dict_parameters[parameter_name]

    return dict_filtered_parameters


def infer_retained_dimension_posterior_samples(
    dict_latent_variables_posterior_samples, parameter_names
):
    list_latent_variable_dimensions = []
    for parameter_name in parameter_names:
        first_posterior_sample = dict_latent_variables_posterior_samples[
            parameter_name
        ][0]
        if isinstance(first_posterior_sample, (list, tuple)):
            list_latent_variable_dimensions.append(len(first_posterior_sample))

    if len(list_latent_variable_dimensions) == 0:
        return None

    return max(list_latent_variable_dimensions)


def infer_retained_dimension_parameters(dict_parameters, parameter_names):
    list_latent_variable_dimensions = []
    for parameter_name in parameter_names:
        parameter_value = dict_parameters[parameter_name]
        if isinstance(parameter_value, (list, tuple)):
            list_latent_variable_dimensions.append(len(parameter_value))

    if len(list_latent_variable_dimensions) == 0:
        return None

    return max(list_latent_variable_dimensions)


def calculate_log_probability_given_context(model, dict_stan_data, dict_parameters):
    # Calculate a log probability. It is important to set the parameter jacobian
    # to False. Otherwise, it calculates a wrong value of the log probability.
    log_prob = model.log_prob(
        params=dict_parameters, data=dict_stan_data, jacobian=False
    )
    return float(log_prob.at[0, "lp__"])


def calculate_log_probability(
    dict_parameters,
    benchmark_name,
    construct_stan_input_data,
    parameter_names,
    retained_dimension=None,
    infer_retained_dimension=False,
    model_filename=None,
):
    dict_filtered_parameters = filter_parameters(dict_parameters, parameter_names)
    if infer_retained_dimension and retained_dimension is None:
        retained_dimension = infer_retained_dimension_parameters(
            dict_filtered_parameters, parameter_names
        )

    model, dict_stan_data = construct_log_probability_context(
        benchmark_name,
        construct_stan_input_data,
        retained_dimension,
        model_filename=model_filename,
    )
    return calculate_log_probability_given_context(
        model, dict_stan_data, dict_filtered_parameters
    )


def read_posterior_samples_language(benchmark_name, language):
    if language == "sandwood":
        (
            dict_latent_variables_posterior_samples,
            dict_metadata,
        ) = read_sandwood_posterior_samples_benchmark_list_chain_ids(
            benchmark_name,
            SANDWOOD_EXECUTION_TARGET,
            SANDWOOD_LIST_CHAIN_IDS,
        )
        dict_metadata[EXECUTION_TARGET_KEY] = SANDWOOD_EXECUTION_TARGET
        dict_metadata[CHAIN_IDS_KEY] = SANDWOOD_LIST_CHAIN_IDS

        return dict_latent_variables_posterior_samples, dict_metadata

    dict_latent_variables_posterior_samples, dict_metadata = read_posterior_samples(
        benchmark_name, language
    )
    dict_metadata = dict(dict_metadata)
    dict_sampling_configuration = dict_metadata.get(SAMPLING_CONFIGURATION_KEY)
    if dict_sampling_configuration is not None:
        num_chains = dict_sampling_configuration[NUM_CHAINS_KEY]
        dict_metadata[CHAIN_IDS_KEY] = list(range(num_chains))

    return dict_latent_variables_posterior_samples, dict_metadata


def get_num_posterior_samples(
    dict_latent_variables_posterior_samples, parameter_names
):
    list_num_posterior_samples = []
    for parameter_name in parameter_names:
        if parameter_name not in dict_latent_variables_posterior_samples:
            raise ValueError("Posterior samples are missing {}".format(parameter_name))
        list_num_posterior_samples.append(
            len(dict_latent_variables_posterior_samples[parameter_name])
        )

    num_posterior_samples = list_num_posterior_samples[0]
    for num_posterior_samples_this_variable in list_num_posterior_samples:
        if num_posterior_samples_this_variable != num_posterior_samples:
            raise ValueError(
                "Latent variables have inconsistent numbers of posterior samples"
            )

    return num_posterior_samples


def calculate_and_save_log_probabilities(
    benchmark_name,
    language,
    construct_stan_input_data,
    parameter_names,
    retained_dimension=None,
    infer_retained_dimension=False,
    model_filename=None,
):
    assert language in SUPPORTED_LANGUAGES
    dict_latent_variables_posterior_samples, dict_metadata = (
        read_posterior_samples_language(benchmark_name, language)
    )
    num_posterior_samples = get_num_posterior_samples(
        dict_latent_variables_posterior_samples, parameter_names
    )
    if infer_retained_dimension and retained_dimension is None:
        retained_dimension = infer_retained_dimension_posterior_samples(
            dict_latent_variables_posterior_samples, parameter_names
        )

    model, dict_stan_data = construct_log_probability_context(
        benchmark_name,
        construct_stan_input_data,
        retained_dimension,
        model_filename=model_filename,
    )
    list_log_probabilities = []
    max_log_prob = None
    for posterior_index in range(num_posterior_samples):
        dict_parameters = {}
        for parameter_name in parameter_names:
            posterior_samples = dict_latent_variables_posterior_samples[parameter_name]
            dict_parameters[parameter_name] = posterior_samples[posterior_index]

        log_prob = calculate_log_probability_given_context(
            model, dict_stan_data, dict_parameters
        )
        list_log_probabilities.append(log_prob)
        if max_log_prob is None or log_prob > max_log_prob:
            max_log_prob = log_prob

        if posterior_index % 100 == 0:
            print(
                "Iteration {:d}: log prob {:f} max log prob so far {}".format(
                    posterior_index, log_prob, max_log_prob
                )
            )

    write_log_probabilities(
        list_log_probabilities,
        benchmark_name,
        language,
        dict_metadata,
    )

    return list_log_probabilities, max_log_prob


def calculate_and_save_log_probabilities_benchmark_spec(benchmark_spec, language):
    return calculate_and_save_log_probabilities(
        benchmark_spec["benchmark_name"],
        language,
        benchmark_spec["construct_stan_input_data"],
        benchmark_spec["parameter_names"],
        infer_retained_dimension=benchmark_spec.get("infer_retained_dimension", False),
        model_filename=benchmark_spec.get("model_filename"),
    )


def calculate_and_save_log_probabilities_all_benchmarks_all_languages():
    dict_max_log_probabilities = {}
    for benchmark_spec in BENCHMARK_SPECS:
        benchmark_name = benchmark_spec["benchmark_name"]
        dict_max_log_probabilities[benchmark_name] = {}
        for language in SUPPORTED_LANGUAGES:
            print(
                "Calculating log probabilities for {} using {}".format(
                    benchmark_name, language
                )
            )
            _, max_log_prob = calculate_and_save_log_probabilities_benchmark_spec(
                benchmark_spec, language
            )
            dict_max_log_probabilities[benchmark_name][language] = max_log_prob
            print(
                "Completed {} using {}. MAP estimate's density: {:.7f}".format(
                    benchmark_name, language, max_log_prob
                )
            )

    return dict_max_log_probabilities


def main(
    benchmark_name,
    construct_stan_input_data,
    parameter_names,
    infer_retained_dimension=False,
    model_filename=None,
):
    args = sys.argv[1:]
    if len(args) == 0:
        raise ValueError("A mode is required")

    mode = args[0]

    if mode == "log_probabilities":
        if len(args) != 2:
            raise ValueError("A language is required")

        language = args[1]
        _, max_log_prob = calculate_and_save_log_probabilities(
            benchmark_name,
            language,
            construct_stan_input_data,
            parameter_names,
            infer_retained_dimension=infer_retained_dimension,
            model_filename=model_filename,
        )
        print("Log probabilities calculated for {}".format(language))
        print("MAP estimate's density: {:.7f}".format(max_log_prob))
    else:
        raise ValueError("Mode {} is not supported".format(mode))


def main_all():
    args = sys.argv[1:]
    if len(args) == 0:
        raise ValueError("A mode is required")

    mode = args[0]

    if mode == "all_log_probabilities":
        if len(args) != 1:
            raise ValueError("Mode all_log_probabilities does not take arguments")

        calculate_and_save_log_probabilities_all_benchmarks_all_languages()
    else:
        raise ValueError("Mode {} is not supported".format(mode))


if __name__ == "__main__":
    main_all()
