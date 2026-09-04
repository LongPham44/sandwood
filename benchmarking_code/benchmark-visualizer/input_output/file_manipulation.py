import os
import json
import sys

from input_output.directories import (
    get_sandwood_bin_posterior_samples,
    get_observed_data_directory,
    get_bin_directory_bin_type,
)

RUNNING_TIME_SECONDS_KEY = "runningTimeSeconds"
METADATA_KEY = "metadata"
SAMPLING_CONFIGURATION_KEY = "samplingConfiguration"
NUM_CHAINS_KEY = "numChains"
NUM_WARMUPS_PER_CHAIN_KEY = "numWarmupsPerChain"
NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY = "numPosteriorSamplesPerChain"
LOG_PROBABILITIES_KEY = "logProbabilities"
CHAIN_IDS_KEY = "chainIds"
EXECUTION_TARGET_KEY = "executionTarget"

# Read posterior samples from Sandwood


def read_sandwood_posterior_samples_benchmark_chain_id(
    benchmark_name, execution_target, chain_id
):
    sandwood_benchmark_directory = get_sandwood_bin_posterior_samples(benchmark_name)
    filename = "GibbsSample{:d}-{}CPU.json".format(chain_id, execution_target)
    file_path = os.path.join(sandwood_benchmark_directory, filename)

    with open(file_path) as json_file:
        dict_posterior_samples_raw = json.load(json_file)

    dict_metadata = dict_posterior_samples_raw.pop(METADATA_KEY)

    dict_latent_variables_posterior_samples = {
        variable_name: posterior_samples["value"]
        for variable_name, posterior_samples in dict_posterior_samples_raw.items()
    }

    return dict_latent_variables_posterior_samples, dict_metadata


def read_sandwood_posterior_samples_benchmark_list_chain_ids(
    benchmark_name, execution_target, list_chain_ids
):
    dict_latent_variables_posterior_samples_all_chains = {}
    dict_metadata_all_chains = None
    list_running_time_seconds = []
    for chain_id in list_chain_ids:
        (
            dict_latent_variables_posterior_samples_this_chain,
            dict_metadata_this_chain,
        ) = read_sandwood_posterior_samples_benchmark_chain_id(
            benchmark_name, execution_target, chain_id
        )
        if dict_metadata_all_chains is None:
            dict_metadata_all_chains = dict(dict_metadata_this_chain)

        list_running_time_seconds.append(
            dict_metadata_this_chain[RUNNING_TIME_SECONDS_KEY]
        )
        for (
            variable_name,
            posterior_samples_this_chain,
        ) in dict_latent_variables_posterior_samples_this_chain.items():
            if variable_name not in dict_latent_variables_posterior_samples_all_chains:
                dict_latent_variables_posterior_samples_all_chains[variable_name] = (
                    posterior_samples_this_chain
                )
            else:
                dict_latent_variables_posterior_samples_all_chains[
                    variable_name
                ].extend(posterior_samples_this_chain)

    dict_metadata_all_chains[RUNNING_TIME_SECONDS_KEY] = list_running_time_seconds
    if SAMPLING_CONFIGURATION_KEY in dict_metadata_all_chains:
        dict_metadata_all_chains[SAMPLING_CONFIGURATION_KEY][NUM_CHAINS_KEY] = len(
            list_chain_ids
        )

    return (
        dict_latent_variables_posterior_samples_all_chains,
        dict_metadata_all_chains,
    )


# Read observed data, which are stored inside the Sandwood benchmarking
# directory


def read_observed_data(benchmark_name):
    observed_data_directory = get_observed_data_directory(benchmark_name)
    filename = "observed-data.json"
    file_path = os.path.join(observed_data_directory, filename)

    with open(file_path) as json_file:
        dict_observed_data = json.load(json_file)

    return dict_observed_data


# Read and write posterior samples of a specified probabilistic programming
# language (i.e., Stan and PyMC) to JSON files


def read_posterior_samples(benchmark_name, language):
    posterior_samples_bin_directory = get_bin_directory_bin_type("posterior-samples")
    posterior_samples_language_bin_directory = os.path.join(
        posterior_samples_bin_directory, language
    )
    posterior_samples_filename = "{}.json".format(benchmark_name)
    posterior_samples_filepath = os.path.join(
        posterior_samples_language_bin_directory, posterior_samples_filename
    )

    with open(posterior_samples_filepath) as json_file:
        dict_latent_variables_posterior_samples = json.load(json_file)

    dict_metadata = dict_latent_variables_posterior_samples.pop(METADATA_KEY)

    return dict_latent_variables_posterior_samples, dict_metadata


def write_posterior_samples(
    dict_latent_variables_posterior_samples,
    benchmark_name,
    language,
    dict_metadata=None,
):
    posterior_samples_bin_directory = get_bin_directory_bin_type("posterior-samples")
    posterior_samples_language_bin_directory = os.path.join(
        posterior_samples_bin_directory, language
    )
    if not os.path.exists(posterior_samples_language_bin_directory):
        os.makedirs(posterior_samples_language_bin_directory)

    posterior_samples_filename = "{}.json".format(benchmark_name)
    posterior_samples_filepath = os.path.join(
        posterior_samples_language_bin_directory, posterior_samples_filename
    )

    def default_json_serializer(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError("{} is not JSON serializable".format(type(value).__name__))

    dict_output = dict(dict_latent_variables_posterior_samples)
    if dict_metadata is not None:
        dict_output[METADATA_KEY] = dict_metadata

    with open(posterior_samples_filepath, "w") as json_file:
        json.dump(
            dict_output,
            json_file,
            default=default_json_serializer,
        )

    print(
        "Posterior samples have been written to a JSON file: {}".format(
            posterior_samples_filepath
        )
    )


# Read and write log probabilities of posterior samples of a specified
# probabilistic programming language to JSON files


def read_log_probabilities(benchmark_name, language):
    log_probabilities_bin_directory = get_bin_directory_bin_type("log-probabilities")
    log_probabilities_language_bin_directory = os.path.join(
        log_probabilities_bin_directory, language
    )
    log_probabilities_filename = "{}.json".format(benchmark_name)
    log_probabilities_filepath = os.path.join(
        log_probabilities_language_bin_directory, log_probabilities_filename
    )

    with open(log_probabilities_filepath) as json_file:
        dict_log_probabilities = json.load(json_file)

    list_log_probabilities = dict_log_probabilities[LOG_PROBABILITIES_KEY]
    dict_metadata = dict_log_probabilities.pop(METADATA_KEY, None)

    return list_log_probabilities, dict_metadata


def write_log_probabilities(
    list_log_probabilities,
    benchmark_name,
    language,
    dict_metadata=None,
):
    log_probabilities_bin_directory = get_bin_directory_bin_type("log-probabilities")
    log_probabilities_language_bin_directory = os.path.join(
        log_probabilities_bin_directory, language
    )
    if not os.path.exists(log_probabilities_language_bin_directory):
        os.makedirs(log_probabilities_language_bin_directory)

    log_probabilities_filename = "{}.json".format(benchmark_name)
    log_probabilities_filepath = os.path.join(
        log_probabilities_language_bin_directory, log_probabilities_filename
    )

    dict_output = {LOG_PROBABILITIES_KEY: list_log_probabilities}
    if dict_metadata is not None:
        dict_output[METADATA_KEY] = dict_metadata

    with open(log_probabilities_filepath, "w") as json_file:
        json.dump(dict_output, json_file)

    print(
        "Log probabilities have been written to a JSON file: {}".format(
            log_probabilities_filepath
        )
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]

    if mode == "test":
        benchmark_name = "SequentialTrajectories"
        execution_target = "SingleThread"
        list_chain_ids = [42, 43]
        (
            dict_latent_variables_posterior_samples_all_chains,
            dict_metadata,
        ) = read_sandwood_posterior_samples_benchmark_list_chain_ids(
            benchmark_name, execution_target, list_chain_ids
        )
        print(dict_latent_variables_posterior_samples_all_chains.keys())
        print(dict_metadata)
        list_posterior_samples = dict_latent_variables_posterior_samples_all_chains["b"]
        print(len(list_posterior_samples))
        print(type(list_posterior_samples[0][0]))
    elif mode == "test2":
        benchmark_name = "LDAK5"
        (
            dict_latent_variables_posterior_samples_all_chains,
            dict_metadata,
        ) = read_posterior_samples(benchmark_name, "pymc")
        # print(dict_latent_variables_posterior_samples_all_chains.keys())
        print(dict_metadata)
        # list_posterior_samples = dict_latent_variables_posterior_samples_all_chains["b"]
        # print(len(list_posterior_samples))
        # print(type(list_posterior_samples[0][0]))
    else:
        raise ValueError("Mode {} is not supported".format(mode))
