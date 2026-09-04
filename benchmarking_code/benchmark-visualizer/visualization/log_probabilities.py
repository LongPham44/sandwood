import os
import sys

import matplotlib.pyplot as plt

from input_output.directories import get_bin_directory_bin_type
from input_output.file_manipulation import (
    NUM_CHAINS_KEY,
    NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY,
    RUNNING_TIME_SECONDS_KEY,
    SAMPLING_CONFIGURATION_KEY,
    read_log_probabilities,
    read_posterior_samples,
)

DICT_LANGUAGE_DISPLAY_NAMES = {
    "sandwood": "Sandwood",
    "stan": "Stan",
    "pymc": "PyMC",
}
BENCHMARK_NAMES = [
    "ARK",
    "SequentialTrajectories",
    "Earnings",
    "EightSchoolsCentered",
    "EightSchoolsNonCentered",
    "NES",
    "SBLRC",
    "HMMExample",
    "LDAK5",
]


def split_log_probabilities_by_chain(list_log_probabilities, dict_metadata):
    if dict_metadata is None:
        raise ValueError("Metadata is required to split log probabilities by chain")

    dict_sampling_configuration = dict_metadata[SAMPLING_CONFIGURATION_KEY]
    num_chains = dict_sampling_configuration[NUM_CHAINS_KEY]
    num_posterior_samples_per_chain = dict_sampling_configuration[
        NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY
    ]
    num_log_probabilities = len(list_log_probabilities)
    num_log_probabilities_expected = num_chains * num_posterior_samples_per_chain
    if num_log_probabilities != num_log_probabilities_expected:
        raise ValueError(
            "Expected {:d} log probabilities, but found {:d}".format(
                num_log_probabilities_expected, num_log_probabilities
            )
        )

    list_list_log_probabilities = []
    for chain_index in range(num_chains):
        start_index = chain_index * num_posterior_samples_per_chain
        end_index = start_index + num_posterior_samples_per_chain
        list_list_log_probabilities.append(
            list_log_probabilities[start_index:end_index]
        )

    return list_list_log_probabilities


def extract_running_time_seconds(running_time_entry):
    if isinstance(running_time_entry, (int, float)):
        return running_time_entry

    if isinstance(running_time_entry, list):
        list_running_times = [
            extract_running_time_seconds(running_time_entry_this)
            for running_time_entry_this in running_time_entry
        ]
        list_running_times = [
            running_time
            for running_time in list_running_times
            if isinstance(running_time, (int, float))
        ]
        if len(list_running_times) > 0:
            return max(list_running_times)

    if isinstance(running_time_entry, dict):
        if "total" in running_time_entry:
            return running_time_entry["total"]

        if "sampling" in running_time_entry:
            return extract_running_time_seconds(running_time_entry["sampling"])

        list_numeric_values = [
            value
            for value in running_time_entry.values()
            if isinstance(value, (int, float))
        ]
        if len(list_numeric_values) > 0:
            return sum(list_numeric_values)

    return None


def format_running_time_text(dict_metadata):
    if dict_metadata is None:
        return "Running Time: Not Available"

    if RUNNING_TIME_SECONDS_KEY not in dict_metadata:
        return "Running Time: Not Available"

    running_time_seconds = dict_metadata[RUNNING_TIME_SECONDS_KEY]
    if not isinstance(running_time_seconds, list):
        running_time = extract_running_time_seconds(running_time_seconds)
        if running_time is None:
            return "Running Time: Not Available"

        return "Running Time: {:.4f} seconds".format(running_time)

    list_running_times = [
        extract_running_time_seconds(running_time_entry)
        for running_time_entry in running_time_seconds
    ]
    list_running_times = [
        running_time for running_time in list_running_times if running_time is not None
    ]
    if len(list_running_times) == 0:
        return "Running Time: Not Available"

    if all(
        running_time == list_running_times[0] for running_time in list_running_times
    ):
        return "Running Time: {:.4f} seconds".format(list_running_times[0])

    return "Max Chain Running Time: {:.4f} seconds".format(max(list_running_times))


def format_max_log_probability_text(list_log_probabilities):
    return "Max Log Probability: {:.4f}".format(max(list_log_probabilities))


def refresh_running_time_from_posterior_samples(
    benchmark_name, language, dict_metadata
):
    if language == "sandwood":
        return dict_metadata

    try:
        _, dict_posterior_metadata = read_posterior_samples(benchmark_name, language)
    except FileNotFoundError:
        return dict_metadata

    if (
        dict_posterior_metadata is None
        or RUNNING_TIME_SECONDS_KEY not in dict_posterior_metadata
    ):
        return dict_metadata

    dict_metadata_refreshed = {} if dict_metadata is None else dict(dict_metadata)
    dict_metadata_refreshed[RUNNING_TIME_SECONDS_KEY] = dict_posterior_metadata[
        RUNNING_TIME_SECONDS_KEY
    ]
    return dict_metadata_refreshed


def plot_log_probabilities_trajectory_ax(
    ax, list_list_log_probabilities, list_chain_labels, ax_title
):
    for chain_index in range(len(list_list_log_probabilities)):
        list_log_probabilities_chain = list_list_log_probabilities[chain_index]
        list_iterations = list(range(len(list_log_probabilities_chain)))
        ax.plot(
            list_iterations,
            list_log_probabilities_chain,
            alpha=0.8,
            linewidth=1,
            label="Chain {}".format(list_chain_labels[chain_index]),
        )

    ax.set_xlabel("Posterior Sample Index")
    ax.set_ylabel("Log Probability")

    ax.tick_params(labelleft=True)

    ax.set_title(ax_title)
    ax.legend(fontsize="small")


def plot_log_probabilities(
    dict_languages_lists_log_probabilities,
    dict_languages_metadata,
    fig_title,
    image_path,
):
    # Figure out the number of probabilistic programming languages stored in the
    # dictionary
    list_languages = list(dict_languages_lists_log_probabilities.keys())
    num_languages = len(list_languages)

    ax_height, ax_width = 4, 8
    ncols = num_languages
    nrows = 1
    FIG_SIZE = (ncols * ax_width, nrows * ax_height)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=FIG_SIZE,
        sharex=False,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )

    for j in range(ncols):
        ax = axs[0, j]
        language = list_languages[j]
        list_log_probabilities = dict_languages_lists_log_probabilities[language]
        dict_metadata = dict_languages_metadata[language]

        list_list_log_probabilities = split_log_probabilities_by_chain(
            list_log_probabilities, dict_metadata
        )
        num_chains = len(list_list_log_probabilities)
        num_log_probabilities = len(list_log_probabilities)
        chain_text = "{:d} Chains".format(num_chains)
        list_chain_labels = list(range(1, num_chains + 1))

        running_time_text = format_running_time_text(dict_metadata)
        max_log_probability_text = format_max_log_probability_text(
            list_log_probabilities
        )
        ax_title = "Language: {}\n{:d} Posterior Samples {}\n{}\n{}".format(
            DICT_LANGUAGE_DISPLAY_NAMES[language],
            num_log_probabilities,
            chain_text,
            running_time_text,
            max_log_probability_text,
        )
        plot_log_probabilities_trajectory_ax(
            ax,
            list_list_log_probabilities,
            list_chain_labels,
            ax_title,
        )
        print("Plot log probability trajectories: j = {:d}".format(j))

    fig.suptitle(fig_title)
    fig.savefig(image_path, format="png", dpi=300)
    plt.close()


def read_log_probabilities_all_languages(benchmark_name):
    list_sandwood_log_probabilities, dict_sandwood_metadata = read_log_probabilities(
        benchmark_name, "sandwood"
    )
    list_stan_log_probabilities, dict_stan_metadata = read_log_probabilities(
        benchmark_name, "stan"
    )
    list_pymc_log_probabilities, dict_pymc_metadata = read_log_probabilities(
        benchmark_name, "pymc"
    )
    dict_stan_metadata = refresh_running_time_from_posterior_samples(
        benchmark_name, "stan", dict_stan_metadata
    )
    dict_pymc_metadata = refresh_running_time_from_posterior_samples(
        benchmark_name, "pymc", dict_pymc_metadata
    )
    return (
        {
            "sandwood": list_sandwood_log_probabilities,
            "stan": list_stan_log_probabilities,
            "pymc": list_pymc_log_probabilities,
        },
        {
            "sandwood": dict_sandwood_metadata,
            "stan": dict_stan_metadata,
            "pymc": dict_pymc_metadata,
        },
    )


def plot_log_probabilities_trajectories_benchmark(benchmark_name):
    # Obtain log probabilities of posterior samples inferred by various
    # probabilistic programming languages
    (
        dict_languages_lists_log_probabilities,
        dict_languages_metadata,
    ) = read_log_probabilities_all_languages(benchmark_name)
    list_languages = list(dict_languages_lists_log_probabilities.keys())
    list_languages_in_title = [
        DICT_LANGUAGE_DISPLAY_NAMES[language] for language in list_languages
    ]
    fig_title = "Log Probability Trajectories for {} Returned by PPLs: {}".format(
        benchmark_name,
        ", ".join(list_languages_in_title),
    )
    image_directory = get_bin_directory_bin_type("image")
    image_filename = "log-probabilities-trajectories-{}.png".format(benchmark_name)
    image_path = os.path.join(image_directory, image_filename)
    plot_log_probabilities(
        dict_languages_lists_log_probabilities,
        dict_languages_metadata,
        fig_title,
        image_path,
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        raise ValueError("A mode is required")

    mode = args[0]

    if mode == "trajectories":
        if len(args) < 2:
            raise ValueError("A benchmark name is required")
        if len(args) > 2:
            raise ValueError("Expected exactly one benchmark name")

        benchmark_name = args[1]
        if benchmark_name == "all":
            for benchmark_name_this in BENCHMARK_NAMES:
                print(
                    "Plot log probability trajectories for {}".format(
                        benchmark_name_this
                    )
                )
                plot_log_probabilities_trajectories_benchmark(benchmark_name_this)
        else:
            plot_log_probabilities_trajectories_benchmark(benchmark_name)
    else:
        raise ValueError("Mode {} is not supported".format(mode))
