import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from input_output.directories import get_bin_directory_bin_type
from input_output.file_manipulation import (
    read_posterior_samples,
    read_sandwood_posterior_samples_benchmark_list_chain_ids,
)

DICT_LANGUAGE_DISPLAY_NAMES = {
    "sandwood": "Sandwood",
    "stan": "Stan",
    "pymc": "PyMC",
}


def plot_posterior_samples_histogram_ax(
    ax, list_posterior_samples, x_axis_limits, ax_title
):
    x_axis_left, x_axis_right = x_axis_limits
    bins = np.linspace(x_axis_left, x_axis_right, num=200)

    # The Boolean variable separate_chains indicates whether we plot separate
    # histograms for individual chains or plot a single histogram for all chains
    # combined.
    separate_chains = False
    if separate_chains:
        num_chains = 10
        num_posterior_samples = len(list_posterior_samples)
        assert num_posterior_samples % num_chains == 0
        num_samples_per_chain = num_posterior_samples // num_chains
        for i in range(num_chains):
            ax.hist(
                list_posterior_samples[
                    i * num_samples_per_chain : (i + 1) * num_samples_per_chain
                ],
                bins=bins,
                alpha=0.5,
            )
    else:
        ax.hist(list_posterior_samples, bins=bins, color="blue", alpha=0.5)

    ax.set_xlabel("Posterior Sample")
    ax.set_ylabel("Frequency")

    x_axis_left, x_axis_right = x_axis_limits
    ax.set_xlim(left=x_axis_left, right=x_axis_right)

    ax.set_title(ax_title)


def extract_posterior_samples_all_iterations(
    dict_latent_variables_posterior_samples, latent_variable, latent_variable_index
):
    # If the latent variable is scalar, we simply return all posterior samples.
    if latent_variable_index is None:
        return dict_latent_variables_posterior_samples[latent_variable]

    list_list_posterior_samples = dict_latent_variables_posterior_samples[
        latent_variable
    ]
    num_posterior_samples = len(list_list_posterior_samples)
    list_posterior_samples_selected = [
        list_list_posterior_samples[i][latent_variable_index]
        for i in range(num_posterior_samples)
    ]
    return list_posterior_samples_selected


def construct_latent_variable_specs(dict_latent_variables_posterior_samples):
    max_vector_components_display = 20
    latent_variable_specs = []

    for (
        latent_variable,
        posterior_samples,
    ) in dict_latent_variables_posterior_samples.items():
        first_posterior_sample = posterior_samples[0]
        if isinstance(first_posterior_sample, (list, tuple, np.ndarray)):
            num_components_display = min(
                max_vector_components_display,
                len(first_posterior_sample),
            )
            latent_variable_specs.extend(
                (latent_variable, latent_variable_index)
                for latent_variable_index in range(num_components_display)
            )
        else:
            latent_variable_specs.append((latent_variable, None))

    return latent_variable_specs


def plot_posterior_samples(
    dict_languages_dicts_latent_variables_posterior_samples,
    fig_title,
    image_path,
):
    # Figure out the number of probabilistic programming languages stored in the
    # dictionary
    list_languages = list(
        dict_languages_dicts_latent_variables_posterior_samples.keys()
    )
    num_languages = len(list_languages)

    first_language = list_languages[0]
    latent_variable_specs = construct_latent_variable_specs(
        dict_languages_dicts_latent_variables_posterior_samples[first_language]
    )
    num_latent_variables_display = len(latent_variable_specs)

    ax_height, ax_width = 3, 5
    ncols = num_languages
    nrows = num_latent_variables_display
    FIG_SIZE = (ncols * ax_width, nrows * ax_height)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=FIG_SIZE,
        sharex=False,
        sharey=False,
        layout="constrained",
        squeeze=False,
    )

    def calculate_x_axis_limits(latent_variable, latent_variable_index):
        x_axis_left, x_axis_right = None, None
        for (
            dict_latent_variables_posterior_samples
        ) in dict_languages_dicts_latent_variables_posterior_samples.values():
            list_posterior_samples_this_latent_variable = (
                extract_posterior_samples_all_iterations(
                    dict_latent_variables_posterior_samples,
                    latent_variable,
                    latent_variable_index,
                )
            )
            min_posterior_sample = min(list_posterior_samples_this_latent_variable)
            if x_axis_left is None or min_posterior_sample < x_axis_left:
                x_axis_left = min_posterior_sample

            max_posterior_sample = max(list_posterior_samples_this_latent_variable)
            if x_axis_right is None or x_axis_right < max_posterior_sample:
                x_axis_right = max_posterior_sample

        return x_axis_left, x_axis_right

    for i in range(nrows):
        latent_variable, latent_variable_index = latent_variable_specs[i]

        for j in range(ncols):
            ax = axs[i, j]
            language = list_languages[j]
            dict_latent_variables_posterior_samples = (
                dict_languages_dicts_latent_variables_posterior_samples[language]
            )

            # Construct a list of posterior samples for the current Axes object
            list_posterior_samples = extract_posterior_samples_all_iterations(
                dict_latent_variables_posterior_samples,
                latent_variable,
                latent_variable_index,
            )

            ax_title = (
                "Variable: {} Index: {}\nLanguage: {}\n{:d} Posterior Samples".format(
                    latent_variable,
                    latent_variable_index,
                    DICT_LANGUAGE_DISPLAY_NAMES[language],
                    len(list_posterior_samples),
                )
            )
            x_axis_limits = calculate_x_axis_limits(
                latent_variable, latent_variable_index
            )
            plot_posterior_samples_histogram_ax(
                ax, list_posterior_samples, x_axis_limits, ax_title
            )
            print(
                "Plot a histogram of posterior samples: i = {:d}, j = {:d}".format(i, j)
            )

    fig.suptitle(fig_title)
    fig.savefig(image_path, format="png", dpi=300)
    plt.close()


def read_posterior_samples_all_languages(
    benchmark_name, sandwood_execution_target, sandwood_list_chain_ids
):
    dict_sandwood_posterior_samples, _ = (
        read_sandwood_posterior_samples_benchmark_list_chain_ids(
            benchmark_name, sandwood_execution_target, sandwood_list_chain_ids
        )
    )
    dict_stan_posterior_samples, _ = read_posterior_samples(benchmark_name, "stan")
    dict_pymc_posterior_samples, _ = read_posterior_samples(benchmark_name, "pymc")
    return {
        "sandwood": dict_sandwood_posterior_samples,
        "stan": dict_stan_posterior_samples,
        "pymc": dict_pymc_posterior_samples,
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        raise ValueError("A mode is required")

    mode = args[0]

    if mode == "hist":
        if len(args) < 2:
            raise ValueError("A benchmark name is required")
        if len(args) > 2:
            raise ValueError("Expected exactly one benchmark name")

        benchmark_name = args[1]
        sandwood_execution_target = "SingleThread"
        num_chains = 10
        sandwood_list_chain_ids = list(range(42, 42 + num_chains))

        # Obtain posterior samples inferred by various probabilistic programming languages
        dict_languages_dicts_latent_variables_posterior_samples = (
            read_posterior_samples_all_languages(
                benchmark_name, sandwood_execution_target, sandwood_list_chain_ids
            )
        )
        list_languages = list(
            dict_languages_dicts_latent_variables_posterior_samples.keys()
        )
        list_languages_in_title = [
            DICT_LANGUAGE_DISPLAY_NAMES[language] for language in list_languages
        ]
        fig_title = "Posterior Distributions for {} Returned by PPLs: {}".format(
            benchmark_name,
            ", ".join(list_languages_in_title),
        )
        image_directory = get_bin_directory_bin_type("image")
        image_filename = "posterior-distributions-{}.png".format(benchmark_name)
        image_path = os.path.join(image_directory, image_filename)
        plot_posterior_samples(
            dict_languages_dicts_latent_variables_posterior_samples,
            fig_title,
            image_path,
        )
    else:
        raise ValueError("Mode {} is not supported".format(mode))
