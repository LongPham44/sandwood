import os


# Obtain a directory of posterior samples returned by Sandwood


def get_sandwood_benchmark_root_directory():
    return os.path.expanduser(
        os.path.join("~", "Research-code", "ppl-benchmarks", "Benchmarking")
    )


def get_sandwood_bin_posterior_samples(benchmark_name):
    sandwood_benchmark_root_directory = get_sandwood_benchmark_root_directory()
    sandwood_benchmark_directory = os.path.join(
        sandwood_benchmark_root_directory,
        "src",
        "main",
        "resources",
        "expectedOutputs",
        "SandwoodBenchmarkDriver",
        benchmark_name,
    )
    return sandwood_benchmark_directory


# Obtain a directory of observed data, which reside inside the project directory
# of Sandwood benchmarking


def get_observed_data_directory(benchmark_name):
    sandwood_benchmark_root_directory = get_sandwood_benchmark_root_directory()
    observed_data_directory = os.path.join(
        sandwood_benchmark_root_directory,
        "src",
        "main",
        "resources",
        "inputs",
        "org",
        "sandwood",
        "benchmarking",
        "observedData",
        benchmark_name,
    )
    return observed_data_directory


# Obtain a directory storing Stan model files


def get_stan_model_directory():
    stan_directory = os.path.expanduser(
        os.path.join(
            "~",
            "Research-code",
            "ppl-benchmarks",
            "benchmark-visualizer",
            "stan_models",
        )
    )
    return stan_directory


# Obtain a bin directory


def get_bin_root_directory():
    bin_root_directory = os.path.expanduser(
        os.path.join(
            "~",
            "Research-code",
            "ppl-benchmarks",
            "benchmark-visualizer",
            "bin",
        )
    )
    if not os.path.exists(bin_root_directory):
        os.makedirs(bin_root_directory)

    return bin_root_directory


def get_bin_directory_bin_type(bin_type):
    bin_root_directory = get_bin_root_directory()
    bin_data_type_directory = os.path.join(bin_root_directory, bin_type)

    if not os.path.exists(bin_data_type_directory):
        os.makedirs(bin_data_type_directory)

    return bin_data_type_directory
