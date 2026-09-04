import sys
import time

from input_output.directories import get_bin_directory_bin_type
from input_output.file_manipulation import (
    NUM_CHAINS_KEY,
    NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY,
    NUM_WARMUPS_PER_CHAIN_KEY,
    RUNNING_TIME_SECONDS_KEY,
    SAMPLING_CONFIGURATION_KEY,
    write_posterior_samples,
)
from inference.stan.ARMA.utility import construct_stan_input_data
from inference.stan.utility import construct_stan_model, extract_stan_running_time_seconds

DEBUG_MODE = False


def run_bayesian_inference():
    model_name = "ARMA"
    model = construct_stan_model("{}.stan".format(model_name))
    dict_stan_data = construct_stan_input_data()

    num_chains = 10
    stan_seed = 42
    stan_seeds_chains = list(range(stan_seed, stan_seed + num_chains))
    num_warmups = 1500
    num_posterior_samples = 1000

    start_time = time.perf_counter()
    stan_output_directory = get_bin_directory_bin_type("stan-output")
    fit = model.sample(
        data=dict_stan_data,
        chains=num_chains,
        seed=stan_seeds_chains,
        iter_warmup=num_warmups,
        iter_sampling=num_posterior_samples,
        adapt_engaged=True,
        show_console=DEBUG_MODE,
        # output_dir=stan_output_directory,
    )
    end_time = time.perf_counter()
    print("Running time of Bayesian inference: {:.4f} seconds".format(end_time - start_time))

    dict_metadata = {
        RUNNING_TIME_SECONDS_KEY: extract_stan_running_time_seconds(fit),
        SAMPLING_CONFIGURATION_KEY: {
            NUM_CHAINS_KEY: num_chains,
            NUM_WARMUPS_PER_CHAIN_KEY: num_warmups,
            NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY: num_posterior_samples,
        },
    }
    write_posterior_samples(fit.stan_variables(), model_name, "stan", dict_metadata)
    print("Bayesian inference is complete")


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]
    if mode == "infer":
        run_bayesian_inference()
    else:
        raise ValueError("Mode {} is not supported".format(mode))

