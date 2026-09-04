import sys
import time


from input_output.directories import (
    get_bin_directory_bin_type,
)
from input_output.file_manipulation import (
    NUM_CHAINS_KEY,
    NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY,
    NUM_WARMUPS_PER_CHAIN_KEY,
    RUNNING_TIME_SECONDS_KEY,
    SAMPLING_CONFIGURATION_KEY,
    write_posterior_samples,
)
from inference.stan.utility import (
    construct_stan_model,
    extract_stan_running_time_seconds,
)
from inference.stan.ARK.utility import (
    construct_stan_input_data,
)

DEBUG_MODE = False


# Run sampling-based Bayesian inference (i.e., NUTS implemented in Stan)


def run_bayesian_inference():
    # Construct a Stan model
    model_name = "ARK"
    model_filename = "{}.stan".format(model_name)
    model = construct_stan_model(model_filename)

    # Construct a dictionary storing input data for the Stan model
    dict_stan_data = construct_stan_input_data()

    # Perform Bayesian inference
    num_chains = 10
    stan_seed = 42
    stan_seeds_chains = list(range(stan_seed, stan_seed + num_chains))
    num_warmups = 1500
    num_posterior_samples = 1000

    # Start a timer to measure execution time of Bayesian inference
    start_time = time.perf_counter()

    # Specify the directory to store the output Stan CSV files
    stan_output_directory = get_bin_directory_bin_type("stan-output")

    fit = model.sample(
        data=dict_stan_data,
        chains=num_chains,
        seed=stan_seeds_chains,
        iter_warmup=num_warmups,
        iter_sampling=num_posterior_samples,
        adapt_engaged=True,
        show_console=DEBUG_MODE,  # For debugging, set this argument to True
        # output_dir=stan_output_directory,
    )

    # Stop the timer
    end_time = time.perf_counter()
    stan_execution_time = end_time - start_time
    print(
        "Running time of Bayesian inference: {:.4f} seconds".format(stan_execution_time)
    )

    # print(fit.summary())

    # Save the posterior samples in a JSON file
    dict_latent_variables_posterior_samples = fit.stan_variables()
    list_running_time_seconds = extract_stan_running_time_seconds(fit)
    dict_metadata = {
        RUNNING_TIME_SECONDS_KEY: list_running_time_seconds,
        SAMPLING_CONFIGURATION_KEY: {
            NUM_CHAINS_KEY: num_chains,
            NUM_WARMUPS_PER_CHAIN_KEY: num_warmups,
            NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY: num_posterior_samples,
        },
    }
    language = "stan"
    write_posterior_samples(
        dict_latent_variables_posterior_samples,
        model_name,
        language,
        dict_metadata,
    )

    print("Bayesian inference is complete")


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]

    if mode == "infer":
        run_bayesian_inference()
    else:
        raise ValueError("Mode {} is not supported".format(mode))
