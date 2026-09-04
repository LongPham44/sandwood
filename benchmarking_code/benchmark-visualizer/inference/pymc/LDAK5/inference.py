import sys
import time

import numpy as np
import pymc as pm

from input_output.file_manipulation import (
    NUM_CHAINS_KEY,
    NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY,
    NUM_WARMUPS_PER_CHAIN_KEY,
    RUNNING_TIME_SECONDS_KEY,
    SAMPLING_CONFIGURATION_KEY,
    write_posterior_samples,
)
from inference.pymc.LDAK5.utility import (
    construct_pymc_model_standard,
    construct_pymc_model_vectorized,
)
from inference.pymc.utility import (
    construct_pymc_model_from_model_type as construct_pymc_model_from_model_type_common,
    extract_pymc_running_time_seconds,
)


def construct_pymc_model_from_model_type(model_type, num_observations_retained=None):
    return construct_pymc_model_from_model_type_common(
        model_type,
        construct_pymc_model_standard,
        construct_pymc_model_vectorized,
        num_observations_retained,
    )


def extract_lda_posterior_samples(inference_result):
    posterior_group = inference_result.posterior
    dict_latent_variables_posterior_samples = {}

    for variable_name in ["theta", "phi"]:
        posterior_samples = np.asarray(posterior_group[variable_name].values)
        print(
            "Variable {}: dimension {}".format(variable_name, posterior_samples.shape)
        )
        dict_latent_variables_posterior_samples[variable_name] = (
            posterior_samples.reshape(
                -1,
                posterior_samples.shape[-2],
                posterior_samples.shape[-1],
            )
        )
        print(
            "Dimension of the flattened posterior distribution: {}".format(
                dict_latent_variables_posterior_samples[variable_name].shape
            )
        )

    return dict_latent_variables_posterior_samples


def run_bayesian_inference(model_type):
    model_name = "LDAK5"
    model, _ = construct_pymc_model_from_model_type(model_type)

    print("Running PyMC Bayesian inference with {} model".format(model_type))

    num_chains = 10
    pymc_seed = 42
    num_warmups = 1500
    num_posterior_samples = 1000

    start_time = time.perf_counter()

    with model:
        inference_result = pm.sample(
            draws=num_posterior_samples,
            tune=num_warmups,
            chains=num_chains,
            cores=num_chains,
            random_seed=pymc_seed,
            progressbar=True,
            compute_convergence_checks=True,
            discard_tuned_samples=True,
            return_inferencedata=True,
        )

    end_time = time.perf_counter()
    pymc_execution_time = end_time - start_time
    print(
        "Running time of Bayesian inference: {:.4f} seconds".format(pymc_execution_time)
    )

    dict_latent_variables_posterior_samples = extract_lda_posterior_samples(
        inference_result
    )

    print("Bayesian inference is complete")

    dict_metadata = {
        RUNNING_TIME_SECONDS_KEY: extract_pymc_running_time_seconds(inference_result),
        SAMPLING_CONFIGURATION_KEY: {
            NUM_CHAINS_KEY: num_chains,
            NUM_WARMUPS_PER_CHAIN_KEY: num_warmups,
            NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY: num_posterior_samples,
        },
    }
    write_posterior_samples(
        dict_latent_variables_posterior_samples,
        model_name,
        "pymc",
        dict_metadata,
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]
    if mode == "infer":
        model_type = "vectorized"
        if len(args) >= 2:
            model_type = args[1]
        run_bayesian_inference(model_type)
    else:
        raise ValueError("Mode {} is not supported".format(mode))
