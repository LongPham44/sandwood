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


def construct_pymc_model_from_model_type(
    model_type,
    construct_pymc_model_standard,
    construct_pymc_model_vectorized,
    *args,
):
    if model_type == "standard":
        return construct_pymc_model_standard(*args)
    if model_type == "vectorized":
        return construct_pymc_model_vectorized(*args)

    raise ValueError(
        "PyMC model type {} is not supported. Use 'standard' or 'vectorized'.".format(
            model_type
        )
    )


def extract_pymc_running_time_seconds(inference_result):
    sample_stats = inference_result.sample_stats
    if "perf_counter_diff" not in sample_stats:
        raise ValueError(
            "Could not extract posterior sampling time from PyMC sample_stats"
        )

    posterior_sampling_time_seconds = (
        sample_stats["perf_counter_diff"].sum(dim="draw").values
    )
    list_posterior_sampling_time_seconds = (
        np.asarray(posterior_sampling_time_seconds).reshape(-1).tolist()
    )

    sampling_time_seconds = None
    for attrs_owner in [
        sample_stats,
        getattr(inference_result, "posterior", None),
        inference_result,
    ]:
        attrs = getattr(attrs_owner, "attrs", {})
        if "sampling_time" in attrs:
            sampling_time_seconds = attrs["sampling_time"]
            break

    if sampling_time_seconds is None:
        raise ValueError("Could not extract sampling_time from PyMC inference result")

    total_sampling_time_seconds = float(
        np.asarray(sampling_time_seconds).reshape(-1)[0]
    )
    return [
        {
            "sampling": posterior_sampling_time_seconds_this_chain,
            "total": total_sampling_time_seconds,
        }
        for posterior_sampling_time_seconds_this_chain in (
            list_posterior_sampling_time_seconds
        )
    ]


def extract_pymc_posterior_samples(
    inference_result,
    list_latent_variables_scalar,
    list_latent_variables_vector,
):
    posterior_group = inference_result.posterior
    dict_latent_variables_posterior_samples = {}

    for variable_name in list_latent_variables_scalar + list_latent_variables_vector:
        posterior_samples = np.asarray(posterior_group[variable_name].values)
        print(
            "Variable {}: dimension {}".format(variable_name, posterior_samples.shape)
        )
        if variable_name in list_latent_variables_scalar:
            dict_latent_variables_posterior_samples[variable_name] = (
                posterior_samples.reshape(-1)
            )
        else:
            dict_latent_variables_posterior_samples[variable_name] = (
                posterior_samples.reshape(-1, posterior_samples.shape[-1])
            )
        print(
            "Dimension of the flattened posterior distribution: {}".format(
                dict_latent_variables_posterior_samples[variable_name].shape
            )
        )

    return dict_latent_variables_posterior_samples


def run_pymc_bayesian_inference(
    model_name,
    model_type,
    model,
    list_latent_variables_scalar,
    list_latent_variables_vector,
):
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

    # The execution time of PyMC given by end_time - start_time contains not
    # only the sampling time (for both warmups and posterior samples) but also
    # the preparation/initialization time of NUTS using jitters+adapt_diag.
    #
    # In the comparison of the running time of probabilistic programming
    # languages, we would like to focus on the sampling time, excluding the
    # compilation time (or any kind of initialization time). Hence, when storing
    # the inference result of PyMC, instead of storing end_time - start_time, we
    # should look at the field sampling_time stored inside the inference-result
    # object returned by the method sample of PyMC.
    #
    # Although we do not store end_time - start_time as the official running
    # time in the output JSON file, we still print it out on the standard
    # output.
    end_time = time.perf_counter()
    pymc_execution_time = end_time - start_time
    print(
        "Running time of Bayesian inference: {:.4f} seconds".format(pymc_execution_time)
    )

    dict_latent_variables_posterior_samples = extract_pymc_posterior_samples(
        inference_result,
        list_latent_variables_scalar,
        list_latent_variables_vector,
    )

    print("Bayesian inference is complete")

    list_running_time_seconds = extract_pymc_running_time_seconds(inference_result)
    dict_metadata = {
        RUNNING_TIME_SECONDS_KEY: list_running_time_seconds,
        SAMPLING_CONFIGURATION_KEY: {
            NUM_CHAINS_KEY: num_chains,
            NUM_WARMUPS_PER_CHAIN_KEY: num_warmups,
            NUM_POSTERIOR_SAMPLES_PER_CHAIN_KEY: num_posterior_samples,
        },
    }
    language = "pymc"
    write_posterior_samples(
        dict_latent_variables_posterior_samples,
        model_name,
        language,
        dict_metadata,
    )
