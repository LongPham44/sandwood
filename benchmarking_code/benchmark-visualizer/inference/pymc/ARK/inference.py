import sys


from inference.pymc.ARK.utility import (
    construct_pymc_model_standard,
    construct_pymc_model_vectorized,
)
from inference.pymc.utility import (
    construct_pymc_model_from_model_type as construct_pymc_model_from_model_type_common,
    run_pymc_bayesian_inference,
)


def construct_pymc_model_from_model_type(model_type, num_time_steps_retained=None):
    return construct_pymc_model_from_model_type_common(
        model_type,
        construct_pymc_model_standard,
        construct_pymc_model_vectorized,
        num_time_steps_retained,
    )


def run_bayesian_inference(model_type):
    model_name = "ARK"
    model, _ = construct_pymc_model_from_model_type(model_type)
    list_latent_variables_scalar = ["alpha", "sigma"]
    list_latent_variables_vector = ["beta"]
    run_pymc_bayesian_inference(
        model_name,
        model_type,
        model,
        list_latent_variables_scalar,
        list_latent_variables_vector,
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]

    if mode == "infer":
        model_type = "standard"
        if len(args) >= 2:
            model_type = args[1]

        run_bayesian_inference(model_type)
    else:
        raise ValueError("Mode {} is not supported".format(mode))
