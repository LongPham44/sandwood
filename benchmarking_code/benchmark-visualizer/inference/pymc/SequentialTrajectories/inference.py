import sys


from inference.pymc.SequentialTrajectories.utility import (
    construct_pymc_model_standard,
    construct_pymc_model_vectorized,
)
from inference.pymc.utility import (
    construct_pymc_model_from_model_type as construct_pymc_model_from_model_type_common,
    run_pymc_bayesian_inference,
)

# Construct an appropriate version of a PyMC model depending on the
# user-specified model type (i.e., standard or vectorized)


def construct_pymc_model_from_model_type(model_type, num_trajectories_retained):
    return construct_pymc_model_from_model_type_common(
        model_type,
        construct_pymc_model_standard,
        construct_pymc_model_vectorized,
        num_trajectories_retained,
    )


def run_bayesian_inference(model_type):
    model_name = "SequentialTrajectories"
    num_trajectories_retained = 1500
    model, _ = construct_pymc_model_from_model_type(
        model_type, num_trajectories_retained
    )
    list_latent_variables_scalar = ["gamma", "a", "c", "sigma2"]
    list_latent_variables_vector = ["b", "qualities"]
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
        model_type = "vectorized"
        if len(args) >= 2:
            model_type = args[1]

        run_bayesian_inference(model_type)
    else:
        raise ValueError("Mode {} is not supported".format(mode))
