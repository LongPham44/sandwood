from cmdstanpy import CmdStanModel
import os


from input_output.directories import get_stan_model_directory


def construct_stan_model(model_filename):
    stan_directory = get_stan_model_directory()
    model_filepath = os.path.join(stan_directory, model_filename)
    model = CmdStanModel(stan_file=model_filepath)
    return model


def extract_stan_running_time_seconds(fit):
    list_running_time_seconds = []
    for dict_running_time in fit.time:
        list_running_time_seconds.append(
            {
                "warmup": dict_running_time["warmup"],
                "sampling": dict_running_time["sampling"],
                "total": dict_running_time["total"],
            }
        )

    return list_running_time_seconds
