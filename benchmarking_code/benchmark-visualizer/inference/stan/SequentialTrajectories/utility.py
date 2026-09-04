from input_output.file_manipulation import read_observed_data


def check_dimension_matrix(nested_list):
    num_inner_lists = len(nested_list)
    print("Number of inners lists: {:d}".format(num_inner_lists))
    for i, inner_list in enumerate(nested_list[:10]):
        print("Inner list {:d}: length {:d}".format(i, len(inner_list)))


def add_padding_matrix(nested_list):
    max_inner_list_length = max([len(inner_list) for inner_list in nested_list])
    nested_list_padded = []
    for inner_list in nested_list:
        inner_list_padded = inner_list + (
            [0] * (max_inner_list_length - len(inner_list))
        )
        nested_list_padded.append(inner_list_padded)

    return nested_list_padded


# Construct a dictionary storing input data for the Stan model. The input
# num_trajectories_retained indicates the number of videos and their time-series
# data to be retained - the rest is discarded. If the input is
# num_trajectories_retained = None, it means all videos will be retained.


def construct_stan_input_data(num_trajectories_retained):
    # Dictionary storing input data (e.g., hyperparameters and observed data)
    # for Stan
    dict_stan_data = {"num_lags": 5, "M": 4096}

    # Read observed data
    benchmark_name = "SequentialTrajectories"
    dict_observed_data = read_observed_data(benchmark_name)
    list_observed_variables = list(dict_observed_data.keys())
    for observed_variable, data in dict_observed_data.items():
        if num_trajectories_retained is not None:
            data = data[:num_trajectories_retained]

        dict_stan_data[observed_variable] = data

    # Figure out the number of videos and the number of time steps for each
    # video
    list_lists_views = dict_stan_data["obsViews"]
    num_videos = len(list_lists_views)
    list_num_time_steps = [len(list_view) for list_view in list_lists_views]
    max_num_time_steps = max(list_num_time_steps)
    dict_stan_data["num_videos"] = num_videos
    dict_stan_data["max_num_time_steps"] = max_num_time_steps
    dict_stan_data["array_num_time_steps"] = list_num_time_steps

    # Check the dimension of matrices for a sanity check
    DEBUG_MODE = False
    if DEBUG_MODE:
        for variable_name in list_observed_variables:
            check_dimension_matrix(dict_stan_data[variable_name])

    # Add padding to matrices so that they are rectangular
    for variable_name in list_observed_variables:
        matrix_non_padded = dict_stan_data[variable_name]
        matrix_padded = add_padding_matrix(matrix_non_padded)
        dict_stan_data[variable_name] = matrix_padded

    return dict_stan_data
