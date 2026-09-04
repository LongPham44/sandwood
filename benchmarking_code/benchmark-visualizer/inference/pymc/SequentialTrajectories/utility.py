import sys
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from inference.stan.SequentialTrajectories.utility import construct_stan_input_data

# Create a standard version of a PyMC model. This version compiles painfully
# slowly. To speed it up, we can use a vectorized version of the model, which is
# implemented later in this file.


def construct_pymc_input_data_standard(num_trajectories_retained):
    dict_stan_data = construct_stan_input_data(num_trajectories_retained)

    num_videos = dict_stan_data["num_videos"]
    max_num_time_steps = dict_stan_data["max_num_time_steps"]
    num_lags = dict_stan_data["num_lags"]

    obs_views = np.asarray(dict_stan_data["obsViews"], dtype=np.float64)
    obs_views_lags = np.asarray(dict_stan_data["obsViewsLags"], dtype=np.float64)
    obs_shares_lag1 = np.asarray(dict_stan_data["obsSharesLag1"], dtype=np.float64)
    array_num_time_steps = np.asarray(
        dict_stan_data["array_num_time_steps"], dtype=np.int64
    )

    dict_pymc_data = {
        "num_videos": num_videos,
        "max_num_time_steps": max_num_time_steps,
        "num_lags": num_lags,
        "M": dict_stan_data["M"],
        "array_num_time_steps": array_num_time_steps,
        "obsViews": obs_views,
        "obsViewsLags": obs_views_lags,
        "obsSharesLag1": obs_shares_lag1,
    }

    return dict_pymc_data


def construct_pymc_model_standard(num_trajectories_retained):
    dict_pymc_data = construct_pymc_input_data_standard(num_trajectories_retained)

    with pm.Model() as model:
        gamma = pm.Beta("gamma", alpha=1, beta=20)
        a = pm.Normal("a", mu=5, sigma=np.sqrt(10))
        b = pm.Beta("b", alpha=1, beta=10, shape=dict_pymc_data["num_lags"])
        c = pm.Beta("c", alpha=1, beta=1)
        sigma2 = pm.Exponential("sigma2", lam=1)
        qualities = pm.Beta(
            "qualities", alpha=1, beta=1, shape=dict_pymc_data["num_videos"]
        )

        obs_views = dict_pymc_data["obsViews"]
        obs_views_lags = dict_pymc_data["obsViewsLags"]
        obs_shares_lag1 = dict_pymc_data["obsSharesLag1"]
        array_num_time_steps = dict_pymc_data["array_num_time_steps"]

        for i in range(dict_pymc_data["num_videos"]):
            num_time_steps = array_num_time_steps[i]
            for t in range(num_time_steps):
                lagged_linear_predictor = 0.0
                for lag_id in range(dict_pymc_data["num_lags"]):
                    lagged_view_index = t + dict_pymc_data["num_lags"] - 1 - lag_id
                    lagged_linear_predictor += (
                        b[lag_id] * obs_views_lags[i, lagged_view_index]
                    )

                prob = pm.math.sigmoid(-a + lagged_linear_predictor)
                mu = qualities[i] * (
                    pt.exp(-gamma * t) * dict_pymc_data["M"] * prob
                    + c * obs_shares_lag1[i, t]
                )
                current_lagged_view_index = t + dict_pymc_data["num_lags"] - 1
                var_view = sigma2 * (obs_views_lags[i, current_lagged_view_index] + 1)

                pm.Normal(
                    "obsViews_{}_{}".format(i, t),
                    mu=mu,
                    sigma=pt.sqrt(var_view),
                    observed=obs_views[i, t],
                )

    return model, dict_pymc_data


def extract_valid_lagged_views(
    obs_views_lags, num_lags, max_num_time_steps, observed_mask
):
    # In Stan, the lagged terms are obsViewsLags[i][t+4], ..., obsViewsLags[i][t]
    # for num_lags = 5 and 1-based indexing. The Python slices below reproduce
    # the same ordering in 0-based indexing, with column 0 corresponding to the
    # current-time lag term obsViewsLags[i][t+4] used in both mu and var_view.
    list_valid_lagged_views = []
    for lag_id in range(num_lags):
        lag_start = num_lags - 1 - lag_id
        lag_stop = lag_start + max_num_time_steps
        lagged_views_matrix = obs_views_lags[:, lag_start:lag_stop]
        valid_lagged_views = lagged_views_matrix[observed_mask]
        list_valid_lagged_views.append(valid_lagged_views)

    valid_all_lag_views = np.stack(list_valid_lagged_views, axis=1)
    return valid_all_lag_views


# Construct a vectorized version of the standard PyMC probabilistic model
# implemented above. The vectorized version compiles faster than the standard
# version.


def construct_pymc_input_data_vectorized(num_trajectories_retained):
    dict_stan_data = construct_stan_input_data(num_trajectories_retained)

    num_videos = dict_stan_data["num_videos"]
    max_num_time_steps = dict_stan_data["max_num_time_steps"]
    num_lags = dict_stan_data["num_lags"]

    obs_views = np.asarray(dict_stan_data["obsViews"], dtype=np.float64)
    obs_views_lags = np.asarray(dict_stan_data["obsViewsLags"], dtype=np.float64)
    obs_shares_lag1 = np.asarray(dict_stan_data["obsSharesLag1"], dtype=np.float64)
    array_num_time_steps = np.asarray(
        dict_stan_data["array_num_time_steps"], dtype=np.int64
    )

    # The original Stan inputs stay as rectangular matrices with zero padding.
    # We construct a mask for valid (video, time) cells and then flatten only
    # the valid observations so the PyMC likelihood can stay vectorized.
    video_ids_matrix = np.broadcast_to(
        np.arange(num_videos, dtype=np.int64)[:, None],
        (num_videos, max_num_time_steps),
    )
    time_indices_matrix = np.broadcast_to(
        np.arange(max_num_time_steps, dtype=np.int64), (num_videos, max_num_time_steps)
    )
    observed_mask = time_indices_matrix < (array_num_time_steps[:, None])

    valid_video_indices = video_ids_matrix[observed_mask]
    valid_time_indices = time_indices_matrix[observed_mask]
    valid_obs_views = obs_views[observed_mask]

    obs_shares_lag1_matrix = obs_shares_lag1[:, :max_num_time_steps]
    valid_obs_shares_lag1 = obs_shares_lag1_matrix[observed_mask]

    valid_all_lag_views = extract_valid_lagged_views(
        obs_views_lags, num_lags, max_num_time_steps, observed_mask
    )
    valid_current_lag_views = valid_all_lag_views[:, 0]

    dict_pymc_data = {
        "num_videos": num_videos,
        "max_num_time_steps": max_num_time_steps,
        "num_lags": num_lags,
        "M": dict_stan_data["M"],
        "video_indices": valid_video_indices,
        "time_indices": valid_time_indices,
        "views_observed": valid_obs_views,
        "shares_lag1_observed": valid_obs_shares_lag1,
        "views_lagged_observed": valid_all_lag_views,
        "current_lag_views_observed": valid_current_lag_views,
    }

    return dict_pymc_data


def construct_pymc_model_vectorized(num_trajectories_retained):
    dict_pymc_data = construct_pymc_input_data_vectorized(num_trajectories_retained)

    with pm.Model() as model:
        gamma = pm.Beta("gamma", alpha=1, beta=20)
        a = pm.Normal("a", mu=5, sigma=np.sqrt(10))
        b = pm.Beta("b", alpha=1, beta=10, shape=dict_pymc_data["num_lags"])
        c = pm.Beta("c", alpha=1, beta=1)
        sigma2 = pm.Exponential("sigma2", lam=1)
        qualities = pm.Beta(
            "qualities", alpha=1, beta=1, shape=dict_pymc_data["num_videos"]
        )

        video_indices = pt.as_tensor_variable(dict_pymc_data["video_indices"])
        time_indices = pt.as_tensor_variable(dict_pymc_data["time_indices"])
        views_lagged_observed = pt.as_tensor_variable(
            dict_pymc_data["views_lagged_observed"]
        )
        shares_lag1_observed = pt.as_tensor_variable(
            dict_pymc_data["shares_lag1_observed"]
        )

        lagged_linear_predictor = pt.sum(b * views_lagged_observed, axis=1)
        prob = pm.math.sigmoid(-a + lagged_linear_predictor)
        mu = qualities[video_indices] * (
            pt.exp(-gamma * time_indices) * dict_pymc_data["M"] * prob
            + c * shares_lag1_observed
        )
        current_lag_views_observed = pt.as_tensor_variable(
            dict_pymc_data["current_lag_views_observed"]
        )
        var_view = sigma2 * (current_lag_views_observed + 1)

        pm.Normal(
            "views",
            mu=mu,
            sigma=pt.sqrt(var_view),
            observed=dict_pymc_data["views_observed"],
        )

    return model, dict_pymc_data


def print_model_sanity_check(num_trajectories_retained):
    model, dict_pymc_data = construct_pymc_model_standard(num_trajectories_retained)

    print("Constructed PyMC model")
    print("  num_videos: {}".format(dict_pymc_data["num_videos"]))
    print("  max_num_time_steps: {}".format(dict_pymc_data["max_num_time_steps"]))
    print("  num_lags: {}".format(dict_pymc_data["num_lags"]))
    print(
        "  valid_observations: {}".format(
            np.sum(dict_pymc_data["array_num_time_steps"])
        )
    )
    print("  obsViews shape: {}".format(dict_pymc_data["obsViews"].shape))
    print("  obsViewsLags shape: {}".format(dict_pymc_data["obsViewsLags"].shape))
    print("  obsSharesLag1 shape: {}".format(dict_pymc_data["obsSharesLag1"].shape))
    print("  b shape: ({},)".format(dict_pymc_data["num_lags"]))
    print("  qualities shape: ({},)".format(dict_pymc_data["num_videos"]))
    print("  model variables: {}".format(list(model.named_vars.keys())))


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]

    if mode == "sanity_check":
        num_trajectories_retained = 1500
        if len(args) >= 2:
            num_trajectories_retained = int(args[1])

        print_model_sanity_check(num_trajectories_retained)
    else:
        raise ValueError("Mode {} is not supported".format(mode))
