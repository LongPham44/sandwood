data {
    int<lower=1> num_videos;
    int<lower=1> max_num_time_steps;
    array[num_videos] int<lower=1> array_num_time_steps;
    
    int<lower=1> num_lags;
    int<lower=1> M;

    matrix<lower=0>[num_videos, max_num_time_steps] obsViews;
    matrix<lower=0>[num_videos, max_num_time_steps + num_lags] obsViewsLags;
    matrix<lower=0>[num_videos, max_num_time_steps + 1] obsSharesLag1;
}
transformed data {
    real sqrt10 = sqrt(10);
}
parameters {
    real<lower=0, upper=1> gamma;
    real a;
    vector<lower=0, upper=1>[num_lags] b;
    real<lower=0, upper=1> c;
    real<lower=0> sigma2;
    vector<lower=0, upper=1>[num_videos] qualities;
}
model {
    // Prior distribution
    target += beta_lpdf(gamma | 1, 20);
    target += normal_lpdf(a | 5, sqrt10);
    target += beta_lpdf(b | 1, 10);
    target += beta_lpdf(c | 1, 1);
    target += exponential_lpdf(sigma2 | 1);
    target += beta_lpdf(qualities | 1, 1);

    int num_time_steps;
    real prob;
    real mu;
    real var_view;

    // Likelihood
    for (i in 1:num_videos) {
        num_time_steps = array_num_time_steps[i];
        for (t in 1:num_time_steps) {
            prob = inv_logit(- a + b[1]*obsViewsLags[i][t+4] + b[2]*obsViewsLags[i][t+3] + b[3]*obsViewsLags[i][t+2] + b[4]*obsViewsLags[i][t+1] + b[5]*obsViewsLags[i][t]);
            mu = qualities[i] * (exp(-gamma * (t-1)) * M * prob + c * obsSharesLag1[i][t]);
            var_view = sigma2 * (obsViewsLags[i][t+4] + 1);
            target += normal_lpdf(obsViews[i][t] | mu, sqrt(var_view));
        }
    }
}
