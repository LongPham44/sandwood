data {
  int<lower=0> N;
  vector[N] earn;
  vector[N] height;
  vector[N] male;
}
transformed data {
  vector[N] log_earn; // log transformation
  vector[N] inter; // interaction
  log_earn = log(earn);
  inter = height .* male;
}
parameters {
  vector[4] beta;
  real<lower=0> sigma;
}
model {
  // The priors of beta and sigma were missing in the original Stan model. I
  // added them so that the model can be translated to Sandwood.
  beta ~ normal(0, 10);
  sigma ~ cauchy(0, 10);

  log_earn ~ normal(beta[1] + beta[2] * height + beta[3] * male
                    + beta[4] * inter, sigma);
}
