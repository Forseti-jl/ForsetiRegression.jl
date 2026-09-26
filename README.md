# ForsetiRegression.jl

Linear, logistic, and generalized linear model regression.

Part of the [Forseti](https://github.com/Forseti-jl) statistical analysis package family.

Named `linear_reg`/`logistic_reg` to match tidymodels' (parsnip) model
naming.

## Usage

```julia
using ForsetiCore, ForsetiRegression, DataFrames

df = DataFrame(x1 = [1.0, 2.0, 3.0, 4.0, 5.0], x2 = [2.0, 1.0, 4.0, 3.0, 6.0],
               y = [2.0, 4.0, 5.0, 4.0, 5.0], outcome = [0.0, 0.0, 1.0, 1.0, 1.0])

# ordinary least squares
df |> linear_reg(:y, :x1) |> tidy       # term, estimate, std_error, statistic, p_value, CI
df |> linear_reg(:y, :x1) |> glance     # r_squared, adj_r_squared, sigma, F statistic, p_value
df |> linear_reg(:y, :x1) |> augment    # data + fitted + resid

fit = linear_reg(df, :y, :x1, :x2)
predict(fit, DataFrame(x1 = [10.0], x2 = [3.0]))

# logistic regression (binomial GLM, logit link, fit by IRLS)
df |> logistic_reg(:outcome, :x1) |> tidy     # term, estimate, std_error, z, p_value
df |> logistic_reg(:outcome, :x1) |> glance   # deviance, null_deviance, aic, converged
```

## Dependencies

This package depends on the following sibling Forseti packages, which are not
yet registered and must be added via local dev paths:

- `ForsetiCore`

## Local development

```julia
using Pkg
Pkg.develop(path="../ForsetiCore.jl")
Pkg.instantiate()
Pkg.test()
```
