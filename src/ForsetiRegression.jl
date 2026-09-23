"""
    ForsetiRegression

Linear and logistic regression, named `linear_reg`/`logistic_reg` to match
tidymodels' (parsnip) model-spec naming.

Both follow the family-wide pipe-curried calling convention and return a
`ForsetiFit` subtype usable with [`tidy`](@ref)/[`glance`](@ref)/[`augment`](@ref):

```julia
df |> linear_reg(:y, :x1, :x2) |> tidy
df |> logistic_reg(:outcome, :x1, :x2) |> glance
```
"""
module ForsetiRegression

using DataFrames
using Statistics
using LinearAlgebra
using Distributions
using ForsetiCore

export LMFit, linear_reg
export GLMFit, logistic_reg
export tidy, glance, augment, predict

"""
    predict(fit::ForsetiFit, newdata)

Return model predictions for `newdata` (fitted values for [`LMFit`](@ref),
predicted probabilities for [`GLMFit`](@ref)).
"""
function predict end

function _design_matrix(df::AbstractDataFrame, x_cols; intercept::Bool = true)
    isempty(x_cols) && !intercept &&
        throw(ArgumentError("must have an intercept or at least one predictor"))
    cols = [Float64.(df[!, c]) for c in x_cols]
    X = isempty(cols) ? Matrix{Float64}(undef, nrow(df), 0) : reduce(hcat, cols)
    intercept && (X = hcat(ones(nrow(df)), X))
    return X
end

function _term_names(x_cols; intercept::Bool = true)
    names = String[string(c) for c in x_cols]
    return intercept ? vcat(["(Intercept)"], names) : names
end

# ---------------------------------------------------------------------------
# Linear regression (OLS)
# ---------------------------------------------------------------------------

"""
    LMFit <: ForsetiFit

Result of [`linear_reg`](@ref) (ordinary least squares).
"""
struct LMFit <: ForsetiFit
    coef::Vector{Float64}
    se::Vector{Float64}
    terms::Vector{String}
    fitted::Vector{Float64}
    resid::Vector{Float64}
    r_squared::Float64
    adj_r_squared::Float64
    sigma::Float64
    f_statistic::Float64
    f_df::Int
    f_df_residual::Int
    f_pvalue::Float64
    n::Int
    df_residual::Int
    data::DataFrame
    y_col::Symbol
    x_cols::Vector{Symbol}
    intercept::Bool
end

"""
    linear_reg(y::AbstractVector, X::AbstractMatrix; intercept = true) -> LMFit

Fit `y ~ X` by ordinary least squares. `X` should *not* include an
intercept column; one is added automatically unless `intercept = false`.

# Formula

```
β̂ = (XᵀX)⁻¹Xᵀy                        (solved via QR, not explicit inversion)
σ̂² = RSS / (n-p),   RSS = Σ(y - Xβ̂)²
Var(β̂) = σ̂² (XᵀX)⁻¹
t = β̂ⱼ / SE(β̂ⱼ),   df = n-p
F = (SS_reg/(p-1)) / (RSS/(n-p))       (overall-model test)
R² = 1 - RSS/SS_tot
```

# References

Gauss-Markov theorem; e.g. Draper, N. R., & Smith, H. (1998). *Applied
Regression Analysis* (3rd ed.). Wiley.
"""
function linear_reg(y::AbstractVector, X::AbstractMatrix; intercept::Bool = true,
                     terms::Vector{String} = intercept ?
                         vcat(["(Intercept)"], ["x$i" for i in 1:size(X, 2)]) :
                         ["x$i" for i in 1:size(X, 2)],
                     data::DataFrame = DataFrame(), y_col::Symbol = :y,
                     x_cols::Vector{Symbol} = Symbol[])
    Xd = intercept ? hcat(ones(size(X, 1)), X) : Matrix{Float64}(X)
    n, p = size(Xd)
    n > p || throw(ArgumentError("need more observations ($n) than parameters ($p)"))
    yv = Float64.(y)

    coef = Xd \ yv
    fitted = Xd * coef
    resid = yv .- fitted
    df_residual = n - p

    rss = sum(abs2, resid)
    tss = sum(abs2, yv .- mean(yv))
    r_squared = 1 - rss / tss
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / df_residual
    sigma2 = rss / df_residual
    sigma = sqrt(sigma2)

    XtX_inv = inv(Xd' * Xd)
    se = sqrt.(sigma2 .* diag(XtX_inv))

    f_df = intercept ? p - 1 : p
    f_df_residual = df_residual
    if f_df > 0
        ss_reg = tss - rss
        f_statistic = (ss_reg / f_df) / sigma2
        f_pvalue = ccdf(FDist(f_df, f_df_residual), f_statistic)
    else
        f_statistic = NaN
        f_pvalue = NaN
    end

    return LMFit(coef, se, terms, fitted, resid, r_squared, adj_r_squared, sigma,
                 f_statistic, f_df, f_df_residual, f_pvalue, n, df_residual,
                 data, y_col, x_cols, intercept)
end

"""
    linear_reg(df::AbstractDataFrame, y_col::Symbol, x_cols::Symbol...; intercept = true) -> LMFit

Fit `y_col ~ x_cols...` on `df` by ordinary least squares.
"""
function linear_reg(df::AbstractDataFrame, y_col::Symbol, x_cols::Symbol...; intercept::Bool = true)
    X = _design_matrix(df, x_cols; intercept = false)
    terms = _term_names(x_cols; intercept = intercept)
    return linear_reg(df[!, y_col], X; intercept = intercept, terms = terms,
                       data = DataFrame(df), y_col = y_col, x_cols = collect(x_cols))
end

"""
    linear_reg(y_col::Symbol, x_cols::Symbol...; kwargs...)

Pipe-curried form: `df |> linear_reg(:y, :x1, :x2)`.
"""
linear_reg(y_col::Symbol, x_cols::Symbol...; kwargs...) =
    df -> linear_reg(df, y_col, x_cols...; kwargs...)

function ForsetiCore.tidy(fit::LMFit)
    tcrit_p = 2 .* ccdf.(TDist(fit.df_residual), abs.(fit.coef ./ fit.se))
    crit = quantile(TDist(fit.df_residual), 0.975)
    return DataFrame(
        term = fit.terms,
        estimate = fit.coef,
        std_error = fit.se,
        statistic = fit.coef ./ fit.se,
        p_value = tcrit_p,
        conf_low = fit.coef .- crit .* fit.se,
        conf_high = fit.coef .+ crit .* fit.se,
    )
end

function ForsetiCore.glance(fit::LMFit)
    return DataFrame(
        r_squared = [fit.r_squared],
        adj_r_squared = [fit.adj_r_squared],
        sigma = [fit.sigma],
        statistic = [fit.f_statistic],
        p_value = [fit.f_pvalue],
        df = [fit.f_df],
        df_residual = [fit.f_df_residual],
        n = [fit.n],
    )
end

function ForsetiCore.augment(fit::LMFit, data::AbstractDataFrame = fit.data)
    out = DataFrame(data)
    out[!, :fitted] = fit.fitted
    out[!, :resid] = fit.resid
    return out
end

function predict(fit::LMFit, newdata::AbstractDataFrame)
    X = _design_matrix(newdata, fit.x_cols; intercept = fit.intercept)
    return X * fit.coef
end
predict(fit::LMFit) = fit.fitted

# ---------------------------------------------------------------------------
# Logistic regression (binomial GLM, logit link, fit by IRLS)
# ---------------------------------------------------------------------------

"""
    GLMFit <: ForsetiFit

Result of [`logistic_reg`](@ref) (binomial GLM with a logit link, fit by
iteratively reweighted least squares).
"""
struct GLMFit <: ForsetiFit
    coef::Vector{Float64}
    se::Vector{Float64}
    terms::Vector{String}
    fitted::Vector{Float64}
    resid::Vector{Float64}
    deviance::Float64
    null_deviance::Float64
    df_residual::Int
    df_null::Int
    aic::Float64
    n::Int
    iterations::Int
    converged::Bool
    data::DataFrame
    y_col::Symbol
    x_cols::Vector{Symbol}
    intercept::Bool
end

_logistic(eta) = 1 / (1 + exp(-eta))

function _binomial_deviance(y, mu)
    eps_ = 1e-10
    mu_c = clamp.(mu, eps_, 1 - eps_)
    return -2 * sum(y .* log.(mu_c) .+ (1 .- y) .* log.(1 .- mu_c))
end

"""
    logistic_reg(y::AbstractVector, X::AbstractMatrix; intercept = true,
                 max_iter = 25, tol = 1e-8) -> GLMFit

Fit a binomial GLM with a logit link (`y` must be 0/1) by IRLS. `X` should
*not* include an intercept column; one is added automatically unless
`intercept = false`.

# Formula

Iteratively reweighted least squares (IRLS): at each iteration, with
current fit `η = Xβ`, `μ = 1/(1+e^{-η})`, `w = μ(1-μ)`, and working
response `z = η + (y-μ)/w`, solve the weighted least squares problem

```
β_new = (XᵀWX)⁻¹XᵀWz,   W = diag(w)
```

until `β` converges. Standard errors come from the Fisher information at
convergence, `Var(β̂) = (XᵀWX)⁻¹`. Deviance = `-2 * Σ[y log μ + (1-y)
log(1-μ)]`; `null_deviance` is the same formula with `μ` fixed at
`mean(y)`.

Cross-checked against R's `glm(family=binomial)`: coefficients and
deviance match to ~8 significant figures; standard errors can differ by
~1e-4 for some data because R's `summary.glm()` reuses the weight matrix
from its second-to-last IRLS iterate rather than recomputing it at the
converged coefficients (both are asymptotically valid) — see
`ForsetiTutorials.jl`'s R cross-validation tutorial for the traced
example.

# References

- Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized linear
  models. *Journal of the Royal Statistical Society: Series A*, 135(3),
  370–384.
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*
  (2nd ed.). Chapman & Hall.
"""
function logistic_reg(y::AbstractVector, X::AbstractMatrix; intercept::Bool = true,
                       max_iter::Int = 25, tol::Real = 1e-8,
                       terms::Vector{String} = intercept ?
                           vcat(["(Intercept)"], ["x$i" for i in 1:size(X, 2)]) :
                           ["x$i" for i in 1:size(X, 2)],
                       data::DataFrame = DataFrame(), y_col::Symbol = :y,
                       x_cols::Vector{Symbol} = Symbol[])
    Xd = intercept ? hcat(ones(size(X, 1)), X) : Matrix{Float64}(X)
    n, p = size(Xd)
    yv = Float64.(y)
    all(v -> v == 0.0 || v == 1.0, yv) || throw(ArgumentError("logistic_reg requires a 0/1 response"))

    beta = zeros(p)
    eps_ = 1e-10
    converged = false
    iter = 0
    local mu, w
    for i in 1:max_iter
        iter = i
        eta = Xd * beta
        mu = _logistic.(eta)
        mu_c = clamp.(mu, eps_, 1 - eps_)
        w = mu_c .* (1 .- mu_c)
        z = eta .+ (yv .- mu_c) ./ w
        W = Diagonal(w)
        beta_new = (Xd' * W * Xd) \ (Xd' * W * z)
        if maximum(abs.(beta_new .- beta)) < tol
            beta = beta_new
            converged = true
            break
        end
        beta = beta_new
    end

    eta = Xd * beta
    mu = _logistic.(eta)
    resid = yv .- mu
    mu_c = clamp.(mu, eps_, 1 - eps_)
    w = mu_c .* (1 .- mu_c)
    XtWX_inv = inv(Xd' * Diagonal(w) * Xd)
    se = sqrt.(diag(XtWX_inv))

    deviance = _binomial_deviance(yv, mu)
    p0 = clamp(mean(yv), eps_, 1 - eps_)
    null_deviance = _binomial_deviance(yv, fill(p0, n))
    df_residual = n - p
    df_null = n - 1
    aic = deviance + 2 * p

    return GLMFit(beta, se, terms, mu, resid, deviance, null_deviance, df_residual,
                  df_null, aic, n, iter, converged, data, y_col, x_cols, intercept)
end

"""
    logistic_reg(df::AbstractDataFrame, y_col::Symbol, x_cols::Symbol...; kwargs...) -> GLMFit

Fit `y_col ~ x_cols...` on `df` as a binomial GLM (logit link).
"""
function logistic_reg(df::AbstractDataFrame, y_col::Symbol, x_cols::Symbol...; intercept::Bool = true, kwargs...)
    X = _design_matrix(df, x_cols; intercept = false)
    terms = _term_names(x_cols; intercept = intercept)
    return logistic_reg(df[!, y_col], X; intercept = intercept, terms = terms,
                         data = DataFrame(df), y_col = y_col, x_cols = collect(x_cols), kwargs...)
end

"""
    logistic_reg(y_col::Symbol, x_cols::Symbol...; kwargs...)

Pipe-curried form: `df |> logistic_reg(:y, :x1, :x2)`.
"""
logistic_reg(y_col::Symbol, x_cols::Symbol...; kwargs...) =
    df -> logistic_reg(df, y_col, x_cols...; kwargs...)

function ForsetiCore.tidy(fit::GLMFit)
    z = fit.coef ./ fit.se
    return DataFrame(
        term = fit.terms,
        estimate = fit.coef,
        std_error = fit.se,
        statistic = z,
        p_value = 2 .* ccdf.(Normal(), abs.(z)),
    )
end

function ForsetiCore.glance(fit::GLMFit)
    return DataFrame(
        deviance = [fit.deviance],
        null_deviance = [fit.null_deviance],
        df_residual = [fit.df_residual],
        df_null = [fit.df_null],
        aic = [fit.aic],
        n = [fit.n],
        iterations = [fit.iterations],
        converged = [fit.converged],
    )
end

function ForsetiCore.augment(fit::GLMFit, data::AbstractDataFrame = fit.data)
    out = DataFrame(data)
    out[!, :fitted] = fit.fitted
    out[!, :resid] = fit.resid
    return out
end

function predict(fit::GLMFit, newdata::AbstractDataFrame)
    X = _design_matrix(newdata, fit.x_cols; intercept = fit.intercept)
    return _logistic.(X * fit.coef)
end
predict(fit::GLMFit) = fit.fitted

end # module ForsetiRegression
