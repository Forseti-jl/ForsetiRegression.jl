using Test
using ForsetiCore
using ForsetiRegression
using DataFrames
using Statistics
using LinearAlgebra
using Distributions

@testset "ForsetiRegression" begin

    @testset "linear_reg: simple regression, hand-derived" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 5.0, 4.0, 5.0]
        fit = linear_reg(y, reshape(x, :, 1))
        @test fit isa LMFit
        @test fit isa ForsetiFit

        # hand-derived: slope=0.6, intercept=2.2 (classic textbook OLS example)
        @test fit.coef ≈ [2.2, 0.6] atol = 1e-10
        @test fit.fitted ≈ [2.8, 3.4, 4.0, 4.6, 5.2] atol = 1e-10
        @test fit.resid ≈ [-0.8, 0.6, 1.0, -0.6, -0.2] atol = 1e-10
        @test fit.r_squared ≈ 0.6 atol = 1e-10
        @test fit.adj_r_squared ≈ 0.4666666667 atol = 1e-9
        @test fit.sigma ≈ sqrt(0.8) atol = 1e-10
        @test fit.se ≈ [sqrt(0.88), sqrt(0.08)] atol = 1e-9
        @test fit.f_statistic ≈ 4.5 atol = 1e-9
        @test fit.f_pvalue ≈ ccdf(FDist(1, 3), 4.5) atol = 1e-12

        t = tidy(fit)
        @test t.term == ["(Intercept)", "x1"]
        @test t.estimate ≈ [2.2, 0.6] atol = 1e-10
        crit = quantile(TDist(3), 0.975)
        @test t.conf_low[2] ≈ 0.6 - crit * sqrt(0.08) atol = 1e-9

        g = glance(fit)
        @test g.r_squared[1] ≈ 0.6 atol = 1e-10
        @test g.n[1] == 5
    end

    @testset "linear_reg: DataFrame/pipe form matches vector form" begin
        df = DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0], y = [2.0, 4.0, 5.0, 4.0, 5.0])
        fit = df |> linear_reg(:y, :x) |> identity
        @test fit.coef ≈ [2.2, 0.6] atol = 1e-10
        @test fit.terms == ["(Intercept)", "x"]

        a = df |> linear_reg(:y, :x) |> augment
        @test a.fitted ≈ [2.8, 3.4, 4.0, 4.6, 5.2] atol = 1e-10
        @test a.resid ≈ [-0.8, 0.6, 1.0, -0.6, -0.2] atol = 1e-10
        @test a.x == df.x  # original columns preserved
    end

    @testset "linear_reg: exact multi-predictor fit + predict" begin
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x2 = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]
        y = 1.0 .+ 2.0 .* x1 .- 1.0 .* x2   # exact linear relationship, no noise
        df = DataFrame(y = y, x1 = x1, x2 = x2)

        fit = linear_reg(df, :y, :x1, :x2)
        @test fit.coef ≈ [1.0, 2.0, -1.0] atol = 1e-8
        @test sum(abs2, fit.resid) ≈ 0.0 atol = 1e-8
        @test fit.r_squared ≈ 1.0 atol = 1e-8

        newdata = DataFrame(x1 = [10.0], x2 = [3.0])
        pred = predict(fit, newdata)
        @test pred[1] ≈ (1.0 + 2.0 * 10.0 - 1.0 * 3.0) atol = 1e-6
    end

    @testset "linear_reg: no intercept" begin
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 3.0, 7.0, 8.0]
        fit = linear_reg(y, reshape(x, :, 1); intercept = false)

        # hand-derived (no-intercept OLS): b = sum(x.*y)/sum(x.^2) = 61/30
        b = 61.0 / 30.0
        @test fit.coef ≈ [b] atol = 1e-10
        @test fit.fitted ≈ b .* x atol = 1e-10
        @test fit.intercept == false
    end

    @testset "logistic_reg: intercept-only reduces to closed-form logit(mean(y))" begin
        y = [0.0, 0.0, 1.0, 1.0, 1.0]  # mean = 0.6
        X = Matrix{Float64}(undef, 5, 0)
        fit = logistic_reg(y, X)
        @test fit isa GLMFit
        @test fit isa ForsetiFit
        @test fit.converged

        expected_beta0 = log(0.6 / 0.4)
        @test fit.coef[1] ≈ expected_beta0 atol = 1e-6
        @test all(fit.fitted .≈ 0.6)
    end

    @testset "logistic_reg: score equations vanish at the MLE" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        df = DataFrame(x = x, y = y)
        fit = df |> logistic_reg(:y, :x) |> identity
        @test fit.converged

        # first-order optimality condition for the MLE: X' * (y - mu) ≈ 0
        Xd = hcat(ones(length(x)), x)
        score = Xd' * (y .- fit.fitted)
        @test maximum(abs.(score)) < 1e-6

        # predictor is positively associated with the outcome here
        @test fit.coef[2] > 0

        # null deviance matches direct computation with p0 = mean(y)
        p0 = mean(y)
        expected_null_dev = -2 * sum(y .* log(p0) .+ (1 .- y) .* log(1 - p0))
        @test fit.null_deviance ≈ expected_null_dev atol = 1e-8

        # a genuinely predictive covariate should fit better than the null model
        @test fit.deviance < fit.null_deviance

        t = tidy(fit)
        @test t.term == ["(Intercept)", "x"]
        g = glance(fit)
        @test g.deviance[1] ≈ fit.deviance
        a = augment(fit)
        @test a.fitted ≈ fit.fitted

        pred = predict(fit, DataFrame(x = [4.5]))
        @test 0.0 <= pred[1] <= 1.0
    end

    @testset "logistic_reg: rejects non-0/1 response" begin
        @test_throws ArgumentError logistic_reg([0.0, 1.0, 2.0], reshape([1.0, 2.0, 3.0], :, 1))
    end

end
