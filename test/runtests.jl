using BetaRegression
using GLM
using StatsBase
using StatsModels
using Test

using GLM: linkinv
using BetaRegression: 🐟, dmueta

@testset "Basics" begin
    @test_throws ArgumentError BetaRegressionModel([1 2 3; 4 5 6], [1, 2])
    @test_throws DimensionMismatch BetaRegressionModel([1 2; 3 4; 5 6], [1, 2])
    @test_throws ArgumentError BetaRegressionModel([1 2; 3 4; 5 6], [1, 2, 3])
    @test_throws ArgumentError BetaRegressionModel([1 2; 3 4; 5 6], [0.1, 0.2, 0.3];
                                                   weights=[1])
    @test_throws ArgumentError BetaRegressionModel([1 2; 3 4; 5 6], [0.1, 0.2, 0.3];
                                                   offset=[1])
    X = [1 2; 3 4; 5 6]
    y = [0.1, 0.2, 0.3]
    b = BetaRegressionModel(X, y, CauchitLink())
    @test b isa BetaRegressionModel{Float64,CauchitLink,Vector{Float64},Matrix{Float64}}
    @test response(b) === y
    @test modelmatrix(b) == X
    @test Link(b) == CauchitLink()
    @test nobs(b) == 3
    @test coef(b) == [0, 0]
    @test dispersion(b) == 0
    fit!(b)
    @test coef(b) != [0, 0]
    @test dispersion(b) > 0
    X = ones(Int, 3, 1)
    y .= 0.5
    b = BetaRegressionModel(X, y, CauchitLink())
    @test_throws ConvergenceException fit!(b)
end

@testset "Example: Food expenditure data (Ferrari table 2)" begin
    expenditure = [15.998 62.476 1
                   16.652 82.304 5
                   21.741 74.679 3
                    7.431 39.151 3
                   10.481 64.724 5
                   13.548 36.786 3
                   23.256 83.052 4
                   17.976 86.935 1
                   14.161 88.233 2
                    8.825 38.695 2
                   14.184 73.831 7
                   19.604 77.122 3
                   13.728 45.519 2
                   21.141 82.251 2
                   17.446 59.862 3
                    9.629 26.563 3
                   14.005 61.818 2
                    9.160 29.682 1
                   18.831 50.825 5
                    7.641 71.062 4
                   13.882 41.990 4
                    9.670 37.324 3
                   21.604 86.352 5
                   10.866 45.506 2
                   28.980 69.929 6
                   10.882 61.041 2
                   18.561 82.469 1
                   11.629 44.208 2
                   18.067 49.467 5
                   14.539 25.905 5
                   19.192 79.178 5
                   25.918 75.811 3
                   28.833 82.718 6
                   15.869 48.311 4
                   14.910 42.494 5
                    9.550 40.573 4
                   23.066 44.872 6
                   14.751 27.167 7]
    data = (; food=expenditure[:, 1], income=expenditure[:, 2], persons=expenditure[:, 3])
    model = fit(BetaRegressionModel, @formula((food / income) ~ 1 + income + persons), data)
    @test responsename(model) == ":(food / income)"
    @test coefnames(model) == ["(Intercept)", "income", "persons", "(Dispersion)"]
    @test coef(model) ≈ [-0.62255, -0.01230, 0.11846] atol=1e-5
    @test dispersion(model) ≈ 35.60975 atol=1e-5
    @test stderror(model) ≈ [0.22385, 0.00304, 0.03534, 8.07960] atol=1e-5
    @test Link(model) == LogitLink()
    @test isempty(weights(model))
    @test informationmatrix(model) \ score(model) ≈ zeros(4) atol=1e-6
    for expected in false:true
        @test inv(informationmatrix(model; expected)) ≈ 🐟(model.model, expected, true) atol=1e-10
    end
    ct = coeftable(model)
    @test ct isa CoefTable
    @test ct.rownms == coefnames(model)
    @test ct.colnms == ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
    @test all(<(0.05), ct.cols[ct.pvalcol])  # Significant! Time to publish!
    @test confint(model) ≈ [-1.06129293 -0.183803181
                            -0.01824848 -0.006349204
                             0.04919566  0.187728532
                            19.77402875 51.445471904] atol=1e-8
    @test loglikelihood(model) ≈ 45.33351 atol=1e-5
    @test dof(model) == 4
    @test dof_residual(model) == 34
    model_with_offset = fit(BetaRegressionModel, formula(model), data; offset=ones(nobs(model)))
    @test first(coef(model_with_offset)) ≈ first(coef(model)) - 1 atol=1e-8
    @test coef(model_with_offset)[2:end] ≈ coef(model)[2:end] atol=1e-8
end

@testset "Example: Prater's gasoline data (Ferrari table 1)" begin
    X = [1  1  0  0  0  0  0  0  0  0  205
         1  1  0  0  0  0  0  0  0  0  275
         1  1  0  0  0  0  0  0  0  0  345
         1  1  0  0  0  0  0  0  0  0  407
         1  0  1  0  0  0  0  0  0  0  218
         1  0  1  0  0  0  0  0  0  0  273
         1  0  1  0  0  0  0  0  0  0  347
         1  0  0  1  0  0  0  0  0  0  212
         1  0  0  1  0  0  0  0  0  0  272
         1  0  0  1  0  0  0  0  0  0  340
         1  0  0  0  1  0  0  0  0  0  235
         1  0  0  0  1  0  0  0  0  0  300
         1  0  0  0  1  0  0  0  0  0  365
         1  0  0  0  1  0  0  0  0  0  410
         1  0  0  0  0  1  0  0  0  0  307
         1  0  0  0  0  1  0  0  0  0  367
         1  0  0  0  0  1  0  0  0  0  395
         1  0  0  0  0  0  1  0  0  0  267
         1  0  0  0  0  0  1  0  0  0  360
         1  0  0  0  0  0  1  0  0  0  402
         1  0  0  0  0  0  0  1  0  0  235
         1  0  0  0  0  0  0  1  0  0  275
         1  0  0  0  0  0  0  1  0  0  358
         1  0  0  0  0  0  0  1  0  0  416
         1  0  0  0  0  0  0  0  1  0  285
         1  0  0  0  0  0  0  0  1  0  365
         1  0  0  0  0  0  0  0  1  0  444
         1  0  0  0  0  0  0  0  0  1  351
         1  0  0  0  0  0  0  0  0  1  424
         1  0  0  0  0  0  0  0  0  0  365
         1  0  0  0  0  0  0  0  0  0  379
         1  0  0  0  0  0  0  0  0  0  428]
    y = [0.122, 0.223, 0.347, 0.457, 0.080, 0.131, 0.266, 0.074, 0.182, 0.304, 0.069,
         0.152, 0.260, 0.336, 0.144, 0.268, 0.349, 0.100, 0.248, 0.317, 0.028, 0.064,
         0.161, 0.278, 0.050, 0.176, 0.321, 0.140, 0.232, 0.085, 0.147, 0.180]
    model = fit(BetaRegressionModel, X, y)
    @test coef(model) ≈ [-6.15957, 1.72773, 1.32260, 1.57231, 1.05971, 1.13375,
                         1.04016, 0.54369, 0.49590, 0.38579, 0.01097] atol=1e-5
    @test dispersion(model) ≈ 440.27838 atol=1e-5
    @test stderror(model) ≈ [0.18232, 0.10123, 0.11790, 0.11610, 0.10236, 0.10352,
                             0.10604, 0.10913, 0.10893, 0.11859, 0.00041, 110.02562] atol=1e-4
end

@testset "dmueta" begin
    centraldiff(f, x, h=0.001) = (f(x + h) - 2 * f(x) + f(x - h)) / h^2
    centraldiff(link::Link, x, h=0.001) = centraldiff(η -> linkinv(link, η), x, h)
    x = range(0, 1; step=0.001)
    @testset "$L" for L in [CauchitLink, CloglogLink, LogitLink, ProbitLink]
        @test dmueta.(L(), x) ≈ centraldiff.(L(), x) atol=1e-5
    end
end
