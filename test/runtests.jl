using BetaRegression
using GLM
using StatsBase
using Test

using GLM: linkinv
using BetaRegression: 🐟, dmueta

# NOTE: Where it makes sense (and is possible) to do so, values are tested against
# `betareg(formula, data, link="logit", link.phi="identity", type="ML")` in R with
# an absolute tolerance of 1e-5 or better

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
    @test precision(b) == 0
    @test_throws ConvergenceException fit!(b; maxiter=0)
    fit!(b)
    @test coef(b) != [0, 0]
    @test precision(b) > 0
    @test startswith(sprint(show, b),
                     """
                     BetaRegressionModel{Float64,CauchitLink}
                         3 observations
                         3 degrees of freedom

                     Coefficients:
                     """)
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
    @test coefnames(model) == ["(Intercept)", "income", "persons", "(Precision)"]
    @test coef(model) ≈ [-0.62255, -0.01230, 0.11846] atol=1e-5
    @test precision(model) ≈ 35.60975 atol=1e-5
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
    @test loglikelihood.(Ref(model), 1:nobs(model)) ≈ loglikelihood(model, :) atol=1e-10
    @test sum(loglikelihood(model, :)) ≈ loglikelihood(model) atol=1e-10
    @test dof(model) == 4
    @test dof_residual(model) == 34
    @test fitted(model) ≈ [0.218835,  0.2606715, 0.234042,  0.3211104, 0.3044348,
                           0.327484,  0.2368291, 0.1717485, 0.186823,  0.2970203,
                           0.3315132, 0.2286988, 0.2797998, 0.1982583, 0.2682745,
                           0.3557506, 0.2412363, 0.2954333, 0.3417903, 0.2645057,
                           0.3395886, 0.3260284, 0.2511916, 0.279832,  0.3160853,
                           0.2429898, 0.1797033, 0.2830605, 0.3455576, 0.413664,
                           0.2681488, 0.2315554, 0.2831051, 0.3223771, 0.3652001,
                           0.3435079, 0.3861205, 0.4681841] atol=1e-5
    @test residuals(model) ≈ [ 0.03723132,  -0.05834841,  0.05708404,  -0.1313068,   -0.1425011,
                               0.04080833,   0.0431882,   0.03502668,  -0.02632745,  -0.06895467,
                              -0.1393988,    0.02549582,  0.02178853,   0.05877201,   0.02316252,
                               0.006746076, -0.01468414,  0.01317129,   0.02871633,  -0.1569799,
                              -0.0089861,   -0.06694573, -0.001006331, -0.04105034,   0.09833504,
                              -0.06471616,   0.0453631,  -0.02000852,   0.01967581,   0.147579,
                              -0.02575824,   0.1103211,   0.06546472,   0.006098814, -0.01432701,
                              -0.1081297,    0.1279194,   0.07479088] atol=1e-5
    @test devresid(model) ≈ [ 0.615805,   -0.7227725,  0.8551208,  -1.765633,  -1.999315,
                              0.5614324,   0.6729245,  0.6522058,  -0.227531,  -0.853578,
                             -1.870453,    0.4459616,  0.3559293,   0.9325464,  0.3819149,
                              0.1243324,  -0.03998864, 0.2334174,   0.4051461, -2.468176,
                             -0.02544287, -0.8080014, -0.05045462, -0.4610732,  1.25752,
                             -0.8383246,   0.7847574, -0.1597831,   0.2920577,  1.770506,
                             -0.2402128,   1.515325,   0.8997969,   0.1262185, -0.1176242,
                             -1.374193,    1.549423,   0.8918376] atol=1e-5
    @test deviance(model) ≈ 37.18054 atol=1e-5
    @test vcov(model) ≈ [ 0.0501104   -0.000519915 -0.00462328 -0.0343372
                         -0.000519915  9.21477e-6  -2.46623e-7 -0.000933215
                         -0.00462328  -2.46623e-7   0.00124896  0.00894725
                         -0.0343372   -0.000933215  0.00894725 65.2799] atol=1e-5
    @test r²(model) ≈ 0.3878327 atol=1e-6
    @test r2(model) === r²(model)
    @test aic(model) ≈ -82.66702 atol=1e-5
    k = dof(model)
    @test aicc(model) ≈ aic(model) + 2 * k * (k + 1) / (nobs(model) - k - 1)
    @test bic(model) ≈ -76.11667 atol=1e-5
    @test predict(model) == fitted(model)
    newobs = [1.0 58.44434 3.578947]
    @test predict(model, newobs) ≈ [0.2854928] atol=1e-6
    model_with_offset = fit(BetaRegressionModel, formula(model), data; offset=ones(nobs(model)))
    @test first(coef(model_with_offset)) ≈ first(coef(model)) - 1 atol=1e-8
    @test coef(model_with_offset)[2:end] ≈ coef(model)[2:end] atol=1e-8
    @test predict(model_with_offset, newobs; offset=[1]) ≈ [0.2854928] atol=1e-6
    @test_throws ArgumentError predict(model_with_offset, newobs)
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
    @test precision(model) ≈ 440.27838 atol=1e-5
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
