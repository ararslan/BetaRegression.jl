var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"CurrentModule = BetaRegression","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"The exported symbols from this package define its interface. Some symbols from other packages are re-exported for convenience. Fields of objects with composite types should not be accessed directly; the internals of any given structure may change at any time and this would not be considered a breaking change.","category":"page"},{"location":"api/#Fitting-a-model","page":"API","title":"Fitting a model","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"BetaRegressionModel\nfit(::BetaRegressionModel, ::AbstractMatrix, ::AbstractVector)\nfit!(::BetaRegressionModel)","category":"page"},{"location":"api/#BetaRegression.BetaRegressionModel","page":"API","title":"BetaRegression.BetaRegressionModel","text":"BetaRegressionModel{T,L,V,M} <: RegressionModel\n\nType representing a regression model for beta-distributed response values in the open interval (0, 1), as described by Ferrari and Cribari-Neto (2004).\n\nThe mean response is linked to the linear predictor by a link function with type L <: Link01, i.e. the link must map (0 1)  ℝ and use the GLM package's interface for link functions.\n\n\n\n\n\n","category":"type"},{"location":"api/#StatsAPI.fit!-Tuple{BetaRegressionModel}","page":"API","title":"StatsAPI.fit!","text":"fit!(b::BetaRegressionModel; maxiter=100, atol=1e-8, rtol=1e-8)\n\nFit the given BetaRegressionModel, updating its values in-place. If model convergence is achieved, b is returned, otherwise a ConvergenceException is thrown.\n\nFitting the model consists of computing the maximum likelihood estimates for the coefficients and the common dispersion via Fisher scoring with analytic derivatives. The model is determined to have converged when the score vector, i.e. the vector of first partial derivatives of the log likelihood with respect to the parameters, is approximately zero. This is determined by isapprox using the specified atol and rtol. maxiter dictates the maximum number of Fisher scoring iterations.\n\n\n\n\n\n","category":"method"},{"location":"api/#Properties-of-a-model","page":"API","title":"Properties of a model","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"The following common functions are extended for beta regression models:","category":"page"},{"location":"api/","page":"API","title":"API","text":"Link: The model's link function\ncoef: The vector boldsymbolbeta of regression coefficients\ndeviance: Model deviance\ndevresid: Vector of deviance residuals\ndispersion: The estimated dispersion parameter phi\ndof: Degrees of freedom\ndof_residual: Residual degrees of freedom\nfitted: The vector hatmathbfy of fitted values from the model\ninformationmatrix: Expected or observed Fisher information\nlinpred: The linear predictor vector boldsymboleta\nloglikelihood: Model log likelihood\nmodelmatrix: The model matrix mathbfX\nnobs: Number of observations used to fit the model\noffset: Model offset, empty if the model was not fit with an offset\nparams: All parameters from the model, including both boldsymbolbeta and phi\nresiduals: Vector of residuals\nresponse: The response vector boldsymboly\nscore: Score vector\nvcov: Variance-covariance matrix\nweights: Model weights, empty if the model was not fit with weights","category":"page"},{"location":"api/","page":"API","title":"API","text":"Note that for a model with p independent variables, the information and variance-covariance matrices will have p + 1 rows and columns, the last of which corresponds to the dispersion term. However, coef does not include the dispersion term and will have length p.","category":"page"},{"location":"api/#Developer-documentation","page":"API","title":"Developer documentation","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"This section documents some functions that are not user facing (and are thus not exported) and may be removed at any time. They're included here for the benefit of anyone looking to contribute to the package and wondering how certain internals work.","category":"page"},{"location":"api/","page":"API","title":"API","text":"dmueta\ninitialize!","category":"page"},{"location":"api/#BetaRegression.dmueta","page":"API","title":"BetaRegression.dmueta","text":"dmueta(link::Link, η)\n\nReturn the second derivative of linkinv, d²μdη², of the link function link evaluated at the linear predictor value η. A method of this function must be defined for a particular link function in order to compute the observed information matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#BetaRegression.initialize!","page":"API","title":"BetaRegression.initialize!","text":"initialize!(b::BetaRegressionModel)\n\nInitialize the given BetaRegressionModel by computing starting points for the parameter estimates and return the updated model object. The initial estimates are based on those from a linear regression model with the same model matrix as b but with linkfun.(Link(b), response(b)) as the response.\n\n\n\n\n\n","category":"function"},{"location":"details/#Details","page":"Details","title":"Details","text":"","category":"section"},{"location":"details/#What-is-beta-regression?","page":"Details","title":"What is beta regression?","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"Beta regression is a type of regression model similar to a generalized linear model (GLM) but with a couple of key differences. It was first described by Ferrari and Cribari-Neto (2004)[1] with later extensions by Vasconcellos and Cribari-Neto (2005)[2], Smithson and Verkuilen (2006)[3], Ospina et al. (2006)[4], and Simas et al. (2010)[5].","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"Let's briefly review some high-level ideas behind GLMs, starting with notation.","category":"page"},{"location":"details/#A-bit-about-GLMs","page":"Details","title":"A bit about GLMs","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"Let mathbfy in mathbbR^n be a vector of n observed outcomes and let mathbfX in mathbbR^n times p be a matrix of n measurements on p independent variables. Further, let y_i sim mathcalD(mu_i phi) for a distribution mathcalD with parameters mu_i and phi. That is, each observed outcome was generated from the same overall distribution but with possibly different means mu_i in mathbbR and a common dispersion phi in mathbbR. We relate the mean vector boldsymbolmu to the independent variables via a link function g mathbbR mapsto mathbbR and a linear predictor boldsymboleta = mathbfX boldsymbolbeta such that mu_i = g^-1(eta_i). Here, boldsymbolbeta in mathbbR^p is a vector of regression coefficients to be estimated.","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"GLMs were first described by Nelder and Wedderburn (1972)[6] and expanded upon in the classic book by McCullagh and Nelder (1989)[7]. Nelder and Wedderburn showed that under certain conditions, most notably when mathcalD is a member of the exponential family of distributions, the maximum likelihood estimate of boldsymbolbeta, denoted hatboldsymbolbeta, can be found by the method of iteratively reweighted least squares (IRLS). Within this framework, phi does not need to be estimated directly; it can be obtained simply from the deviance of the model, n, and p. This relies, however, on the orthogonality of boldsymbolbeta and phi. Indeed, orthogonality these parameters was proved for all GLMs by Huang and Rathouz (2017)[8], expanding upon much earlier results for specific models.","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"In Julia, the canonical implementation of GLMs is in the package GLM.jl, which uses IRLS and supports specific distributions from the Distributions.jl package.","category":"page"},{"location":"details/#The-beta-distribution","page":"Details","title":"The beta distribution","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"Our primary concern will be with the beta distribution, a continuous probability distribution with support on the open interval (0 1) and two positive real shape parameters p and q. It has probability density function","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"f(y p q) = fracy^p - 1 (1 - y)^q - 1Beta(p q)","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"where Beta(cdot cdot) is the beta function. The beta distribution in this parameterization is available from Distributions.jl as Beta.","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"Ferrari and Cribari-Neto reparameterize the distribution in terms of a mean 0  mu  1 and dispersion phi  0 such that","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"mu = fracpp + q quad quad phi = p + q","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"In this parameterization, it's clear that mu and phi are not separable; indeed, phi appears in the denominator of mu's definition!","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"We then have, for y sim mathcalB(mu phi), where mathcalB is the beta distribution in this parameterization, the probability density function","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"f(y mu phi) =\n    fracy^mu phi - 1 (1 - y)^(1 - mu) phi - 1Beta(mu phi (1 - mu) phi)","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"with","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"textE(y) = mu quad quad textVar(y) = fracmu (1 - mu)phi + 1","category":"page"},{"location":"details/#Beta-regression","page":"Details","title":"Beta regression","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"With all of these definitions in mind, we can now formulate the beta regression model. Assume now that","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"y_i sim mathcalB(g^-1(mathbfx_i^top boldsymbolbeta) phi) quad quad\ni = 1 ldots n","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"where the link function is g (0 1) mapsto mathbbR and mathbfx_i^top is the ith row of mathbfX. Just like with GLMs, we're modeling mu as a function of the linear predictor and our ultimate goal is to estimate boldsymbolbeta. But since mu depends on phi, so does boldsymbolbeta! Thus to estimate boldsymbolbeta, we must also estimate phi. As it turns out, we don't have to resort to anything fancy in order to do this; we can simply use maximum likelihood.","category":"page"},{"location":"details/#Fitting-a-model","page":"Details","title":"Fitting a model","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"In BetaRegression.jl, the maximum likelihood estimation is carried out via Fisher scoring using closed-form expressions for the score vector and expected information matrix. There is no canonical link function for the beta regression model in this parameterization in the same manner as for GLMs (anything that constrains mu within (0 1) will do just fine) but for simplicity the default link function is logit. In the parlance of GLM.jl, this means that any Link01 can be used and the default is LogitLink.","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"Mirroring the API for GLMs provided by GLM.jl, a beta regression model is fit by passing an explicit design matrix X and response vector y as in","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"fit(BetaRegressionModel, X, y, link; kwargs...)","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"or by providing a Tables.jl-compatible table table and a formula specified via @formula in Wilkinson notation as in","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"fit(BetaRegressionModel, @formula(y ~ 1 + x1 + ... + xn), table, link; kwargs...)","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"In both methods, the link argument is optional and, as previously mentioned, defaults to LogitLink(). The keyword arguments provide control over the fitting process as well as the ability to specify an offset and weights. (Note however that weights are currently unsupported.)","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"The variables passed to the model, be it by way of design matrix or formula, cannot be collinear. Unlike lm from GLM.jl, which provides facilities for automatically dropping collinear variables, BetaRegression.jl does not handle this case. It's up to you, dear user, to just not do that.","category":"page"},{"location":"details/#References","page":"Details","title":"References","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"[1]: Ferrari, S. and Cribari-Neto, F. (2004). Beta Regression for Modelling Rates and Proportions. Journal of Applied Statistics, 31, issue 7, p. 799-815.","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"[2]: Vasconcellos, K. L. P. and Cribari-Neto F. (2005). Improved Maximum Likelihood Estimation in a New Class of Beta Regression Models. Brazilian Journal of Probability and Statistics, 19(1), 13–31.","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"[3]: Smithson, M. and Verkuilen, J. (2006). A Better Lemon Squeezer? Maximum-Likelihood Regression with Beta-Distributed Dependent Variables. Psychological Methods, 11(1), 54–71.","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"[4]: Ospina, R., Cribari-Neto, F., and Vasconcellos K. L. P. (2006). Improved Point and Interval Estimation for a Beta Regression Model. Computational Statistics & Data Analysis, 51(2), 960–981.","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"[5]: Simas, A. B., Barreto-Souza, W., and Rocha, A. V. (2010). Improved Estimators for a General Class of Beta Regression Models. Computational Statistics & Data Analysis, 54(2), 348–366.","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"[6]: Nelder, J. A. and Wedderburn, R. W. M. (1972). Generalized Linear Models. Journal of the Royal Statistical Society. Series A (General), 135(3), 370. doi:10.2307/2344614","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"[7]: McCullagh, P. and Nelder, J. A. (1989). Generalized Linear Models. 2nd Edition, Chapman and Hall, London. doi:10.1007/978-1-4899-3242-6","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"[8]: Huang, A., and Rathouz, P. J. (2017). Orthogonality of the Mean and Error Distribution in Generalized Linear Models. Communications in statistics: theory and methods, 46(7), 3290–3296.","category":"page"},{"location":"#BetaRegression.jl","page":"Home","title":"BetaRegression.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"BetaRegression.jl is a package that provides beta regression functionality for the Julia language.","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package is not yet registered in the official package registry. To install the package, run","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg\n\njulia> Pkg.add(; url=\"https://github.com/ararslan/BetaRegression.jl\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"or, using the Pkg REPL mode (press ]),","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/ararslan/BetaRegression.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"These instructions will be updated once the package is registered.","category":"page"},{"location":"","page":"Home","title":"Home","text":"If you're looking for package documentation, welcome! You've found it. Documentation for the package's API is available on the API page and further information about beta regression in general and the methodology used by this package is available in Details.","category":"page"},{"location":"#Note","page":"Home","title":"Note","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Beta regression is implemented in R in the betareg package and in Python in statsmodels. Note that BetaRegression.jl is not based on either of these (betareg in particular is GPL-licensed) nor is feature or implementation parity a goal.","category":"page"}]
}
