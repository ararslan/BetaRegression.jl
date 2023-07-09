# Details

## What is beta regression?

Beta regression is a type of regression model similar to a
[generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLM)
but with a couple of key differences.
It was first described by Ferrari and Cribari-Neto (2004)[^1] with later extensions by
Vasconcellos and Cribari-Neto (2005)[^2], Smithson and Verkuilen (2006)[^3], Ospina et al.
(2006)[^4], and Simas et al. (2010)[^5].

Let's briefly review some high-level ideas behind GLMs, starting with notation.

### A bit about GLMs

Let ``\mathbf{y} \in \mathbb{R}^n`` be a vector of ``n`` observed outcomes and let
``\mathbf{X} \in \mathbb{R}^{n \times p}`` be a matrix of ``n`` measurements on ``p``
independent variables.
Further, let ``y_i \sim \mathcal{D}(\mu_i, \phi)`` for a distribution ``\mathcal{D}`` with
parameters ``\mu_i`` and ``\phi``.
That is, each observed outcome was generated from the same overall distribution but with
possibly different means ``\mu_i \in \mathbb{R}`` and a common precision parameter
``\phi \in \mathbb{R}``.
We relate the mean vector ``\boldsymbol{\mu}`` to the independent variables via a _link
function_ ``g: \mathbb{R} \mapsto \mathbb{R}`` and a _linear predictor_
``\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\beta}`` such that ``\mu_i = g^{-1}(\eta_i)``.
Here, ``\boldsymbol{\beta} \in \mathbb{R}^p`` is a vector of regression coefficients to be
estimated.

GLMs were first described by Nelder and Wedderburn (1972)[^6] and expanded upon in the
classic book by McCullagh and Nelder (1989)[^7].
Nelder and Wedderburn showed that under certain conditions, most notably when
``\mathcal{D}`` is a member of the
[exponential family](https://en.wikipedia.org/wiki/Exponential_family) of distributions,
the maximum likelihood estimate of ``\boldsymbol{\beta}``, denoted
``\hat{\boldsymbol{\beta}}``, can be found by the method of iteratively reweighted least
squares (IRLS).
Within this framework, ``\phi`` does not need to be estimated directly; it can be obtained
simply from the deviance of the model, ``n``, and ``p``.
This relies, however, on the orthogonality of ``\boldsymbol{\beta}`` and ``\phi``.
Indeed, orthogonality these parameters was proved for _all_ GLMs by Huang and Rathouz
(2017)[^8], expanding upon much earlier results for specific models.

In Julia, the canonical implementation of GLMs is in the package
[GLM.jl](https://github.com/JuliaStats/GLM.jl), which uses IRLS and supports specific
distributions from the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
package.

### The beta distribution

Our primary concern will be with the
[beta distribution](https://en.wikipedia.org/wiki/Beta_distribution), a continuous
probability distribution with support on the open interval ``(0, 1)`` and two positive
real shape parameters ``p`` and ``q``.
It has probability density function

```math
f(y; p, q) = \frac{y^{p - 1} (1 - y)^{q - 1}}{\Beta(p, q)}
```

where ``\Beta(\cdot, \cdot)`` is the [beta function](https://en.wikipedia.org/wiki/Beta_function).
The beta distribution in this parameterization is available from Distributions.jl as
`Beta`.

Ferrari and Cribari-Neto reparameterize the distribution in terms of a mean ``0 < \mu < 1``
and precision ``\phi > 0`` such that

```math
\mu = \frac{p}{p + q}, \quad \quad \phi = p + q
```

In this parameterization, it's clear that ``\mu`` and ``\phi`` are not separable; indeed,
``\phi`` appears in the denominator of ``\mu``'s definition!

We then have, for ``y \sim \mathcal{B}(\mu, \phi)``, where ``\mathcal{B}`` is the
beta distribution in this parameterization, the probability density function

```math
f(y; \mu, \phi) =
    \frac{y^{\mu \phi - 1} (1 - y)^{(1 - \mu) \phi - 1}}{\Beta(\mu \phi, (1 - \mu) \phi)}
```

with

```math
\text{E}(y) = \mu, \quad \quad \text{Var}(y) = \frac{\mu (1 - \mu)}{\phi + 1}.
```

### Beta regression

With all of these definitions in mind, we can now formulate the beta regression model.
Assume now that

```math
y_i \sim \mathcal{B}(g^{-1}(\mathbf{x}_i^\top \boldsymbol{\beta}), h^{-1}(\phi)),
\quad \quad i = 1, \ldots, n
```

where the link function for the mean ``\mu`` is ``g: (0, 1) \mapsto \mathbb{R}``,
``\mathbf{x}_i^\top`` is the ``i``th row of ``\mathbf{X}``, and the link function for
the precision ``\phi`` is ``h: (0, \infty) \mapsto \mathbb{R}``.
Just like with GLMs, we're modeling ``\mu`` as a function of the linear predictor and
our ultimate goal is to estimate ``\boldsymbol{\beta}``.
But since ``\mu`` depends on ``\phi``, so does ``\boldsymbol{\beta}``!
Thus to estimate ``\boldsymbol{\beta}``, we must also estimate ``\phi``, which is assumed
to be an unknown constant.

Some formulations of the beta regression model use separate sub-models for the mean and
precision with separate coefficients and possibly non-overlapping sets of independent
variables, thereby not assuming constant precision.
That is not implemented in this package.
By analogy, what is implemented here is an intercept-only sub-model for the precision.

We don't have to resort to anything fancy in order to fit beta regression models; we can
simply use [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
on the full parameter vector for the model, which we define to be
``\theta = [\beta_1, \ldots, \beta_p, \phi]``.

## Fitting a model

In BetaRegression.jl, the maximum likelihood estimation is carried out via
[Fisher scoring](https://en.wikipedia.org/wiki/Scoring_algorithm) using closed-form
expressions for the score vector and expected information matrix.

The information matrix is symmetric with the following block structure:
```math
\left[
    \begin{array}{ccc|c}
        \frac{\partial^2 \ell}{\partial \beta_1^2} & \cdots &
        \frac{\partial^2 \ell}{\partial \beta_1 \partial \beta_p} &
        \frac{\partial^2 \ell}{\partial \beta_1 \partial \phi} \\
        \vdots & \ddots & \vdots & \vdots \\
        \frac{\partial^2 \ell}{\partial \beta_p \partial \beta_1} & \cdots &
        \frac{\partial^2 \ell}{\partial \beta_p^2} &
        \frac{\partial^2 \ell}{\partial \beta_p \partial \phi} \\
        \hline \\
        \frac{\partial^2 \ell}{\partial \phi \partial \beta_1} & \cdots &
        \frac{\partial^2 \ell}{\partial \phi \partial \beta_p} &
        \frac{\partial^2 \ell}{\partial \phi^2}
    \end{array}
\right]
```
Since ``\mu`` depends on ``\phi``, we have that
``\mathbb{E}\left(\frac{\partial^2 \ell}{\partial \beta_i \partial \phi}\right) \neq 0``,
so the matrix is not block diagonal.

There is no canonical link function for the beta regression model in this parameterization
in the same manner as for GLMs (anything that constrains ``\mu`` within ``(0, 1)`` will
do just fine) but for simplicity and interpretability the default link function is
[logit](https://en.wikipedia.org/wiki/Logit).
In the parlance of GLM.jl, this means that any `Link01` can be used and the default is
`LogitLink`.

Providing a separate link function for the precision can improve numerical stability
when fitting models by naturally constraining the precision to be nonnegative.
The default is the identity link function, or in GLM.jl terms, `IdentityLink`, but other
common choices include the logarithm and square root links, `LogLink` and `SqrtLink`,
respectively.

Mirroring the API for GLMs provided by GLM.jl, a beta regression model is fit by passing
an explicit design matrix `X` and response vector `y` as in

```julia
fit(BetaRegressionModel, X, y, meanlink, precisionlink; kwargs...)
```

or by providing a [Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible table
`table` and a formula specified via `@formula` in Wilkinson notation as in

```julia
fit(BetaRegressionModel, @formula(y ~ 1 + x1 + ... + xn), table, meanlink, precisionlink; kwargs...)
```

In both methods, the `meanlink` and `precisionlink` arguments are optional and, as
previously mentioned, default to `LogitLink()` and `IdentityLink()`, respectively.
The keyword arguments provide control over the fitting process as well as the ability
to specify an offset and weights.
(Note however that weights are currently unsupported.)

The variables passed to the model, be it by way of design matrix or formula, cannot be
collinear.
Unlike `lm` from GLM.jl, which provides facilities for automatically dropping collinear
variables, BetaRegression.jl does not handle this case.
It's up to you, dear user, to just not do that.

## References

[^1]: Ferrari, S. and Cribari-Neto, F. (2004). _Beta Regression for Modelling Rates and Proportions_. Journal of Applied Statistics, 31, issue 7, p. 799-815.
[^2]: Vasconcellos, K. L. P. and Cribari-Neto F. (2005). _Improved Maximum Likelihood Estimation in a New Class of Beta Regression Models_. Brazilian Journal of Probability and Statistics, 19(1), 13–31.
[^3]: Smithson, M. and Verkuilen, J. (2006). _A Better Lemon Squeezer? Maximum-Likelihood Regression with Beta-Distributed Dependent Variables_. Psychological Methods, 11(1), 54–71.
[^4]: Ospina, R., Cribari-Neto, F., and Vasconcellos K. L. P. (2006). _Improved Point and Interval Estimation for a Beta Regression Model_. Computational Statistics & Data Analysis, 51(2), 960–981.
[^5]: Simas, A. B., Barreto-Souza, W., and Rocha, A. V. (2010). _Improved Estimators for a General Class of Beta Regression Models_. Computational Statistics & Data Analysis, 54(2), 348–366.
[^6]: Nelder, J. A. and Wedderburn, R. W. M. (1972). _Generalized Linear Models_. Journal of the Royal Statistical Society. Series A (General), 135(3), 370. doi:10.2307/2344614
[^7]: McCullagh, P. and Nelder, J. A. (1989). _Generalized Linear Models_. 2nd Edition, Chapman and Hall, London. doi:10.1007/978-1-4899-3242-6
[^8]: Huang, A., and Rathouz, P. J. (2017). _Orthogonality of the Mean and Error Distribution in Generalized Linear Models_. Communications in statistics: theory and methods, 46(7), 3290–3296.
