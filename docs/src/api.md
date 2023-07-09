```@meta
CurrentModule = BetaRegression
```

# API

The exported symbols from this package define its interface.
Some symbols from other packages are re-exported for convenience.
Fields of objects with composite types should not be accessed directly; the internals
of any given structure may change at any time and this would not be considered a breaking
change.

## Fitting a model

```@docs
BetaRegressionModel
BetaRegressionModel(::AbstractMatrix, ::AbstractVector)
fit(::Type{BetaRegressionModel}, ::AbstractMatrix, ::AbstractVector)
fit!(::BetaRegressionModel)
```

## Properties of a model

```@docs
aic
aicc
bic
coef(::BetaRegressionModel)
coefnames(::TableRegressionModel{<:BetaRegressionModel})
coeftable(::BetaRegressionModel)
confint(::BetaRegressionModel)
deviance(::BetaRegressionModel)
devresid(::BetaRegressionModel)
dof(::BetaRegressionModel)
dof_residual(::BetaRegressionModel)
fitted
informationmatrix(::BetaRegressionModel)
linearpredictor
Link(::BetaRegressionModel)
loglikelihood
modelmatrix
nobs(::BetaRegressionModel)
offset
params(::BetaRegressionModel)
precision(::BetaRegressionModel)
precisionlink
predict
r2(::BetaRegressionModel)
residuals
response
responsename(::TableRegressionModel{<:BetaRegressionModel})
score(::BetaRegressionModel)
stderror(::BetaRegressionModel)
vcov(::BetaRegressionModel)
weights
```

There is a subtlety here that bears repeating.
The function `coef` does _not_ include the precision term, only the regression
coefficients, so for a model with ``p`` independent variables, `coef` will return a vector
of length ``p``.
A number of other functions, such as `informationmatrix`, `vcov`, `stderror`, etc., _do_
include the precision term, and thus will return an array with (non-singleton) dimension
``p + 1``.
While this difference may seem strange at first blush, the design was chosen intentionally
to ensure that the model matrix and regression coefficient vector are conformable for
multiplication.
Use `params` to retrieve the full parameter vector with length ``p + 1``.

## Link functions

This package employs the system for link functions defined by the GLM.jl package.
In short, each link function has its own concrete type which subtypes `Link`.
Some may actually subtype `Link01`, which is itself a subtype of `Link`; this denotes
that the function's domain is the open unit interval, ``(0, 1)``.
Link functions are applied with `linkfun` and their inverse is applied with `linkinv`.
Relevant docstrings from GLM.jl are reproduced below.

Any mention of "the" link function for a `BetaRegressionModel` refers to that applied to
the mean (at least in this document).
However, despite only having one linear predictor, `BetaRegressionModel`s actually have
two link functions: one for the mean and one for the precision.

### Mean

```@docs
Link01
LogitLink
CauchitLink
CloglogLink
ProbitLink
```

### Precision

```@docs
IdentityLink
InverseLink
InverseSquareLink
LogLink
PowerLink
SqrtLink
```

## Developer documentation

This section documents some functions that are _not_ user facing (and are thus not
exported) and may be removed at any time.
They're included here for the benefit of anyone looking to contribute to the package
and wondering how certain internals work.
Other internal functions may be documented with comments in the source code rather
than with docstrings; read the source directly for more information on those.

```@docs
dmueta
initialize!
```
