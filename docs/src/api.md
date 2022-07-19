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
fit(::Type{<:BetaRegressionModel}, ::AbstractMatrix, ::AbstractVector)
fit!(::BetaRegressionModel)
```

## Properties of a model

The following common functions are extended for beta regression models:
- `Link`: The model's link function
- `coef`: The vector ``\boldsymbol{\beta}`` of regression coefficients
- `deviance`: Model deviance
- `devresid`: Vector of deviance residuals
- `dispersion`: The estimated dispersion parameter ``\phi``
- `dof`: Degrees of freedom
- `dof_residual`: Residual degrees of freedom
- `fitted`: The vector ``\hat{\mathbf{y}}`` of fitted values from the model
- `informationmatrix`: Expected or observed Fisher information
- `linpred`: The linear predictor vector ``\boldsymbol{\eta}``
- `loglikelihood`: Model log likelihood
- `modelmatrix`: The model matrix ``\mathbf{X}``
- `nobs`: Number of observations used to fit the model
- `offset`: Model offset, empty if the model was not fit with an offset
- `params`: All parameters from the model, including both ``\boldsymbol{\beta}`` and ``\phi``
- `residuals`: Vector of residuals
- `response`: The response vector ``\boldsymbol{y}``
- `score`: Score vector
- `vcov`: Variance-covariance matrix
- `weights`: Model weights, empty if the model was not fit with weights

Note that for a model with ``p`` independent variables, the information and
variance-covariance matrices will have ``p + 1`` rows and columns, the last of which
corresponds to the dispersion term.
However, `coef` does _not_ include the dispersion term and will have length ``p``.

## Developer documentation

This section documents some functions that are _not_ user facing (and are thus not
exported) and may be removed at any time.
They're included here for the benefit of anyone looking to contribute to the package
and wondering how certain internals work.

```@docs
dmueta
initialize!
```
