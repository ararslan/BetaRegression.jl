# BetaRegression.jl

[![Work in Progress](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://github.com/ararslan/BetaRegression.jl/workflows/CI/badge.svg)](https://github.com/ararslan/BetaRegression.jl/actions?query=workflow%3ACI+branch%3Amain)
[![Code Coverage](http://codecov.io/github/ararslan/BetaRegression.jl/coverage.svg?branch=main)](http://codecov.io/github/ararslan/BetaRegression.jl?branch=main)

This package provides regression modeling functionality for beta-distributed responses
as described by Ferrari and Cribari-Neto<sup>[1]</sup>.
The concept is quite similar to that of a generalized linear model (GLM) except that the
beta distribution is not in the exponential dispersion family and the model coefficients
and dispersion parameter are not orthogonal.

The exported symbols from the package define its intended interface.
Note however that the interface and/or underlying implementation details may change
significantly prior to an initial release.

Still to do:
- Integration with StatsModels for passing `@formula`s to `fit`
- `coeftable` with Wald statistics
- Better tests and documentation

<sup>[1]</sup>Silvia Ferrari & Francisco Cribari-Neto (2004) Beta Regression for Modelling
Rates and Proportions, Journal of Applied Statistics, 31:7, 799-815,
DOI: 10.1080/0266476042000214501
