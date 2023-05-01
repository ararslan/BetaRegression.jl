# BetaRegression.jl

[![Build Status](https://github.com/ararslan/BetaRegression.jl/workflows/CI/badge.svg)](https://github.com/ararslan/BetaRegression.jl/actions?query=workflow%3ACI+branch%3Amain)
[![Code Coverage](http://codecov.io/github/ararslan/BetaRegression.jl/coverage.svg?branch=main)](http://codecov.io/github/ararslan/BetaRegression.jl?branch=main)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ararslan.github.io/BetaRegression.jl/stable)

This package provides regression modeling functionality for continuous,
[beta-distributed](https://en.wikipedia.org/wiki/Beta_distribution) responses with values
in the open interval (0, 1).
Models of this kind are particularly useful when modeling rates and proportions, which
naturally fall within the unit interval (or can be trivially transformed to do so).

In concept, beta regression models are quite similar to generalized linear models (GLMs),
though they are not actually GLMs due to the parameterization used for the beta
distribution.
However, users familiar with [GLM.jl](https://github.com/JuliaStats/GLM.jl) will likely
find that this package feels familiar in its interface and behavior.

See the package documentation for more information, including usage examples.

## References

Ferrari, S. & Cribari-Neto, F. (2004). Beta regression for modelling rates and proportions.
_Journal of Applied Statistics_, 31(7), 799–815.
https://doi.org/10.1080/0266476042000214501
