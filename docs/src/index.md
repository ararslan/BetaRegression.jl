# BetaRegression.jl

BetaRegression.jl is a package that provides beta regression functionality for the Julia
language.

## Getting started

This package is not yet registered in the official package registry.
To install the package, run

```julia-repl
julia> using Pkg

julia> Pkg.add(; url="https://github.com/ararslan/BetaRegression.jl")
```

or, using the Pkg REPL mode (press `]`),

```julia-repl
pkg> add https://github.com/ararslan/BetaRegression.jl
```

These instructions will be updated once the package is registered.

If you're looking for package documentation, welcome!
You've found it.
Documentation for the package's API is available on the [API](@ref) page and further
information about beta regression in general and the methodology used by this package
is available in [Details](@ref).

## Note

Beta regression is implemented in R in the
[betareg](https://cran.r-project.org/web/packages/betareg/betareg.pdf) package and in
Python in [statsmodels](https://www.statsmodels.org).
Note that BetaRegression.jl is not based on either of these (betareg in particular is
GPL-licensed) nor is feature or implementation parity a goal.
