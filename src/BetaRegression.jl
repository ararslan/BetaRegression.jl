module BetaRegression

using Distributions
using GLM
using LinearAlgebra
using LinearAlgebra.BLAS
using LogExpFunctions
using SpecialFunctions
using Statistics
using StatsAPI
using StatsBase
using StatsModels

# Necessary stuff that isn't exported from dependencies
using GLM: Link01, LmResp, cholpred, inverselink, linkfun, linkinv, mueta
using LinearAlgebra: copytri!
using StatsAPI: aic, aicc, bic, linearpredictor, linearpredictor!, offset, params
using StatsModels: TableRegressionModel, @delegate

export
    BetaRegressionModel,
    precisionlink,
    # Extensions/utilities from GLM:
    CauchitLink,
    CloglogLink,
    IdentityLink,
    InverseLink,
    InverseSquareLink,
    Link,
    Link01,
    LogLink,
    LogitLink,
    NegativeBinomialLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
    # Utilities from StatsModels:
    @formula,
    # Extensions from StatsAPI:
    aic,
    aicc,
    bic,
    coef,
    coefnames,
    coeftable,
    confint,
    deviance,
    devresid,
    dof,
    dof_residual,
    fit!,
    fit,
    fitted,
    informationmatrix,
    linearpredictor,
    loglikelihood,
    modelmatrix,
    nobs,
    offset,
    params,
    predict,
    r2,
    r²,
    residuals,
    response,
    responsename,
    score,
    stderror,
    vcov,
    weights

"""
    BetaRegressionModel{T,L1,L2,V,M} <: RegressionModel

Type representing a regression model for beta-distributed response values in the open
interval (0, 1), as described by Ferrari and Cribari-Neto (2004).

The mean response is linked to the linear predictor by a link function with type
`L1 <: Link01`, i.e. the link must map ``(0, 1) \\mapsto \\mathbb{R}`` and use the GLM
package's interface for link functions.
While there is no canonical link function for the beta regression model as there is for
GLMs, logit is the most common choice.

The precision is transformed by a link function with type `L2 <: Link` which should map
``\\mathbb{R} \\mapsto \\mathbb{R}`` or, ideally, ``(0, \\infty) \\mapsto \\mathbb{R}``
because the precision must be positive.
The most common choices are the identity, log, and square root links.
"""
struct BetaRegressionModel{T<:AbstractFloat,L1<:Link01,L2<:Link,V<:AbstractVector{T},
                           M<:AbstractMatrix{T}} <: RegressionModel
    y::V
    X::M
    weights::Vector{T}
    offset::Vector{T}
    parameters::Vector{T}
    linearpredictor::Vector{T}
end

"""
    BetaRegressionModel(X, y, link=LogitLink(), precisionlink=IdentityLink();
                        weights=nothing, offset=nothing)

Construct a `BetaRegressionModel` object with the given model matrix `X`, response
`y`, mean link function `link`, precision link function `precisionlink`, and optionally
`weights` and `offset`.
Note that the returned object is not fit until `fit!` is called on it.

!!! warning
    Support for user-provided weights is currently incomplete; passing a value other
    than `nothing` or an empty array for `weights` will result in an error for now.
"""
function BetaRegressionModel(X::AbstractMatrix, y::AbstractVector,
                             link::Link01=LogitLink(), precisionlink::Link=IdentityLink();
                             weights=nothing, offset=nothing)
    n, p = size(X)
    p < n || throw(ArgumentError("model matrix must have fewer columns than rows"))
    if n != length(y)
        throw(DimensionMismatch("model matrix and response have different numbers " *
                                "of observations"))
    end
    all(yᵢ -> 0 < yᵢ < 1, y) || throw(ArgumentError("response values must be 0 < y < 1"))
    T = promote_type(float(eltype(X)), float(eltype(y)))
    if weights === nothing
        weights = Vector{T}()
    else
        nw = length(weights)
        nw == 0 || throw(ArgumentError("user-provided weights are currently unsupported"))
    end
    if offset === nothing
        offset = Vector{T}()
    else
        no = length(offset)
        no == 0 || no == n || throw(ArgumentError("offset must be empty or have length $n"))
    end
    parameters = zeros(T, p + 1)
    η = Vector{T}(undef, n)
    _X = convert(AbstractMatrix{T}, X)
    _y = convert(AbstractVector{T}, y)
    L1 = typeof(link)
    L2 = typeof(precisionlink)
    return BetaRegressionModel{T,L1,L2,typeof(_y),typeof(_X)}(_y, _X, weights,
                                                              offset, parameters, η)
end

function Base.show(io::IO, b::BetaRegressionModel{T,L1,L2}) where {T,L1,L2}
    print(io, """
          BetaRegressionModel{$T}
              $(nobs(b)) observations
              $(dof(b)) degrees of freedom
              Mean link: $L1
              Precision link: $L2

          Coefficients:
          """)
    show(io, coeftable(b))
    return nothing
end

StatsAPI.response(b::BetaRegressionModel) = b.y

StatsAPI.modelmatrix(b::BetaRegressionModel) = b.X

StatsAPI.weights(b::BetaRegressionModel) = b.weights

StatsAPI.offset(b::BetaRegressionModel) = b.offset

"""
    params(model::BetaRegressionModel)

Return the vector of estimated model parameters
``\\theta = [\\beta_1, \\ldots, \\beta_p, \\phi]``, i.e. the regression coefficients and
precision.

!!! danger
    Mutating this array may invalidate the model object.

See also: [`coef`](@ref), [`precision`](@ref)
"""
StatsAPI.params(b::BetaRegressionModel) = b.parameters

"""
    coef(model::BetaRegressionModel)

Return a copy of the vector of regression coefficients ``\\mathbf{\\beta}``.

See also: [`precision`](@ref), [`params`](@ref)
"""
StatsAPI.coef(b::BetaRegressionModel) = params(b)[1:(end - 1)]

"""
    precision(model::BetaRegressionModel)

Return the estimated precision parameter, ``\\phi``, for the model.
This function returns ``\\phi`` on the natural scale, _not_ on the precision link scale.
This parameter is estimated alongside the regression coefficients and is included in
coefficient tables, where it _is_ displayed on the precision link scale.

See also: [`coef`](@ref), [`params`](@ref)
"""
Base.precision(b::BetaRegressionModel) = linkinv(precisionlink(b), last(params(b)))

precisioninverselink(b::BetaRegressionModel) = inverselink(precisionlink(b), last(params(b)))

StatsAPI.linearpredictor(b::BetaRegressionModel) = b.linearpredictor

function StatsAPI.linearpredictor!(b::BetaRegressionModel)
    X = modelmatrix(b)
    β = view(params(b), 1:size(X, 2))
    η = linearpredictor(b)
    if isempty(offset(b))
        mul!(η, X, β)
    else
        copyto!(η, offset(b))
        mul!(η, X, β, true, true)
    end
    return η
end

StatsAPI.fitted(b::BetaRegressionModel) = linkinv.(Link(b), linearpredictor(b))

StatsAPI.residuals(b::BetaRegressionModel) = response(b) .- fitted(b)

"""
    nobs(model::BetaRegressionModel)

Return the effective number of observations used to fit the model. For weighted models,
this is the number of nonzero weights, otherwise it's the number of elements of the
response (or equivalently, the number of rows in the model matrix).
"""
StatsAPI.nobs(b::BetaRegressionModel) =
    isempty(weights(b)) ? length(response(b)) : count(>(0), weights(b))

"""
    dof(model::BetaRegressionModel)

Return the number of estimated parameters in the model. For a model with ``p`` independent
variables, this is ``p + 1``, since the precision must also be estimated.
"""
StatsAPI.dof(b::BetaRegressionModel) = length(params(b))

"""
    dof_residual(model::BetaRegressionModel)

Return the residual degrees of freedom for the model, defined as [`nobs`](@ref) minus
[`dof`](@ref).
"""
StatsAPI.dof_residual(b::BetaRegressionModel) = nobs(b) - dof(b)

"""
    r2(model::BetaRegressionModel)
    r²(model::BetaRegressionModel)

Return the Pearson correlation between the linear predictor ``\\eta`` and the
link-transformed response ``g(y)``.
"""
StatsAPI.r2(b::BetaRegressionModel) = cor(linearpredictor(b), linkfun.(Link(b), response(b)))^2

StatsAPI.predict(b::BetaRegressionModel) = fitted(b)

function StatsAPI.predict(b::BetaRegressionModel, newX::AbstractMatrix; offset=nothing)
    if !isempty(b.offset) && (offset === nothing || isempty(offset))
        throw(ArgumentError("model was fit with an offset but no offset was provided"))
    end
    η̂ = newX * coef(b)
    if offset !== nothing && !isempty(offset)
        η̂ .+= offset
    end
    return linkinv.(Link(b), η̂)
end

"""
    Link(model::BetaRegressionModel)

Return the link function ``g`` that links the mean ``\\mu`` to the linear predictor
``\\eta`` by ``\\mu = g^{-1}(\\eta)``.
"""
GLM.Link(::BetaRegressionModel{T,L1}) where {T,L1} = L1()

"""
    precisionlink(model::BetaRegressionModel)

Return the link function ``h`` that links the precision ``\\phi`` to the estimated
constant parameter ``\\theta_{p+1}`` such that ``\\phi = h^{-1}(\\theta_{p+1})``.
"""
precisionlink(::BetaRegressionModel{T,L1,L2}) where {T,L1,L2} = L2()

"""
    coeftable(model::BetaRegressionModel; level=0.95)

Return a table of the point estimates of the model parameters, their respective
standard errors, ``z``-statistics, Wald ``p``-values, and confidence intervals at
the given `level`. The precision parameter is included as the last row in the table.

The object returned by this function implements the
[Tables.jl](https://github.com/JuliaData/Tables.jl/) interface for tabular data.
"""
function StatsAPI.coeftable(b::BetaRegressionModel; level::Real=0.95)
    θ = params(b)
    se = stderror(b)
    z = θ ./ se
    p = 2 * ccdf.(Normal(), abs.(z))
    ci = confint(b; level)
    level *= 100
    s = string(isinteger(level) ? trunc(Int, level) : level)
    return CoefTable(hcat(θ, se, z, p, ci),
                     ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $s%", "Upper $s%"],
                     push!(map(i -> "x$i", 1:(length(θ) - 1)), "(Precision)"), 4, 3)
end

"""
    confint(model::BetaRegressionModel; level=0.95)

For a model with ``p`` regression coefficients, return a ``(p + 1) \\times 2`` matrix
of confidence intervals for the estimated coefficients and precision at the given `level`.
"""
function StatsAPI.confint(b::BetaRegressionModel; level::Real=0.95)
    θ = params(b)
    se = stderror(b)
    side = se .* quantile(Normal(), (1 - level) / 2)
    return hcat(θ .+ side, θ .- side)
end

betalogpdf(μ, ϕ, y) = logpdf(Beta(μ * ϕ, (1 - μ) * ϕ), y)

function StatsAPI.loglikelihood(b::BetaRegressionModel)
    y = response(b)
    μ = fitted(b)
    ϕ = precision(b)
    w = weights(b)
    if isempty(w)
        return sum(i -> betalogpdf(μ[i], ϕ, y[i]), eachindex(y, μ))
    else
        return sum(i -> w[i] * betalogpdf(μ[i], ϕ, y[i]), eachindex(y, μ, w))
    end
end

function StatsAPI.loglikelihood(b::BetaRegressionModel, ::Colon)
    y = response(b)
    μ = fitted(b)
    ϕ = precision(b)
    w = weights(b)
    if isempty(w)
        return map((yᵢ, μᵢ) -> betalogpdf(μᵢ, ϕ, yᵢ), y, μ)
    else
        return map((yᵢ, μᵢ, wᵢ) -> wᵢ * betalogpdf(μᵢ, ϕ, yᵢ), y, μ, w)
    end
end

function StatsAPI.loglikelihood(b::BetaRegressionModel, i::Integer)
    y = response(b)
    @boundscheck checkbounds(y, i)
    η = linearpredictor(b)[i]
    μ = linkinv(Link(b), η)
    ϕ = precision(b)
    ℓ = betalogpdf(μ, ϕ, y[i])
    isempty(weights(b)) || (ℓ *= weights(b)[i])
    return ℓ
end

"""
    devresid(model::BetaRegressionModel)

Compute the signed deviance residuals of the model,
```math
\\mathrm{sgn}(y_i - \\hat{y}_i) \\sqrt{2 \\lvert \\ell(y_i, \\hat{\\phi}) - \\ell(\\hat{y}_i, \\hat{\\phi}) \\rvert}
```
where ``\\ell`` denotes the log likelihood, ``y_i`` is the ``i``th observed value of the
response, ``\\hat{y}_i`` is the ``i``th fitted value, and ``\\hat{\\phi}`` is the estimated
common precision parameter.

See also: [`deviance`](@ref)
"""
function GLM.devresid(b::BetaRegressionModel)
    ϕ = precision(b)
    return map(response(b), fitted(b)) do y, μ
        r = y - μ
        ℓ₁ = betalogpdf(y, ϕ, y)
        ℓ₂ = betalogpdf(μ, ϕ, y)
        return sign(r) * sqrt(2 * abs(ℓ₁ - ℓ₂))
    end
end

"""
    deviance(model::BetaRegressionModel)

Compute the deviance of the model, defined as the sum of the squared deviance residuals.

See also: [`devresid`](@ref)
"""
StatsAPI.deviance(b::BetaRegressionModel) = sum(abs2, devresid(b))

# Initialization method as recommended at the end of Ferrari (2004), section 2
"""
    initialize!(b::BetaRegressionModel)

Initialize the given [`BetaRegressionModel`](@ref) by computing starting points for
the parameter estimates and return the updated model object.
The initial estimates are based on those from a linear regression model with the same
model matrix as `b` but with `linkfun.(Link(b), response(b))` as the response.

If the initial estimate of the precision is invalid (not strictly positive) then it
is taken instead to be 1 prior to applying the precision link function.
"""
function initialize!(b::BetaRegressionModel)
    link = Link(b)
    X = modelmatrix(b)
    y = linkfun.(link, response(b))
    # We have to use the constructors directly because `LinearModel` supports an
    # offset but it isn't exposed by `lm`
    model = LinearModel(LmResp{typeof(y)}(zero(y), offset(b), weights(b), y),
                        cholpred(X, true), nothing)
    fit!(model)
    β = coef(model)
    η = fitted(model)
    n = nobs(model)
    k = length(β)
    e = sum(abs2, residuals(model)) / (n - k)
    ϕ = zero(eltype(β))
    for ηᵢ in η
        μᵢ = linkinv(link, ηᵢ)
        σᵢ² = e * mueta(link, ηᵢ)^2
        ϕ += μᵢ * (1 - μᵢ) / σᵢ² - 1
    end
    ϕ /= n
    # No valid estimate for the precision, follow suit with `betareg` in R and set to 1
    # See https://stats.stackexchange.com/a/593670
    ϕ > 0 || (ϕ = one(ϕ))
    ϕ = linkfun(precisionlink(b), ϕ)
    copyto!(params(b), push!(β, ϕ))
    copyto!(linearpredictor(b), η)
    return b
end

"""
    score(model::BetaRegressionModel)

Compute the score vector of the model, i.e. the vector of first partial derivatives
of [`loglikelihood`](@ref) with respect to each element of [`params`](@ref).

See also: [`informationmatrix`](@ref)
"""
function StatsAPI.score(b::BetaRegressionModel)
    X = modelmatrix(b)
    y = response(b)
    η = linearpredictor(b)
    link = Link(b)
    ϕ, dϕ, _ = precisioninverselink(b)
    ψϕ = digamma(ϕ)
    ∂θ = zero(params(b))
    Tr = copy(η)
    @inbounds for i in eachindex(y, η)
        yᵢ = y[i]
        μᵢ, dμdη, omμᵢ = inverselink(link, η[i])
        ψp = digamma(ϕ * μᵢ)
        ψq = digamma(ϕ * omμᵢ)
        Δ = logit(yᵢ) - ψp + ψq   # logit(yᵢ) - 𝔼(logit(yᵢ))
        z = log1p(-yᵢ) - ψq + ψϕ  # log(1 - yᵢ) - 𝔼(log(1 - yᵢ))
        ∂θ[end] += fma(μᵢ, Δ, z)
        Tr[i] = ϕ * Δ * dμdη
    end
    mul!(view(∂θ, 1:size(X, 2)), X', Tr)
    ∂θ[end] *= dϕ
    return ∂θ
end

# Square root of the diagonal of the weight matrix, W for expected information (pg 7),
# Q for observed information (pg 10). `p = μ * ϕ` and `q = (1 - μ) * ϕ` are the beta
# distribution parameters in the typical parameterization, `ψ′_` is `trigamma(_)`.
function weightdiag(link, p, q, ψ′p, ψ′q, ϕ, yᵢ, ηᵢ, dμdη, expected)
    w = abs(ϕ) * (ψ′p + ψ′q)
    if expected
        return sqrt(w) * abs(dμdη)
    else
        w *= dμdη^2
        w += (logit(yᵢ) - digamma(p) + digamma(q)) * dmueta(link, ηᵢ)
        return sqrt(w)
    end
end

# Fisher information, expected or observed, inverted or not. Used for computing the
# maximum likelihood estimates via Fisher scoring as well as for variance estimation.
# TODO: There's likely plenty of room for implementation improvement here in terms of
# speed and memory use.
function 🐟(b::BetaRegressionModel, expected::Bool, inverse::Bool)
    X = modelmatrix(b)
    T = eltype(X)
    k = length(params(b))
    y = response(b)
    η = linearpredictor(b)
    link = Link(b)
    ϕ, dϕ, _ = precisioninverselink(b)
    ψ′ϕ = trigamma(ϕ)
    Tc = similar(η)
    Tc .= dϕ
    w = similar(η)
    γ = zero(ϕ)
    for i in eachindex(y, η, w)
        ηᵢ = η[i]
        μᵢ, dμdη, omμᵢ = inverselink(link, ηᵢ)
        p = μᵢ * ϕ
        q = omμᵢ * ϕ
        ψ′p = trigamma(p)
        ψ′q = trigamma(q)
        w[i] = weightdiag(link, p, q, ψ′p, ψ′q, ϕ, y[i], ηᵢ, dμdη, expected)
        Tc[i] *= (ψ′p * p - ψ′q * q) * dμdη
        γ += ψ′p * μᵢ^2 + ψ′q * omμᵢ^2 - ψ′ϕ
    end
    γ *= dϕ^2
    Xᵀ = copy(adjoint(X))
    XᵀTc = Xᵀ * Tc
    Xᵀ .*= w'
    if inverse
        XᵀWX = cholesky!(Symmetric(syrk('U', 'N', one(T), Xᵀ)))
        A = XᵀWX \ XᵀTc
        γ -= dot(A, XᵀTc) / ϕ
        # Upper left block
        Kββ = copytri!(syrk('U', 'N', inv(γ * ϕ), XᵀTc), 'U')
        rdiv!(Kββ, XᵀWX)
        for i in axes(Kββ, 1)
            @inbounds Kββ[i, i] += 1
        end
        ldiv!(XᵀWX, Kββ)
        rdiv!(Kββ, ϕ)
        # Upper right and lower left
        Kβϕ = rdiv!(A, -γ * ϕ)
        # Lower right
        Kϕϕ = 1 / γ
    else
        Kββ = syrk('U', 'N', ϕ, Xᵀ)
        Kβϕ = XᵀTc
        Kϕϕ = γ
    end
    K = Matrix{T}(undef, k, k)
    copyto!(view(K, 1:(k - 1), 1:(k - 1)), Symmetric(Kββ))
    copyto!(view(K, 1:(k - 1), k), Kβϕ)
    K[k, k] = Kϕϕ
    return Symmetric(K)
end

"""
    informationmatrix(model::BetaRegressionModel; expected=true)

Compute the information matrix of the model. By default, this is the Fisher information,
i.e. the expected value of the matrix of second partial derivatives of
[`loglikelihood`](@ref) with respect to each element of [`params`](@ref). Set `expected`
to `false` to obtain the observed information.

See also: [`vcov`](@ref), [`score`](@ref)
"""
StatsAPI.informationmatrix(b::BetaRegressionModel; expected::Bool=true) =
    🐟(b, expected, false)

"""
    vcov(model::BetaRegressionModel)

Compute the variance-covariance matrix of the model, i.e. the inverse of the Fisher
information matrix.

See also: [`stderror`](@ref), [`informationmatrix`](@ref)
"""
StatsAPI.vcov(b::BetaRegressionModel) = 🐟(b, true, true)

function checkfinite(x, iters)
    if any(!isfinite, x)
        throw(ConvergenceException(iters, NaN, NaN,
                                   "Coefficient update contains infinite values."))
    end
    return nothing
end

"""
    fit!(b::BetaRegressionModel{T}; maxiter=100, atol=sqrt(eps(T)), rtol=Base.rtoldefault(T))

Fit the given [`BetaRegressionModel`](@ref), updating its values in-place. If model
convergence is achieved, `b` is returned, otherwise a `ConvergenceException` is thrown.

Fitting the model consists of computing the maximum likelihood estimates for the
coefficients and precision parameter via Fisher scoring with analytic derivatives.
The model is determined to have converged when the score vector, i.e. the vector of
first partial derivatives of the log likelihood with respect to the parameters, is
approximately zero. This is determined by `isapprox` using the specified `atol` and
`rtol`. `maxiter` dictates the maximum number of Fisher scoring iterations.
"""
function StatsAPI.fit!(b::BetaRegressionModel{T}; maxiter=100, atol=sqrt(eps(T)),
                       rtol=Base.rtoldefault(T)) where {T}
    initialize!(b)
    θ = params(b)
    z = zero(θ)
    for iter in 1:maxiter
        U = score(b)
        checkfinite(U, iter)
        isapprox(U, z; atol, rtol) && return b  # converged!
        K = 🐟(b, true, true)
        checkfinite(K, iter)
        mul!(θ, K, U, true, true)
        θ[end] = max(θ[end], eps(eltype(θ)))  # impose positivity constraint on ϕ
        linearpredictor!(b)
    end
    throw(ConvergenceException(maxiter))
end

"""
    fit(BetaRegressionModel, formula, data, link=LogitLink(), precisionlink=IdentityLink();
        kwargs...)

Fit a [`BetaRegressionModel`](@ref) to the given table `data`, which may be any
Tables.jl-compatible table (e.g. a `DataFrame`), using the given `formula`, which can
be constructed using `@formula`. In this method, the response and model matrix are
determined from the formula and table. It is also possible to provide them explicitly.

    fit(BetaRegressionModel, X::AbstractMatrix, y::AbstractVector, link=LogitLink(),
        precisionlink=IdentityLink(); kwargs...)

Fit a beta regression model using the provided model matrix `X` and response vector `y`.
In both of these methods, a link function may be provided, otherwise the default logit
link is used.
Similarly, a link for the precision may be provided, otherwise the default identity link
is used.

## Keyword Arguments

- `weights`: A vector of weights or `nothing` (default). Currently only `nothing` is accepted.
- `offset`: An offset vector to be added to the linear predictor or `nothing` (default).
- `maxiter`: Maximum number of Fisher scoring iterations to use when fitting. Default is 100.
- `atol`: Absolute tolerance to use when checking for model convergence. Default is
  `sqrt(eps(T))` where `T` is the type of the estimates.
- `rtol`: Relative tolerance to use when checking for convergence. Default is the Base
  default relative tolerance for `T`.

!!! tip
    If you experience convergence issues, you may consider trying a different link for
    the precision; `LogLink()` is a common choice. Increasing the maximum number of
    iterations may also be beneficial, especially when working with `Float32`.
"""
function StatsAPI.fit(::Type{BetaRegressionModel}, X::AbstractMatrix, y::AbstractVector,
                      link::Link01=LogitLink(), precisionlink::Link=IdentityLink();
                      weights=nothing, offset=nothing, kwargs...)
    b = BetaRegressionModel(X, y, link, precisionlink; weights, offset)
    fit!(b; kwargs...)
    return b
end

# TODO: Move the StatsAPI extensions to the delegations that happen in StatsModels itself
@delegate(TableRegressionModel{<:BetaRegressionModel}.model,
          [Base.precision, GLM.Link, GLM.devresid, StatsAPI.informationmatrix,
           StatsAPI.linearpredictor, StatsAPI.offset, StatsAPI.score, StatsAPI.weights,
           precisionlink])

"""
    responsename(model::TableRegressionModel{<:BetaRegressionModel})

For a `BetaRegressionModel` fit using a table and `@formula`, return a string containing
the left hand side of the formula, i.e. the model's response.
"""
StatsAPI.responsename(m::TableRegressionModel{<:BetaRegressionModel}) =
    sprint(show, formula(m).lhs)

"""
    coefnames(model::TableRegressionModel{<:BetaRegressionModel})

For a `BetaRegressionModel` fit using a table and `@formula`, return the names of the
coefficients as a vector of strings. The precision term is included as the last element
in the array and has name `"(Precision)"`.
"""
StatsAPI.coefnames(m::TableRegressionModel{<:BetaRegressionModel}) =
    vcat(coefnames(m.mf), "(Precision)")

# Define a more specific method than the one in StatsModels since that one calls
# `coeftable` on the model's `ModelFrame` directly and that will have too few
# coefficients
function StatsAPI.coeftable(m::TableRegressionModel{<:BetaRegressionModel}; kwargs...)
    ct = coeftable(m.model; kwargs...)
    ct.rownms = coefnames(m)
    return ct
end

# NOTE: We're applying a custom docstring to a method that is not directly defined,
# which surprisingly works
"""
    stderror(model::BetaRegressionModel)

Return the standard errors of the estimated model parameters, including both the
regression coefficients and the precision.

See also: [`vcov`](@ref)
"""
StatsAPI.stderror(::BetaRegressionModel)

"""
    dmueta(link::Link, η)

Return the second derivative of `linkinv`, ``\\frac{\\partial^2 \\mu}{\\partial \\eta^2}``,
of the link function `link` evaluated at the linear predictor value `η`. A method of this
function must be defined for a particular link function in order to compute the observed
information matrix.
"""
function dmueta end

dmueta(link::CauchitLink, η) = -2π * η * mueta(link, η)^2

dmueta(link::CloglogLink, η) = -expm1(η) * mueta(link, η)

function dmueta(link::LogitLink, η)
    μ, dμdη, _ = inverselink(link, η)
    return dμdη * (1 - 2μ)
end

dmueta(link::ProbitLink, η) = -η * mueta(link, η)

end  # module
