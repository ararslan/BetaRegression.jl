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
using GLM: Link01, LmResp, cholpred, inverselink, linkfun, linkinv, linpred!, mueta
using LinearAlgebra: copytri!
using StatsAPI: aic, aicc, bic, offset, params
using StatsModels: TableRegressionModel, @delegate

export
    BetaRegressionModel,
    # Extensions/utilities from GLM:
    CauchitLink,
    CloglogLink,
    Link,
    Link01,
    LogitLink,
    ProbitLink,
    linpred,
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
    BetaRegressionModel{T,L,V,M} <: RegressionModel

Type representing a regression model for beta-distributed response values in the open
interval (0, 1), as described by Ferrari and Cribari-Neto (2004).

The mean response is linked to the linear predictor by a link function with type
`L <: Link01`, i.e. the link must map ``(0, 1) ↦ ℝ`` and use the GLM package's interface
for link functions.
"""
struct BetaRegressionModel{T<:AbstractFloat,L<:Link01,V<:AbstractVector{T},
                           M<:AbstractMatrix{T}} <: RegressionModel
    y::V
    X::M
    weights::Vector{T}
    offset::Vector{T}
    parameters::Vector{T}
    linearpredictor::Vector{T}
end

"""
    BetaRegressionModel(X, y, link=LogitLink(); weights=nothing, offset=nothing)

Construct a `BetaRegressionModel` object with the given model matrix `X`, response
`y`, link function `link`, and optionally `weights` and `offset`.

!!! warn
    Support for user-provided weights is currently incomplete; passing a value other
    than `nothing` or an empty array for `weights` will result in an error for now.
"""
function BetaRegressionModel(X::AbstractMatrix, y::AbstractVector,
                             link::Link01=LogitLink();
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
    return BetaRegressionModel{T,typeof(link),typeof(_y),typeof(_X)}(y, X, weights, offset,
                                                                     parameters, η)
end

StatsAPI.response(b::BetaRegressionModel) = b.y

StatsAPI.modelmatrix(b::BetaRegressionModel) = b.X

StatsAPI.weights(b::BetaRegressionModel) = b.weights

StatsAPI.offset(b::BetaRegressionModel) = b.offset

StatsAPI.params(b::BetaRegressionModel) = b.parameters

StatsAPI.coef(b::BetaRegressionModel) = params(b)[1:(end - 1)]

"""
    precision(model::BetaRegressionModel)

Return the estimated precision parameter, ``\\phi``, for the model.
This parameter is estimated alongside the regression coefficients and is included in
coefficient tables.

See also: `coef`, `params`
"""
Base.precision(b::BetaRegressionModel) = params(b)[end]

GLM.linpred(b::BetaRegressionModel) = b.linearpredictor

function GLM.linpred!(b::BetaRegressionModel)
    X = modelmatrix(b)
    β = view(params(b), 1:size(X, 2))
    η = linpred(b)
    if isempty(offset(b))
        mul!(η, X, β)
    else
        copyto!(η, offset(b))
        mul!(η, X, β, true, true)
    end
    return η
end

StatsAPI.fitted(b::BetaRegressionModel) = linkinv.(Link(b), linpred(b))

StatsAPI.residuals(b::BetaRegressionModel) = response(b) .- fitted(b)

StatsAPI.nobs(b::BetaRegressionModel) =
    isempty(weights(b)) ? length(response(b)) : count(>(0), weights(b))

StatsAPI.dof(b::BetaRegressionModel) = length(params(b))

StatsAPI.dof_residual(b::BetaRegressionModel) = nobs(b) - dof(b)

StatsAPI.r2(b::BetaRegressionModel) = cor(linpred(b), linkfun.(Link(b), response(b)))^2

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

GLM.Link(b::BetaRegressionModel{T,L}) where {T,L} = L()

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
    η = linpred(b)[i]
    μ = linkinv(Link(b), η)
    ϕ = precision(b)
    ℓ = betalogpdf(μ, ϕ, y[i])
    isempty(weights(b)) || (ℓ *= weights(b)[i])
    return ℓ
end

function GLM.devresid(b::BetaRegressionModel)
    ϕ = precision(b)
    return map(response(b), fitted(b)) do y, μ
        r = y - μ
        ℓ₁ = betalogpdf(y, ϕ, y)
        ℓ₂ = betalogpdf(μ, ϕ, y)
        return sign(r) * sqrt(2 * abs(ℓ₁ - ℓ₂))
    end
end

StatsAPI.deviance(b::BetaRegressionModel) = sum(abs2, devresid(b))

# Initialization method as recommended at the end of Ferrari (2004), section 2
"""
    initialize!(b::BetaRegressionModel)

Initialize the given [`BetaRegressionModel`](@ref) by computing starting points for
the parameter estimates and return the updated model object.
The initial estimates are based on those from a linear regression model with the same
model matrix as `b` but with `linkfun.(Link(b), response(b))` as the response.
"""
function initialize!(b::BetaRegressionModel)
    link = Link(b)
    X = modelmatrix(b)
    y = linkfun.(link, response(b))
    # We have to use the constructors directly because `LinearModel` supports an
    # offset but it isn't exposed by `lm`
    model = LinearModel(LmResp{typeof(y)}(zero(y), offset(b), weights(b), y),
                        cholpred(X, true))
    fit!(model)
    β = coef(model)
    η = fitted(model)
    μ = linkinv.(link, η)
    e = residuals(model)
    n = nobs(model)
    k = length(β)
    σ² = sum(abs2, e) .* abs2.(mueta.(link, η)) ./ (n .- k)
    ϕ = mean(i -> μ[i] * (1 - μ[i]) / σ²[i] - 1, eachindex(μ, σ²))
    copyto!(params(b), push!(β, ϕ))
    copyto!(linpred(b), η)
    return b
end

function StatsAPI.score(b::BetaRegressionModel)
    X = modelmatrix(b)
    y = response(b)
    η = linpred(b)
    link = Link(b)
    ϕ = precision(b)
    ψϕ = digamma(ϕ)
    ∂θ = zero(params(b))
    Tr = copy(η)
    @inbounds for i in eachindex(y, η)
        ηᵢ = η[i]
        μᵢ = linkinv(link, ηᵢ)
        yᵢ = y[i]
        a = digamma((1 - μᵢ) * ϕ)
        r = logit(yᵢ) - digamma(μᵢ * ϕ) + a
        ∂θ[end] += μᵢ * r + log(1 - yᵢ) - a + ψϕ
        Tr[i] = ϕ * r * mueta(link, ηᵢ)
    end
    mul!(view(∂θ, 1:size(X, 2)), X', Tr)
    return ∂θ
end

# Square root of the diagonal of the weight matrix, W for expected information (pg 7),
# Q for observed information (pg 10). `p = μ * ϕ` and `q = (1 - μ) * ϕ` are the beta
# distribution parameters in the typical parameterization, `ψ′_` is `trigamma(_)`.
function weightdiag(link, p, q, ψ′p, ψ′q, ϕ, yᵢ, ηᵢ, dμdη, expected)
    w = ϕ * (ψ′p + ψ′q)
    if expected
        return sqrt(w) * dμdη
    else
        w *= dμdη^2
        ystar = logit(yᵢ)
        μstar = digamma(p) - digamma(q)
        w += (ystar - μstar) * dmueta(link, ηᵢ)
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
    η = linpred(b)
    link = Link(b)
    ϕ = precision(b)
    ψ′ϕ = trigamma(ϕ)
    Tc = similar(η)
    w = similar(η)
    γ = zero(ϕ)
    for i in eachindex(y, η, w)
        ηᵢ = η[i]
        μᵢ = linkinv(link, ηᵢ)
        p = μᵢ * ϕ
        q = (1 - μᵢ) * ϕ
        ψ′p = trigamma(p)
        ψ′q = trigamma(q)
        dμdη = mueta(link, ηᵢ)
        w[i] = weightdiag(link, p, q, ψ′p, ψ′q, ϕ, y[i], ηᵢ, dμdη, expected)
        Tc[i] = (ψ′p * p - ψ′q * q) * dμdη
        γ += ψ′p * μᵢ^2 + ψ′q * (1 - μᵢ)^2 - ψ′ϕ
    end
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

StatsAPI.informationmatrix(b::BetaRegressionModel; expected::Bool=true) =
    🐟(b, expected, false)

StatsAPI.vcov(b::BetaRegressionModel) = 🐟(b, true, true)

function checkfinite(x, iters)
    if any(!isfinite, x)
        throw(ConvergenceException(iters, NaN, NaN,
                                   "Coefficient update contains infinite values."))
    end
    return nothing
end

"""
    fit!(b::BetaRegressionModel; maxiter=100, atol=1e-8, rtol=1e-8)

Fit the given [`BetaRegressionModel`](@ref), updating its values in-place. If model
convergence is achieved, `b` is returned, otherwise a `ConvergenceException` is thrown.

Fitting the model consists of computing the maximum likelihood estimates for the
coefficients and precision parameter via Fisher scoring with analytic derivatives.
The model is determined to have converged when the score vector, i.e. the vector of
first partial derivatives of the log likelihood with respect to the parameters, is
approximately zero. This is determined by `isapprox` using the specified `atol` and
`rtol`. `maxiter` dictates the maximum number of Fisher scoring iterations.
"""
function StatsAPI.fit!(b::BetaRegressionModel; maxiter=100, atol=1e-8, rtol=1e-8)
    initialize!(b)
    z = zero(params(b))
    for iter in 1:maxiter
        U = score(b)
        checkfinite(U, iter)
        isapprox(U, z; atol, rtol) && return b  # converged!
        K = 🐟(b, true, true)
        checkfinite(K, iter)
        mul!(params(b), K, U, true, true)
        linpred!(b)
    end
    throw(ConvergenceException(maxiter))
end

"""
    fit(BetaRegressionModel, formula, data, link=LogitLink(); kwargs...)

Fit a [`BetaRegressionModel`](@ref) to the given table `data`, which may be any
Tables.jl-compatible table (e.g. a `DataFrame`), using the given `formula`, which can
be constructed using `@formula`. In this method, the response and model matrix are
determined from the formula and table. It is also possible to provide them explicitly.

    fit(BetaRegressionModel, X::AbstractMatrix, y::AbstractVector, link=LogitLink(); kwargs...)

Fit a beta regression model using the provided model matrix `X` and response vector `y`.
In both of these methods, a link function may be provided. If left unspecified, a
logit link is used.

## Keyword Arguments

- `weights`: A vector of weights or `nothing` (default). Currently only `nothing` is accepted.
- `offset`: An offset vector to be added to the linear predictor or `nothing` (default).
- `maxiter`: Maximum number of Fisher scoring iterations to use when fitting. Default is 100.
- `atol`: Absolute tolerance to use when checking for model convergence. Default is 1e-8.
- `rtol`: Relative tolerance to use when checking for convergence. Default is also 1e-8.
"""
function StatsAPI.fit(::Type{BetaRegressionModel}, X::AbstractMatrix, y::AbstractVector,
                      link=LogitLink(); weights=nothing, offset=nothing, maxiter=100,
                      atol=1e-8, rtol=1e-8)
    b = BetaRegressionModel(X, y, link; weights, offset)
    fit!(b; maxiter, atol, rtol)
    return b
end

# TODO: Move the StatsAPI extensions to the delegations that happen in StatsModels itself
@delegate(TableRegressionModel{<:BetaRegressionModel}.model,
          [Base.precision, GLM.Link, GLM.devresid, GLM.linpred, StatsAPI.informationmatrix,
           StatsAPI.score, StatsAPI.weights])

StatsAPI.responsename(m::TableRegressionModel{<:BetaRegressionModel}) =
    sprint(show, formula(m).lhs)

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

"""
    dmueta(link::Link, η)

Return the second derivative of `linkinv`, ``d²μ/dη²``, of the link function `link`
evaluated at the linear predictor value `η`. A method of this function must be defined
for a particular link function in order to compute the observed information matrix.
"""
function dmueta end

dmueta(link::CauchitLink, η) = -2π * η * mueta(link, η)^2

dmueta(link::CloglogLink, η) = -expm1(η) * mueta(link, η)

function dmueta(link::LogitLink, η)
    μ, _, dμdη = inverselink(link, η)
    return dμdη * (1 - 2μ)
end

dmueta(link::ProbitLink, η) = -η * mueta(link, η)

end  # module
