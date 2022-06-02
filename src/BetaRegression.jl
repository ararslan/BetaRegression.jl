module BetaRegression

using Distributions
using GLM
using LinearAlgebra
using LinearAlgebra.BLAS
using LogExpFunctions
using SpecialFunctions
using StatsAPI
using StatsBase
using StatsModels

using GLM: Link01, LmResp, cholpred, dispersion, inverselink, linkfun, linkinv, mueta  # not exported
using LinearAlgebra: dot  # shadow the one from BLAS
using StatsAPI: meanresponse, params  # not exported nor reexported from elsewhere
using StatsModels: TableRegressionModel, @delegate, termvars  # not exported

export
    BetaRegressionModel,
    # Extensions/utilities from GLM:
    CauchitLink,
    CloglogLink,
    Link,
    Link01,
    LogitLink,
    ProbitLink,
    dispersion,
    linpred,
    # Extensions from StatsAPI:
    coef,
    fit!,
    fit,
    informationmatrix,
    loglikelihood,
    meanresponse,
    modelmatrix,
    nobs,
    response,
    score,
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
end
# TODO: Should probably store `η` in the model object too

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
    _X = convert(AbstractMatrix{T}, X)
    _y = convert(AbstractVector{T}, y)
    return BetaRegressionModel{T,typeof(link),typeof(_y),typeof(_X)}(y, X, weights, offset,
                                                                     parameters)
end

StatsAPI.response(b::BetaRegressionModel) = b.y

StatsAPI.modelmatrix(b::BetaRegressionModel) = b.X

StatsAPI.weights(b::BetaRegressionModel) = b.weights

StatsAPI.params(b::BetaRegressionModel) = b.parameters

StatsAPI.coef(b::BetaRegressionModel) = params(b)[1:(end - 1)]

GLM.dispersion(b::BetaRegressionModel) = params(b)[end]

function GLM.linpred(b::BetaRegressionModel)
    X = modelmatrix(b)
    β = coef(b)
    η = X * β
    if !isempty(b.offset)
        η .+= b.offset
    end
    return η
end

StatsAPI.meanresponse(b::BetaRegressionModel, η=linpred(b)) = linkinv.(Link(b), η)

StatsAPI.nobs(b::BetaRegressionModel) =
    isempty(weights(b)) ? length(response(b)) : count(>(0), weights(b))

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
                     push!(map(i -> "x$i", 1:(length(θ) - 1)), "(Dispersion)"), 4, 3)
end

function StatsAPI.confint(b::BetaRegressionModel; level::Real=0.95)
    θ = params(b)
    se = stderror(b)
    side = se .* quantile(Normal(), (1 - level) / 2)
    return hcat(θ .+ side, θ .- side)
end

function StatsAPI.loglikelihood(b::BetaRegressionModel)
    y = response(b)
    μ = meanresponse(b)
    ϕ = dispersion(b)
    w = weights(b)
    if isempty(w)
        return sum(i -> logpdf(Beta(μ[i] * ϕ, (1 - μ[i]) * ϕ), y[i]), eachindex(y, μ))
    else
        return sum(i -> w[i] * logpdf(Beta(μ[i] * ϕ, (1 - μ[i]) * ϕ), y[i]),
                   eachindex(y, μ, w))
    end
end

function StatsAPI.loglikelihood(b::BetaRegressionModel, ::Colon)
    y = response(b)
    μ = meanresponse(b)
    ϕ = dispersion(b)
    w = weights(b)
    if isempty(w)
        return map((yᵢ, μᵢ) -> logpdf(Beta(μᵢ * ϕ, (1 - μᵢ) * ϕ), yᵢ), y, μ)
    else
        return map((yᵢ, μᵢ, wᵢ) -> wᵢ * logpdf(Beta(μᵢ * ϕ, (1 - μᵢ) * ϕ), yᵢ), y, μ, w)
    end
end

function StatsAPI.loglikelihood(b::BetaRegressionModel, i::Integer)
    y = response(b)
    @boundscheck checkbounds(y, i)
    η = dot(view(modelmatrix(b), i, :), coef(b))
    isempty(b.offset) || (η += b.offset[i])
    μ = linkinv(Link(b), η)
    ϕ = dispersion(b)
    ℓ = logpdf(Beta(μ * ϕ, (1 - μ) * ϕ), y[i])
    isempty(weights(b)) || (ℓ *= weights(b)[i])
    return ℓ
end

# Initialize the coefficients based on the recommendations at the end of section 2:
# perform an OLS regression on g(y)
function initialize!(b::BetaRegressionModel)
    link = Link(b)
    X = modelmatrix(b)
    y = linkfun.(link, response(b))
    # We have to use the constructors directly because `LinearModel` supports an
    # offset but it isn't exposed by `lm`
    model = LinearModel(LmResp{typeof(y)}(zero(y), b.offset, weights(b), y),
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
    copyto!(b.parameters, push!(β, ϕ))
    return b
end

function StatsAPI.score(b::BetaRegressionModel)
    X = modelmatrix(b)
    y = response(b)
    η = linpred(b)
    link = Link(b)
    ϕ = dispersion(b)
    ψϕ = digamma(ϕ)
    ∂β = zero(coef(b))
    ∂ϕ = zero(ψϕ)
    for i in eachindex(y, η)
        ηᵢ = η[i]
        μᵢ = linkinv(link, ηᵢ)
        yᵢ = y[i]
        a = digamma((1 - μᵢ) * ϕ)
        r = logit(yᵢ) - digamma(μᵢ * ϕ) + a
        ∂ϕ += μᵢ * r + log(1 - yᵢ) - a + ψϕ
        η[i] = ϕ * r * mueta(link, ηᵢ)  # reusing `η` as scratch space
    end
    mul!(∂β, X', η)
    return push!(∂β, ∂ϕ)
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
    ϕ = dispersion(b)
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
        Kββ = XᵀTc * adjoint(XᵀTc)
        rdiv!(Kββ, XᵀWX)
        rdiv!(Kββ, γ * ϕ)
        Kββ[diagind(Kββ)] .+= 1
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
coefficients and the common dispersion via Fisher scoring with analytic derivatives.
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
    end
    throw(ConvergenceException(maxiter))
end

function StatsAPI.fit(::Type{BetaRegressionModel}, X::AbstractMatrix, y::AbstractVector,
                      link=LogitLink(); weights=nothing, offset=nothing, maxiter=100,
                      atol=1e-8, rtol=1e-8)
    b = BetaRegressionModel(X, y, link; weights, offset)
    fit!(b; maxiter, atol, rtol)
    return b
end

# TODO: Move the StatsAPI extensions to the delegations that happen in StatsModels itself
@delegate(TableRegressionModel{<:BetaRegressionModel}.model,
          [GLM.Link, GLM.dispersion, GLM.linpred, StatsAPI.meanresponse,
           StatsAPI.informationmatrix, StatsAPI.score, StatsAPI.weights])

function StatsAPI.responsename(m::TableRegressionModel{<:BetaRegressionModel})
    lhs = formula(m).lhs
    y = only(termvars(lhs))
    return String(y)
end

function StatsAPI.coefnames(m::TableRegressionModel{<:BetaRegressionModel})
    names = coefnames(m.mf)
    return vcat(names, "(Dispersion)")
end

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
