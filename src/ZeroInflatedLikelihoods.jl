module ZeroInflatedLikelihoods

using Distributions
using StatsFuns
using Random

export AbstractZeroInflatedLink,
    LogitLogLink,
    PoissonLink,
    IdentityLink,
    encprob,
    posrate,
    ZeroInflatedLikelihood

abstract type AbstractZeroInflatedLink end
# Bias is not stored in any of the link structs because it usualy depends on the
# observation likelihood and its parameters (e.g. σ² / 2 for log-normal).

"""
    encprob(link::AbstractZeroInflatedLink, proc1, proc2)

Probability of encounter (non-zero observation) given linear predictors proc1
and proc2. Depending on the link, proc2 may not be required as an
argument.
"""
function encprob end

"""
    posrate(link::AbstractZeroInflatedLink, proc1, proc2; bias = zero(T))

Expected positive observation conditional on an encounter given linear
predictors proc1 and proc2. Depending on the link, proc1 may be dropped as an
argument. The bias keyword argument is *subtracted* from the calculated value.
This is useful e.g. when modeling the mean rather than median of log-normal
positive observations.
"""
function posrate end

"""
    LogitLogLink <: AbstractZeroInflatedLink

A link function where for observation y modeled by linear predictors p₁ and p₂,

p(y > 0) = logit⁻¹(p₁)

𝔼[y ∣ y > 0] = exp(p₂) - b

where b is an optional bias term.
"""
struct LogitLogLink <: AbstractZeroInflatedLink end

encprob(lll::LogitLogLink, proc1) = logistic(proc1)
encprob(lll::LogitLogLink, proc1, proc2) = encprob(lll, proc1)
posrate(lll::LogitLogLink, proc2::T; bias = zero{T}) where T = exp(proc2) - bias
posrate(lll::LogitLogLink, proc1::T, proc2::T; bias = zero(T)) where T = posrate(lll, proc2; bias)

"""
    PoissonLink <: AbstractZeroInflatedLink

The Poisson link function of Thorson (2018), where for observation y modeled by
linear predictors n (log numbers-density) and w (log weight-per-group),

p(y > 0) = 1 - exp(-a exp(n))

𝔼[y ∣ y > 0] = exp(n) exp(w) / p(y > 0)

where a is an optional offset (e.g. area swept) and b is an optional bias term.

Encounter probability uses the complementary log-log link. This link
approximates a compound Poisson-gamma (Tweedie with 1 < p < 2) distribution.
This implies that number of groups sampled is drawn from a Poisson distribution
with mean exp(n) and each group has expected weight exp(w).
"""
struct PoissonLink{T} <: AbstractZeroInflatedLink
    offset::T

    function PoissonLink(offset)
        offset > 0 || throw(DomainError("offset must be positive"))
        new{T}(offset)
    end
end


function encprob(pl::PoissonLink, logn)
    cdf(Poisson(pl.offset * exp(logn)), 0)
end
function encprob(pl::PoissonLink, logn, logw)
    encprob(pl, logn)
end
function posrate(pl::PoissonLink, logn::T, logw::T; bias = zero(T)) where T
    exp(logn + logw) / encprob(pl, logn) - bias
end

"""
    IdentityLink <: AbstractZeroInflatedLink

A simple link function where the first process is the probability of encounter
and the second is expected non-zero observation.
"""
struct IdentityLink <: AbstractZeroInflatedLink end

function encprob(il::IdentityLink, proc1)
    0 < proc1 < 1 || throw(DomainError("proc1 must be between zero and one."))
    proc1
end
encprob(il::IdentityLink, proc1, proc2) = encprob(il, proc1)
posrate(il::IdentityLink, proc2::T; bias = zero(T)) where T = proc2
function posrate(il::IdentityLink, proc1::T, proc2::T; bias = zero(T)) where T
    posrate(il, proc2)
end

abstract type AbstractZeroInflatedLikelihood end

"""
    ZeroInflatedLikelihood{Te, Tp} <: AbstractZeroInflatedLikelihood

Inner constructor:
    - encdist::Bernoulli : Bernoulli distribution with probability of encounter
    - posdist::Distributions.ContinuousUnivariateDistribution : Continuous
            distribution with distribution of observations conditional on an
            encounter

Outer constructor:
    - zil::ZeroInflatedLink : Link function
    - posdist::Type{ContinuousUnivariateDistribution} : Type of the distribution
        of observations conditional on an encounter
    - p1 : First linear predictor
    - p2 : Second linear predictor
    - disp : Positive observation dispersion parameter (standard deviation)
    - biascorrect::Bool : For log-normal positive observations, parameterize by
        the mean rather than the median

A likelihood with positive probability of a zero observation as well as a
continuous response. Generally determined by two processes and a relevant link
function. Outer constructors are currently implemented for the following
distributions:

    - LogNormal
    - Gamma
    - InverseGamma
    - InverseGaussian
"""
struct ZeroInflatedLikelihood{Te, Tp} <: AbstractZeroInflatedLikelihood
    encdist::Te
    posdist::Tp

    function ZeroInflatedLikelihood(encdist::Te, posdist::Tp) where
            {Te<:Distributions.Bernoulli,
             Tp<:Distributions.ContinuousUnivariateDistribution}
        new{Te, Tp}(encdist, posdist)
    end
end

function ZeroInflatedLikelihood(
    zil::AbstractZeroInflatedLink, posdist::Type{LogNormal}, p1, p2, disp;
    biascorrect = true) where {Tp <:Type{LogNormal},T}
    p = encprob(zil, p1, p2)
    if biascorrect
        bias = disp^2 / 2
    else
        bias = zero(disp)
    end
    r = posrate(zil, p1, p2; bias)
    ZeroInflatedLikelihood(Bernoulli(p), posdist(r, disp))
end

function ZeroInflatedLikelihood(
    zil::AbstractZeroInflatedLink, posdist::Type{Gamma}, p1, p2, disp)
    p = encprob(zil, p1, p2)
    mu = posrate(zil, p1, p2)
    alpha = (mu / disp)^2
    beta = disp^2 / mu
    ZeroInflatedLikelihood(Bernoulli(p), posdist(alpha, beta))
end

"""
μ = β / (α - 1) for α > 1
σ² = β² / [(α - 1)²(α - 2)] for α > 2
β = μ(α - 1)
σ² = μ² / (α - 2)
α = μ² / σ² + 2
β = μ(μ² / σ² + 1)
β = μ³ / σ² + μ
"""
function ZeroInflatedLikelihood(
    zil::AbstractZeroInflatedLink, posdist::Type{InverseGamma}, p1, p2, disp)
    p = encprob(zil, p1, p2)
    mu = posrate(zil, p1, p2)
    alpha = (mu / disp)^2 + 2
    beta = mu * (alpha - 1)
    ZeroInflatedLikelihood(Bernoulli(p), posdist(alpha, beta))
end

function ZeroInflatedLikelihood(
    zil::AbstractZeroInflatedLink, posdist::Type{InverseGaussian}, p1, p2, disp)
    p = encprob(zil, p1, p2)
    mu = posrate(zil, p1, p2)
    ZeroInflatedLikelihood(Bernoulli(p), posdist(mu, disp))
end

function Distributions.loglikelihood(zil::ZeroInflatedLikelihood, obs)
    loglik = logpdf(zil.encdist, obs ≠ 0)
    if obs ≠ 0
        loglik += logpdf(zil.posdist, obs)
    end
    loglik
end

# FIXME Better to define two samplers? One the multiplies like this and one that
# branches to only generate a positive if there is an encounter?
function Base.rand(rng::Random.AbstractRNG, zil::ZeroInflatedLikelihood)
    rand(rng, zil.encdist) * rand(rng, zil.posdist)
end
Base.rand(zil::ZeroInflatedLikelihood) = rand(Random._GLOBAL_RNG, zil)

end # module