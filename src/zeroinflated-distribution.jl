"""
    ZeroInflatedDistribution{Te, Tp} <: Distributions.Sampleable{Univariate,Continuous}

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

The dispersion parameter is the usual standard deviation on the log scale for
log-normal, standard deviation for the gamma and inverse gamma, and the shape
parameter for the inverse Gaussian distribution.
"""
struct ZeroInflatedDistribution{Te, Tp} <: Distributions.Sampleable{Univariate,Continuous}
    encdist::Te
    posdist::Tp

    function ZeroInflatedDistribution(encdist::Te, posdist::Tp) where
            {Te<:Distributions.Bernoulli,
             Tp<:Distributions.ContinuousUnivariateDistribution}
        new{Te, Tp}(encdist, posdist)
    end
end

function ZeroInflatedDistribution(
    zil::AbstractZeroInflatedLink, posdist::Type{LogNormal}, p1, p2, disp;
    biascorrect = true)
    p = encprob(zil, p1, p2)
    if biascorrect
        bias = disp^2 / 2
    else
        bias = zero(disp)
    end
    r = posrate(zil, p1, p2)
    ZeroInflatedDistribution(Bernoulli(p), posdist(log(r) - bias, disp))
end

function ZeroInflatedDistribution(
    zil::AbstractZeroInflatedLink, posdist::Type{Gamma}, p1, p2, disp)
    p = encprob(zil, p1, p2)
    mu = posrate(zil, p1, p2)
    alpha = (mu / disp)^2
    beta = disp^2 / mu
    ZeroInflatedDistribution(Bernoulli(p), posdist(alpha, beta))
end

function ZeroInflatedDistribution(
    zil::AbstractZeroInflatedLink, posdist::Type{InverseGamma}, p1, p2, disp)
    p = encprob(zil, p1, p2)
    # From mean and standard deviation to alpha and beta:
    # μ = β / (α - 1) for α > 1
    # σ² = β² / [(α - 1)²(α - 2)] for α > 2
    # β = μ(α - 1)
    # σ² = μ² / (α - 2)
    # α = μ² / σ² + 2
    # β = μ(μ² / σ² + 1)
    # β = μ³ / σ² + μ
    mu = posrate(zil, p1, p2)
    alpha = (mu / disp)^2 + 2
    beta = mu * (alpha - 1)
    ZeroInflatedDistribution(Bernoulli(p), posdist(alpha, beta))
end

function ZeroInflatedDistribution(
    zil::AbstractZeroInflatedLink, posdist::Type{InverseGaussian}, p1, p2, disp)
    p = encprob(zil, p1, p2)
    mu = posrate(zil, p1, p2)
    ZeroInflatedDistribution(Bernoulli(p), posdist(mu, disp))
end

# Density and log-likelihood functions: pdf, logpdf, loglikelihood
function Distributions.pdf(zil::ZeroInflatedDistribution, obs)
    dens = pdf(zil.encdist, obs ≠ 0)
    if obs ≠ 0
        dens *= pdf(zil.posdist, obs)
    end
    dens
end
function Distributions.logpdf(zil::ZeroInflatedDistribution, obs)
    logdens = logpdf(zil.encdist, obs ≠ 0)
    if obs ≠ 0
        logdens += logpdf(zil.posdist, obs)
    end
    logdens
end
function Distributions.loglikelihood(zil::ZeroInflatedDistribution, obs)
    loglik = logpdf(zil.encdist, obs ≠ 0)
    if obs ≠ 0
        loglik += logpdf(zil.posdist, obs)
    end
    loglik
end

# CDF functions: cdf
function Distributions.cdf(zil::ZeroInflatedDistribution, obs)
    # Zero has positive probability
    cum = failprob(zil.encdist)
    if obs > 0
        # Then add CDF of positive distribution rescaled to probability of
        # encounter
        cum += cdf(zil.posdist, obs) * succprob(zil.encdist)
    end
    cum
end

# Quantile functions: quantile
function Distributions.quantile(zil::ZeroInflatedDistribution, p)
    p0 = failprob(zil.encdist)
    if obs ≤ p0
        q = zero(p)
    else
        p2 = (p - p0) / succprob(zil.encdist)
        q = quantile(zil.posdist, p2)
    end
    q
end

# FIXME Consider handline Bernoulli(0), Bernoulli(1) corner cases
# Support functions: minimum, maximum, insupport
function Base.minimum(zil::ZeroInflatedDistribution)
    min(minimum(zil.encdist), minimum(zil.posdist))
end
function Base.maximum(zil::ZeroInflatedDistribution)
    max(maximum(zil.encdist), maximum(zil.posdist))
end
function Distributions.insupport(zil::ZeroInflatedDistribution, x)
    minimum(zil) ≤ x ≤ maximum(zil)
end

# Statistics: mean, var, modes, mode, skewness, kurtosis, entropy, mgf, cf
function Statistics.mean(zil::ZeroInflatedDistribution)
    succprob(zil.encdist) * mean(zil.posdist)
end
function Statistics.var(zil::ZeroInflatedDistribution)
    ppos = succprob(zil.encdist)
    varpos = var(zil.posdist)
    meanpos = mean(zil.posdist)
    ppos * (varpos + meanpos^2) - (ppos * meanpos)^2
end
Statistics.std(zil::ZeroInflatedDistribution) = sqrt(var(zil))
# Consider handling this
Distributions.modes(zil::ZeroInflatedDistribution) = (zero(eltype(zil.encdist)), mode(zil.posdist))

# Facilities for random number generation
function Base.rand(rng::AbstractRNG, zil::ZeroInflatedDistribution)
    rand(rng, zil.encdist) * rand(rng, zil.posdist)
end
Base.rand(zil::ZeroInflatedDistribution) = rand(Random.GLOBAL_RNG, zil)
