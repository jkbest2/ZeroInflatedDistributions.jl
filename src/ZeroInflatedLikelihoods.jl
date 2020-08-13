module ZeroInflatedLikelihoods

using Distributions
using StatsFuns

abstract type AbstractZeroInflatedLink end
abstract type AbstractZeroInflatedLikelihood end

struct LogitLogLink <: AbstractZeroInflatedLink end

encprob(lll::LogitLogLink, proc1) = logistic(proc1)
encprob(lll::LogitLogLink, proc1, proc2) = encprob(lll, proc1)
posrate(lll::LogitLogLink, proc2::T; bias = zero{T}) where T = exp(proc2) - bias
posrate(lll::LogitLogLink, proc1::T, proc2::T; bias = zero(T)) where T = posrate(lll, proc2; bias)

struct PoissonLink <: AbstractZeroInflatedLink end

encprob(pl::PoissonLink, numdens) = cdf(Poisson(exp(numdens)), 0)
encprob(pl::PoissonLink, numdens, wtgrp) = encprob(pl, numdens)
function posrate(pl::PoissonLink, numdens::T, wtgrp::T; bias = zero{T}) where T
    exp(numdens + wtgrp) / encprob(pl, numdens) - bias
end

struct IdentityLink <: AbstractZeroInflatedLink end

encprob(il::IdentityLink, proc1) = proc1
encprob(il::IdentityLink, proc1, proc2) = encprob(il, proc1)
posrate(il::IdentityLink, proc2::T; bias = zero(T)) where T = proc2 - bias
posrate(il::IdentityLink, proc1::T, proc2::T; bias = zero(T)) where T = posrate(il, proc2; bias)


struct ZeroInflatedLikelihood{Tpos, Tp, Tl, Td} <: AbstractZeroInflatedLikelihood
    posdist::Td
    proc1::Tp
    proc2::Tp
    meanbias::Tp
    link::Tl
    dispersion::T
    
    function ZeroInflatedLikelihood(
        posdist::Type{Tpos},
        proc1, proc2, meanbias,
        link, dispersion) where
            {Tpos<:Distributions.ContinuousUnivariateDistribution,
             Tl<:AbstractZeroInflatedLink}
        length(proc1) == length(proc2) ||
            throw(DimensionMismatch("proc1 and proc2 must be the same length"))
        new(posdist, proc1, proc2, meanbias, link, dispersion)
    end
end

function ZeroInflatedLikelihood(zil::ZeroInflatedLikelihood, proc1, proc2)
    ZeroInflatedLikelihood(zil.posdist, proc1, proc2, zil.link, zil.dispersion)
end

length(zil::ZeroInflatedLikelihood) = length(zil.proc1)
function encprob(zil::ZeroInflatedLikelihood)
    encprob.(Ref(zil.link), zil.proc1, zil.proc2)
end
function posrate(zil::ZeroInflatedLikelihood)
    posrate.(Ref(zil.link), zil.proc1, zil.proc2)
end

function loglikelhood(zil::ZeroInflatedLikelihood, obs)
    p = encprob(zil)
    r = posrate(zil)

    loglik = zero()
    for i in 1:length(zil)
        global loglik += logpdf(Bernoulli(p[i]), obs[i] ≠  0)
        if obs[i] ≠ 0
            global loglik += logpdf(zil.posdist(r[i], σ), obs[i])
        end
    end

    loglik
end
