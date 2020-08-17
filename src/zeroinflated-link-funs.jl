abstract type AbstractZeroInflatedLink end

"""
    encprob(link::AbstractZeroInflatedLink, proc1, proc2)

Probability of encounter (non-zero observation) given linear predictors proc1
and proc2. Depending on the link, proc2 may not be required as an
argument.
"""
function encprob end

"""
    posrate(link::AbstractZeroInflatedLink, proc1, proc2)

Expected positive observation conditional on an encounter given linear
predictors proc1 and proc2. Depending on the link, proc1 may be dropped as an
argument.
"""
function posrate end

"""
    LogitLogLink <: AbstractZeroInflatedLink

A link function where for observation y modeled by linear predictors p₁ and p₂,

p(y > 0) = logit⁻¹(p₁)

𝔼[y ∣ y > 0] = exp(p₂) - b
"""
struct LogitLogLink <: AbstractZeroInflatedLink end

encprob(lll::LogitLogLink, proc1) = logistic(proc1)
encprob(lll::LogitLogLink, proc1, proc2) = encprob(lll, proc1)
posrate(lll::LogitLogLink, proc2) = exp(proc2)
posrate(lll::LogitLogLink, proc1, proc2) = posrate(lll, proc2)

"""
    PoissonLink <: AbstractZeroInflatedLink

The Poisson link function of Thorson (2018), where for observation y modeled by
linear predictors n (log numbers-density) and w (log weight-per-group),

p(y > 0) = 1 - exp(-a exp(n))

𝔼[y ∣ y > 0] = exp(n) exp(w) / p(y > 0)

where a is an optional offset (e.g. area swept).

Encounter probability uses the complementary log-log link. This link
approximates a compound Poisson-gamma (Tweedie with 1 < p < 2) distribution.
This implies that number of groups sampled is drawn from a Poisson distribution
with mean exp(n) and each group has expected weight exp(w).
"""
struct PoissonLink{T} <: AbstractZeroInflatedLink
    offset::T

    function PoissonLink(offset::T) where T
        offset > 0 || throw(DomainError("offset must be positive"))
        new{T}(offset)
    end
end
function PoissonLink()
    # Default to integer offset so that it doesn't inadvertantly convert e.g.
    # Float32 to Float64
    PoissonLink(one(Int))
end

encprob(pl::PoissonLink, logn) = ccdf(Poisson(pl.offset * exp(logn)), 0)
encprob(pl::PoissonLink, logn, logw) = encprob(pl, logn)
posrate(pl::PoissonLink, logn, logw) = exp(logn + logw) / encprob(pl, logn)

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
posrate(il::IdentityLink, proc2) = proc2
posrate(il::IdentityLink, proc1, proc2) = posrate(il, proc2)
