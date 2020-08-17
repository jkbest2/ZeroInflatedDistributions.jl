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
    ZeroInflatedDistribution

include("zeroinflated-link-funs.jl")
include("zeroinflated-distribution.jl")
end # module
