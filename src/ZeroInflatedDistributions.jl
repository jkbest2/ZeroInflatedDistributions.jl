module ZeroInflatedDistributions

using Random
using Statistics

using Distributions
using StatsFuns

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
