# ZeroInflatedDistributions

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jkbest2.github.io/ZeroInflatedDistributions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jkbest2.github.io/ZeroInflatedDistributions.jl/dev)
[![Build Status](https://github.com/jkbest2/ZeroInflatedDistributions.jl/workflows/CI/badge.svg)](https://github.com/jkbest2/ZeroInflatedDistributions.jl/actions)
[![Coverage](https://codecov.io/gh/jkbest2/ZeroInflatedDistributions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jkbest2/ZeroInflatedDistributions.jl)

This package defines zero-inflated distributions. It is still under development
and has a bias toward non-negative continuous observations with many exact
zeros. That said, there is (probably) nothing stopping this being used for e.g.
a zero-inflated Poisson distribution, but this is not currently tested. Link
functions used for defining models with two processes that jointly model
probability of encounter and positive rate are also defined.
