using ZeroInflatedLikelihoods
using Documenter

makedocs(;
    modules=[ZeroInflatedLikelihoods],
    authors="John Best <jkbest@gmail.com> and contributors",
    repo="https://github.com/jkbest2/ZeroInflatedLikelihoods.jl/blob/{commit}{path}#L{line}",
    sitename="ZeroInflatedLikelihoods.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jkbest2.github.io/ZeroInflatedLikelihoods.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jkbest2/ZeroInflatedLikelihoods.jl",
)
