using ZeroInflatedDistributions
using Documenter

makedocs(;
    modules=[ZeroInflatedDistributions],
    authors="John Best <jkbest@gmail.com> and contributors",
    repo="https://github.com/jkbest2/ZeroInflatedDistributions.jl/blob/{commit}{path}#L{line}",
    sitename="ZeroInflatedDistributions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jkbest2.github.io/ZeroInflatedDistributions.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jkbest2/ZeroInflatedDistributions.jl",
)
