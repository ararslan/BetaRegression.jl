using BetaRegression
using Documenter

using Documenter: HTML

makedocs(; modules=[BetaRegression],
         sitename="BetaRegression.jl",
         pages=["Home" => "index.md",
                "Details" => "details.md",
                "API" => "api.md"],
         format=HTML(; prettyurls=(get(ENV, "CI", "false") == "true")))

#deploydocs(; repo="github.com/ararslan/BetaRegression.jl.git")
