using BetaRegression
using Documenter

using Documenter: HTML

makedocs(; sitename="BetaRegression.jl",
         pages=["Home" => "index.md",
                "API" => "api.md",
                "Details" => "details.md"],
         format=HTML(; prettyurls=(get(ENV, "CI", "false") == "true")))

deploydocs(; repo="github.com/ararslan/BetaRegression.jl.git")
