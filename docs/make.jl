using Documenter
using ForsetiRegression

DocMeta.setdocmeta!(ForsetiRegression, :DocTestSetup, :(using ForsetiRegression); recursive = true)

makedocs(;
    sitename = "ForsetiRegression.jl",
    modules = [ForsetiRegression],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = ["Home" => "index.md"],
)

deploydocs(;
    repo = "github.com/Forseti-jl/ForsetiRegression.jl.git",
    devbranch = "main",
)
