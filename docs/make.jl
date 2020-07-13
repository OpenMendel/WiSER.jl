using Documenter, WiSER

makedocs(
    format = Documenter.HTML(),
    sitename = "WiSER.jl",
    authors = "Hua Zhou, Chris German",
    clean = true,
    debug = true,
    pages = [
        "index.md"
    ]
)

deploydocs(
    repo   = "github.com/OpenMendel/WiSER.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)
