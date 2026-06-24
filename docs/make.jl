using RimuRealTime
using Documenter
using DocumenterInterLinks

DocMeta.setdocmeta!(RimuRealTime, :DocTestSetup, :(using RimuRealTime); recursive=true)

links = InterLinks(
    "Rimu" => "https://rimuqmc.github.io/Rimu.jl/stable/objects.inv"
)

makedocs(;
    modules=[RimuRealTime],
    authors="Joachim Brand <joachim.brand@gmail.com> and contributors",
    sitename="RimuRealTime.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Guide" => "index.md",
        "User Documentation" => [
            "Real-time Dynamics" => "realtime.md",
            "Evolution Strategies" => "evolution.md",
            "Clock Hamiltonian" => "clock.md",
            "Population Control" => "population.md",
        ],
        "API"   => "api.md",
    ],
    plugins = [links]
)

deploydocs(
    repo = "github.com/RimuQMC/RimuRealTime.jl.git",
    push_preview = true,
)



