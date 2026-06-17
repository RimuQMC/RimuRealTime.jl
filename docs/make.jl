using RimuRealTime
using Documenter

DocMeta.setdocmeta!(RimuRealTime, :DocTestSetup, :(using RimuRealTime); recursive=true)

makedocs(;
    modules=[RimuRealTime],
    authors="Joachim Brand <joachim.brand@gmail.com> and contributors",
    sitename="RimuRealTime.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Guide" => "guide.md",
        "API"   => "api.md",
    ],
)

deploydocs(
    repo = "github.com/RimuQMC/RimuRealTime.jl.git",
    push_preview = true,
)
