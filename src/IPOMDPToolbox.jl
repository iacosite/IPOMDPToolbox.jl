module IPOMDPToolbox

export
    # IPOMDP model exploration
    exploreAgent,
    exploreModel,

    # IPOMDP to POMDP conversion
    cPOMDP


    include("ipomdptoolbox.jl")
    include("convertedpomdp.jl")
end
