module IPOMDPToolbox

using IPOMDPs
using POMDPs

export
    # IPOMDP model exploration
    exploreProblem,
    exploreAgent,
    exploreModel,

    # IPOMDP to POMDP conversion
    generatePOMDP


    include("ipomdptoolbox.jl")
    include("convertedpomdp.jl")
end
