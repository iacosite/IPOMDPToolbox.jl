module IPOMDPToolbox

using IPOMDPs
using POMDPs
using SARSOP
using BeliefUpdaters

export
    # IPOMDP model exploration
    exploreProblem,
    exploreAgent,
    exploreFrame,

    # IPOMDP to POMDP conversion
    generalPOMDP,
    generatePOMDP


    include("ipomdptoolbox.jl")
    include("convertedpomdp.jl")
end
