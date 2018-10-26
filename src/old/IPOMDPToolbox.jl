module IPOMDPToolbox

using IPOMDPs
using POMDPs
using BeliefUpdaters

export
    # IPOMDP model exploration
    exploreProblem,
    exploreAgent,
    exploreFrame,

    # POMDP frame to POMDP conversion
    gPOMDP,
    generatePOMDP,
    # IPOMDP frame to IPOMDP conversion
    gIPOMDP,
    generateIPOMDP,

    # Interactive belief
    InteractiveBelief,
    InteractiveBeliefUpdater,
    initialize_belief,
    update

    include("ipomdptoolbox.jl")
    include("generalPOMDP.jl")
    include("generalIPOMDP")
    include("interactiveBelief.jl")
end
