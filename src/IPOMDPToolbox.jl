module IPOMDPToolbox

using POMDPs
using IPOMDPs
using SARSOP
using POMDPModelTools
using BeliefUpdaters

export
    pomdpModel,
    Model,
    action,
    tau,
    actionP,
    printPOMDP,

    DiscreteInteractiveBelief,
    DiscreteInteractiveUpdater,
    
    ReductionSolver,
    ReductionPolicy

    include("functions.jl")
    include("interactivebelief.jl")
    include("gpomdp.jl")
    include("reductionsolver.jl")
    include("ipomdptoolbox.jl")
end
