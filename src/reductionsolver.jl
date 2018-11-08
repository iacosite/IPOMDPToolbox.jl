# Data in order to avoid to solve multiple times the same model
struct modelData
    id
    e_policy
    belief
end

struct ReductionSolver
    # Here should go some settings
end

struct ReductionPolicy{S,A,W}
    ipomdp::IPOMDP{S,A,W}
    mData::Dict{gPOMDP{S,A,W}, modelData}
end

"""
    Returns the appropriate updater to work with ReductionSolver
    updater(p::ReductionPolicy)
Return:
    DiscreteInteractiveUpdater
"""
function updater(p::ReductionPolicy)
    return DiscreteInteractiveUpdater(p.ipomdp)
end

"""
    Return the policy type used by the solver. Since ReductionSolver is an online solver, the policy doesn't really exist.
    It is used as a container to maintain data through time
    solve(solver::ReductionSolver, ipomdp::IPOMDP{S,A,W})
Return:
    ReductionPolicy{S,A,W}
"""
function solve(solver::ReductionSolver, ipomdp::IPOMDP{S,A,W}) where {S,A,W}
    return ReductionPolicy(ipomdp, Dict{gPOMDP{S,A,W},modelData}())
end

"""
    Convertes the IPOMDP problem in a POMDP, solves the POMDP and returns the best action.
    action(policy::ReductionPolicy{S,A,W}, b::DiscreteInteractiveBelief{S,A,W})
Return:
    action::A
"""
function action(policy::ReductionPolicy{S,A,W}, b::DiscreteInteractiveBelief{S,A,W}) where {S,A,W}
    gpomdp = gPOMDP(b)
    (e_policy, belief) = update_data!(policy, gpomdp)
    return SARSOP.action(e_policy, belief)
end

# Utility function
# Used in order to manage the data stored in the policy.
# The soilver works with SARSOP, hence the data stored is the one necessary to call SARSOP.action
# Return:
#   e_policy::POMDPPolicy
#   belief::DiscreteBelief
function update_data!(policy::ReductionPolicy{S,A,W}, pomdp::gPOMDP{S,A,W}) where {S,A,W}
    id = length(policy.mData)
    e_policy = nothing
    belief = nothing

    # Check if data for this POMDP is already present
    found = false
    for (g,v) in policy.mData
        if isa(g.belief.ipomdp, typeof(pomdp.belief.ipomdp))
            if myApprox(g.belief.dist, pomdp.belief.dist)
                e_policy = v.e_policy
                belief = v.belief
                found = true
                break
            end
        end
    end

    # If not present, solve the POMDP and store the data
    if !found
        pol = SARSOP.POMDPPolicy(pomdp, "_$id.policy")
        solver = SARSOP.SARSOPSolver()
        e_policy = POMDPs.solve(solver, pomdp, pol, silent=true, pomdp_file_name="_$id.pomdpx")
        updater = SARSOP.updater(e_policy)
        belief = SARSOP.initialize_belief(updater, POMDPs.initialstate_distribution(pomdp))
        policy.mData[pomdp] = modelData(id, e_policy, belief)

        # Clear the temp files in the working directory
        rm("_$id.policy", force=true)
        rm("_$id.pomdpx", force=true)
    end
    return (e_policy, belief)
end
# note that update, initialize_belief are already defined in interactivebelief.jl.


