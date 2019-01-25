
struct ReductionSolver
    # Here should go some settings
    timeout::Float64
end

struct ReductionPolicy{S,A,W}
    ipomdp::IPOMDP{S,A,W}
    timeout::Float64
end

"""
    Returns the appropriate updater to work with ReductionSolver
    updater(p::ReductionPolicy)
Return:
    DiscreteInteractiveUpdater
"""
function IPOMDPs.updater(p::ReductionPolicy)
    return DiscreteInteractiveUpdater(p.ipomdp)
end

"""
    Return the policy type used by the solver. Since ReductionSolver is an online solver, the policy doesn't really exist.
    It is used as a container to maintain data through time
    solve(solver::ReductionSolver, ipomdp::IPOMDP{S,A,W})
Return:
    ReductionPolicy{S,A,W}
"""
function IPOMDPs.solve(solver::ReductionSolver, ipomdp::IPOMDP{S,A,W}) where {S,A,W}
    # Create the folder used by the action function
    try
    mkdir("./tmp")
    catch
    # Already present
    end
    return ReductionPolicy(ipomdp, solver.timeout)
end

"""
    Convertes the IPOMDP problem in a POMDP, solves the POMDP and returns the best action.
    action(policy::ReductionPolicy{S,A,W}, b::DiscreteInteractiveBelief{S,A,W})
Return:
    action::A
"""
function IPOMDPs.action(policy::ReductionPolicy{S,A,W}, b::DiscreteInteractiveBelief{S,A,W}) where {S,A,W}
    pomdp = gPOMDP3(b)
    name = "$(hash(pomdp))$(rand())"
    pol = SARSOP.POMDPPolicy(pomdp, "./tmp/_$name.policy")
    solver = SARSOP.SARSOPSolver(;timeout=policy.timeout)
    e_policy = SARSOP.solve(solver, pomdp, pol, silent=true, pomdp_file_name="./tmp/_$name.pomdpx")
    updater = SARSOP.updater(e_policy)
    belief = SARSOP.initialize_belief(updater, POMDPs.initialstate_distribution(pomdp))
    a = SARSOP.action(e_policy, belief)
    rm("./tmp/_$name.pomdpx", force=true)
    rm("./tmp/_$name.policy", force=true)
    return a
end

