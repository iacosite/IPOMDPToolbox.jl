"""
    Wrapper for representing the model of a POMDP frame
    Uses SARSOP in order to solve the POMDP
"""
struct pomdpModel{S,A,W} <: IPOMDPs.Model{A,W}
    history::DiscreteBelief

    # Immutable part of the structure! This is commo to all the models of the same frame!
    frame::POMDP{S,A,W}

    # Data
    updater::DiscreteUpdater
    policy::POMDPPolicy

end

"""
    Wrapper for reresenting the model of a IPOMDP frame
    uses ReductionSolver in order to solve the IPOMDP
"""
struct ipomdpModel{S,A,W} <: IPOMDPs.Model{A,W}
	history::DiscreteInteractiveBelief
	frame::IPOMDP{S,A,W}
    updater::DiscreteInteractiveUpdater
    policy::ReductionPolicy
end

"""
    Constructs a Model given the relative problem.
    In case the model uses an offline solver (e.g. pomdpModel uses SARSOP), the problem is solved an all the necessary data in order to retreive the best action is stored
    Model(pomdp::POMDP)
    Model(ipomdp::IPOMDP)
"""
function IPOMDPs.Model(pomdp::POMDP)
    policy = SARSOP.POMDPPolicy(pomdp, "_temp_pomdp.policy")
    solver = SARSOP.SARSOPSolver()
    e_policy = POMDPs.solve(solver, pomdp, policy, silent=true, pomdp_file_name="_temp_pomdp.pomdpx")
    updater = SARSOP.updater(e_policy)
    belief = SARSOP.initialize_belief(updater, POMDPs.initialstate_distribution(pomdp))

    # Remove temporary files
    rm("_temp_pomdp.policy", force=true)
    rm("_temp_pomdp.pomdpx", force=true)

    return pomdpModel(belief, pomdp, updater, e_policy)
end
function IPOMDPs.Model(ipomdp::IPOMDP)
    solver = ReductionSolver()
    updater = DiscreteInteractiveUpdater(ipomdp)
    policy = solve(solver, ipomdp)
    belief = initialize_belief(updater)

    return ipomdpModel(belief, ipomdp, updater, policy)
end

"""
    Returns the best action for the model. If the model uses an online solver (e.g. ipomdpModel uses ReductionSolver), the problem is solved and the optimal action is returned
    action(model::pomdpModel)
    action(model::ipomdpModel)
"""
function IPOMDPs.action(model::pomdpModel)
    return SARSOP.action(model.policy, model.history)
end
function IPOMDPs.action(model::ipomdpModel)
    return action(model.policy, model.history)
end

"""
    Updates the belief and returns the updated model
    tau(model::pomdpModel{S,A,W}, a::A, o::W)
    tau(model::ipomdpModel{S,A,W}, a::A, o::W)
"""
function IPOMDPs.tau(model::pomdpModel{A,W}, a::A, o::W) where {A,W}
    belief = BeliefUpdaters.update(model.updater, model.history, a, o)
    return pomdpModel(belief, model.frame, model.updater, model.policy)
end
function IPOMDPs.tau(model::ipomdpModel{S,A,W}, a::A, o::W) where {S,A,W}
    # A = ai x Aj x Ak x ...
    # O = Oj x Ok x ...
    # Get all the other agents in the problem (J,K,...)

    belief = update(model.updater, model.history, a, o)
    return ipomdpModel(belief, model.frame, model.updater, model.policy)
end
