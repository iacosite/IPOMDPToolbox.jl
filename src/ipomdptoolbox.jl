#
#   POMDP MODEL
#
struct pomdpModel{S,A,W} <: IPOMDPs.Model{A,W}
    history::DiscreteBelief

    # Immutable part of the structure! This is commo to all the models of the same frame!
    frame::POMDP{S,A,W}

    # Data
    updater::DiscreteUpdater
    policy::POMDPPolicy

end

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

function IPOMDPs.action(model::pomdpModel)
    return SARSOP.action(model.policy, model.history)
end

function IPOMDPs.tau(model::pomdpModel{A,W}, a::A, o::W) where {A,W}
    belief = BeliefUpdaters.update(model.updater, model.history, a, o)
    return pomdpModel(belief, model.frame, model.updater, model.policy)
end




#
#   IPOMDP MODEL
#
struct ipomdpModel{S,A,W} <: IPOMDPs.Model{A,W}
	history::DiscreteInteractiveBelief
	frame::IPOMDP{S,A,W}
    updater::DiscreteInteractiveUpdater
    policy::ReductionPolicy
end


function IPOMDPs.Model(frame::IPOMDP)
    solver = ReductionSolver()

    updater = DiscreteInteractiveUpdater(frame)
    policy = solve(solver, frame)
    belief = initialize_belief(updater)

    return ipomdpModel(belief, frame, updater, policy)
end

function IPOMDPs.action(model::ipomdpModel; debug=false, printpomdp=false)

    return action(model.policy, model.history)
end


function IPOMDPs.tau(model::ipomdpModel{S,A,W}, a::A, o::W) where {S,A,W}
    # A = ai x Aj x Ak x ...
    # O = Oj x Ok x ...
    # Get all the other agents in the problem (J,K,...)

    belief = update(model.updater, model.history, a, o)
    return ipomdpModel(belief, model.frame, model.updater, model.policy)
end






