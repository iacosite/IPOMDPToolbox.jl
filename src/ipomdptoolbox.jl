#
#   POMDP MODEL
#

struct pomdpModel{S,A,W} <: IPOMDPs.Model{A,W}
    history

    # Immutable part of the structure! This is commo to all the models of the same frame!
    frame::POMDP{S,A,W}

    # Data
    updater
    policy

end

function IPOMDPs.Model(pomdp::POMDP)
#TODO: Get frame id from IPOMDPs
    policy = SARSOP.POMDPPolicy(pomdp, "test.policy")
    solver = SARSOP.SARSOPSolver()
    e_policy = POMDPs.solve(solver, pomdp, policy, pomdp_file_name="test.pomdpx")
    updater = SARSOP.updater(e_policy)
    belief = SARSOP.initialize_belief(updater, POMDPs.initialstate_distribution(pomdp))


    return pomdpModel(belief, pomdp, updater, e_policy)
end

function IPOMDPs.action(model::pomdpModel)
    return SARSOP.action(model.policy, model.history)
end

function IPOMDPs.tau(model::pomdpModel{A,W}, a::A, o::W) where {A,W}
    belief = update(model.updater, model.history, a, o)
    return pomdpModel(belief, model.frame, model.updater, model.policy)
end


#
#   SUBINTENTIONAL MODEL
#







