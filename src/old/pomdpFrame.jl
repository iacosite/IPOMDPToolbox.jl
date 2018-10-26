abstract type pomdpFrame{S,A,W} <: POMDP{S,A,W} end

# States
function POMDPs.states(frame::pomdpFrame)
    return IPOMDPs.states(frame.ipomdp)
end

function POMDPs.n_states(frame::pomdpFrame)
    return size(POMDPs.states(frame), 1)
end

function POMDPs.stateindex(frame::pomdpFrame{S,A,W}, s::S) where {S,A,W}
    return IPOMDPs.stateindex(frame.ipomdp, s)
end

# Actions
function POMDPs.actions(frame::pomdpFrame)
    return IPOMDPs.actions(frame.problem, frame.agent)
end

function POMDPs.n_actions(frame::pomdpFrame)
    return size(POMDPs.actions(frame), 1)
end

function POMDPs.actionindex(frame::pomdpFrame{S,A,W}, action::A) where {S,A,W}
    return IPOMDPs.actionindex(frame.problem, frame.agent, action)
end

#Observations
function POMDPs.observations(frame::pomdpFrame)
    return IPOMDPs.observations(frame.ipomdp, frame.agent)
end

function POMDPs.n_observations(frame::pomdpFrame)
    return size(POMDPs.observations(frame), 1)
end

function POMDPs.obsindex(frame::pomdpFrame{S,A,W}, o::W) where {S,A,W}
    return IPOMDPs.observationindex(frame.ipomdp, frame.agent, o)
end

# History
function initialize_history(frame::pomdpFrame)
    updater = BeliefUpdaters.DiscreteUpdater(frame)
    return BeliefUpdaters.initialize_belief(updater, POMDPs.initialstate_distribution(frame))
end


# Update

# Action

# Action distribution



# All the other functions are defined by the user







#
# pomdpModel
#

function action(model::pomdpModel, data::pomdpFrameData{S,A,W}) where {S,A,W}
    return SARSOP.action(data.policy, model.belief)
end

function update(model::pomdpModel, action::A, observation::W) where {S,A,W}
    return SARSOP.update(data.updater, model.belief, action, observation)
end








struct pomdpModel{S,A,W,H} <: Model{S,A,W,H}
    index::Int64
    belief::DiscreteBelief
end

struct pomdpFrameData{S,A,W} <: FrameData
    id::Int64
    frame::POMDP{S,A,W}
    policy::POMDPPolicy
    updater::DiscreteUpdater
end

function frameData(ipomdp::IPOMDP{S}, agent::Agent{S,A,W}, frame::pomdpFrame{S,A,W}) where {S,A,W}
    pomdpFrame = IPOMDPToolbox.generatePOMDP(ipomdp, agent, frame)
    id = IPOMDPs.frameindex(ipomdp, agent, frame)
    policy = SARSOP.POMDPPolicy(pomdpFrame, "$id.policy")
    solver = SARSOP.SARSOPSolver()
    e_policy = POMDPs.solve(solver, pomdpFrame, policy, pomdp_file_name="$id.pomdpx")
    updater = SARSOP.updater(e_policy)

    return pomdpFrameData(id, pomdpFrame, e_policy, updater)
end

function frameModel(data::pomdpFrameData{S,A,W}) where {S,A,W}
    return pomdpModel(data.id, IPOMDPs.initial_belief(data))
end

function initialize_belief(data::pomdpFrameData{S,A,W}) where {S,A,W}
    dist = POMDPs.initialstate_distribution(data.pomdp)
    belief = SARSOP.initialize_belief(data.updater, dist)

    return belief
end





# Utility functions
function getData(frame::pomdpFrame)
    ipomdp = frame.ipomdp
    agent =  frame.agent
    id = IPOMDPs.frameindex(ipomdp, agent, frame)
    policy = SARSOP.POMDPPolicy(frame, "$id.policy")
    solver = SARSOP.SARSOPSolver()
    e_policy = POMDPs.solve(solver, frame, policy, pomdp_file_name="$id.pomdpx")
    updater = SARSOP.updater(e_policy)
    
    return pomdpFrameData(id, pomdpFrame, e_policy, updater)
end


