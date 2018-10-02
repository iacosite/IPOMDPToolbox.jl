#Conversion of a IPOMDP{S}, Agent{S,A,W} and pomdpFrame{S,A,W} in a pomdp
struct cPOMDP <: POMDP{Any, Any, Any}
    ipomdp::IPOMDP
    agent::Agent
    frame::pomdpFrame
end

function POMDPs.states(c::cPOMDP)
    return IPOMDPs.states(c.ipomdp)
end

function POMDPs.n_states(c::cPOMDP)
    return IPOMDPs.n_states(c.ipomdp)
end

function POMDPs.stateindex(c::cPOMDP, s::Any)
    return IPOMDPs.stateindex(c.ipomdp, s)
end

function POMDPs.initialstate_distribution(c::cPOMDP)
    return IPOMDPs.initialstate_distribution(c.ipomdp, c.frame)
end

function POMDPs.isterminal(c::cPOMDP, s::Any)
    return IPOMDPs.isterminal(c.ipomdp, s)
end

function POMDPs.actions(c::cPOMDP)
    return IPOMDPs.actions(c.ipomdp, c.agent)
end

function POMDPs.n_actions(c::cPOMDP)
    return IPOMDPs.n_actions(c.ipomdp, c.agent)
end

function POMDPs.actionindex(c::cPOMDP, action::Any)
    return IPOMDPs.actionindex(c.ipomdp, c.agent, action)
end

function POMDPs.observations(c::cPOMDP)
    return IPOMDPs.observations(c.ipomdp, c.agent)
end

function POMDPs.n_observations(c::cPOMDP)
    return IPOMDPs.n_observations(c.ipomdp, c.agent)
end

function POMDPs.obsindex(c::cPOMDP, observation::Any)
    return IPOMDPs.observationindex(c.ipomdp, c.agent, observation)
end

function POMDPs.transition(c::cPOMDP, from::Any, action::Any)
    return IPOMDPs.transition(c.ipomdp, c.frame, from, action)
end

function POMDPs.observation(c::cPOMDP, action::Any, to::Any)
    return IPOMDPs.observation(c.ipomdp, c.frame, c.action, to)
end

function POMDPs.reward(c::cPOMDP, from::Any, action::Any)
    return IPOMDPs.reward(c.ipomdp, c.frame, s, a)
end

"""
    Generate a GeneralPOMDP object from
        IPOMDP{S}
        Agent{S,A,W}
        pomdpFrame{S,A,W}
"""
function generatePOMDP(ipomdp::IPOMDP{S}, agent::Agent{S,A,W}, frame::pomdpFrame{S,A,W}) where {S,A,W}
    return cPOMDP(ipomdp, agent, frame)
end
