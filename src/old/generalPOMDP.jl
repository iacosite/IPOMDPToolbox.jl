#Conversion of a IPOMDP{S}, Agent{S,A,W} and pomdpFrame{S,A,W} in a POMDP{S,A,W}
struct gPOMDP{S,A,W, X <: IPOMDP{S}, Y <: Agent{S,A,W}, Z <: pomdpFrame{S,A,W}} <: POMDP{S, A, W}
    ipomdp::X
    agent::Y
    frame::Z
end

function POMDPs.discount(g::gPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.discount(g.ipomdp, g.frame)
end

function POMDPs.states(g::gPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.states(g.ipomdp)
end

function POMDPs.n_states(g::gPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.n_states(g.ipomdp)
end

function POMDPs.stateindex(g::gPOMDP{S,A,W,X,Y,Z}, s::S) where {S,A,W,X,Y,Z}
    return IPOMDPs.stateindex(g.ipomdp, s)
end

function POMDPs.initialstate_distribution(g::gPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.initialstate_distribution(g.ipomdp, g.frame)
end

function POMDPs.isterminal(g::gPOMDP{S,A,W,X,Y,Z}, s::S) where {S,A,W,X,Y,Z}
    return IPOMDPs.isterminal(g.ipomdp, s)
end

function POMDPs.actions(g::gPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.actions(g.ipomdp, g.agent)
end

function POMDPs.n_actions(g::gPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.n_actions(g.ipomdp, g.agent)
end

function POMDPs.actionindex(g::gPOMDP{S,A,W,X,Y,Z}, action::A) where {S,A,W,X,Y,Z}
    return IPOMDPs.actionindex(g.ipomdp, g.agent, action)
end

function POMDPs.observations(g::gPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.observations(g.ipomdp, g.agent)
end

function POMDPs.n_observations(g::gPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.n_observations(g.ipomdp, g.agent)
end

function POMDPs.obsindex(g::gPOMDP{S,A,W,X,Y,Z}, observation::A) where {S,A,W,X,Y,Z}
    return IPOMDPs.observationindex(g.ipomdp, g.agent, observation)
end

function POMDPs.transition(g::gPOMDP{S,A,W,X,Y,Z}, from::S, action::A) where {S,A,W,X,Y,Z}
    return IPOMDPs.transition(g.ipomdp, g.frame, from, action)
end

function POMDPs.observation(g::gPOMDP{S,A,W,X,Y,Z}, action::A, to::S) where {S,A,W,X,Y,Z}
    return IPOMDPs.observation(g.ipomdp, g.frame, g.action, to)
end

function POMDPs.reward(g::gPOMDP{S,A,W,X,Y,Z}, from::S, action::A) where {S,A,W,X,Y,Z}
    return IPOMDPs.reward(g.ipomdp, g.frame, s, a)
end

"""
    Generate a GeneralPOMDP object from
        IPOMDP{S}
        Agent{S,A,W}
        pomdpFrame{S,A,W}
"""
function generatePOMDP(ipomdp::IPOMDP{S}, agent::Agent{S,A,W}, frame::pomdpFrame{S,A,W}) where {S,A,W}
    return gPOMDP(ipomdp, agent, frame)
end
