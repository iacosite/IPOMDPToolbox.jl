#Conversion of a IPOMDP{S,A,W,X,Y,Z}, Agent{S,A,W} and pomdpFrame{S,A,W} in a pomdp
struct gIPOMDP{S, A, W, X <: IPOMDP{S}, Y <: Agent{S,A,W}, X <: ipomdpFrame{S,A,W}} <: IPOMDP{S}
    ipomdp::X # The problem of the frame
    agent::Y # The agent of the frame
    frame::X # The frame
end

function IPOMDPs.discount(g::gIPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.discount(g.ipomdp, g.frame)
end

function IPOMDPs.discount(g::gIPOMDP{S,A,W,X,Y,Z}, frame::Frame{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.discount(g.ipomdp, frame)
end

function IPOMDPs.states(g::gIPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.states(g.ipomdp)
end

function IPOMDPs.n_states(g::gIPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return IPOMDPs.n_states(g.ipomdp)
end

function IPOMDPs.stateindex(g::gIPOMDP{S,A,W,X,Y,Z}, s::S) where {S,A,W,X,Y,Z}
    return IPOMDPs.stateindex(g.ipomdp, s)
end

function IPOMDPs.agents(g::gIPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return g.agent
end

function IPOMDPs.agents(g::gIPOMDP{S,A,W,X,Y,Z}, frame::ipomdpFrame{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.agents(g.ipomdp, frame)
end

function IPOMDPs.n_agents(g::gIPOMDP{S,A,W,X,Y,Z}) where {S,A,W,X,Y,Z}
    return size(agents(g), 1)
end

function IPOMDPs.n_agents(g::gIPOMDP{S,A,W,X,Y,Z}, frame::ipomdpFrame{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.n_agents(g.ipomdp, frame)
end

function IPOMDPs.agentindex(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.agentindex(g.ipomdp, agent)
end

function IPOMDPs.agentindex(g::gIPOMDP{S,A,W,X,Y,Z}, frame::ipomdpFrame{S,A,W}, agent::Agent{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.agentindex(g.ipomdp, frame, agent)
end

function IPOMDPs.actions(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.actions(g.ipomdp, agent)
end

function IPOMDPs.n_actions(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.n_actions(g.ipomdp, agent)
end

function IPOMDPs.actionindex(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W}, action::A) where {S,A,W,X,Y,Z}
    return IPOMDPs.actionindex(g.ipomdp, agent, action)
end

function IPOMDPs.observations(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.observations(g.ipomdp, agent)
end

function IPOMDPs.n_observations(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.n_observations(g.ipomdp, agent)
end

function IPOMDPs.observationindex(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W} observation::W) where {S,A,W,X,Y,Z}
    return IPOMDPs.observationindex(g.ipomdp, agent, observation)
end

function IPOMDPs.frames(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Y) where {S,A,W,X,Y,Z}
    return g.frame
end

function IPOMDPs.frames(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.frames(g.ipomdp, agent)
end

function IPOMDPs.n_frames(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.n_frames(g.ipomdp, agent)
end

function IPOMDPs.frameindex(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Agent{S,A,W}, frame::Frame{S,A,W}) where {S,A,W,X,Y,Z}
    return IPOMDPs.frameindex(g.ipomdp, agent, frame)
end

function IPOMDPs.transition(g::gIPOMDP{S,A,W,X,Y,Z}, frame::ipomdpFrame{S,A,W} from::S, action::Vector{A}) where {S,A,W,X,Y,Z}
    return IPOMDPs.transition(g.ipomdp, frame, from, action)
end

function IPOMDPs.transition(g::gIPOMDP{S,A,W,X,Y,Z}, frame::pomdpFrame{S,A,W} from::S, action::A) where {S,A,W,X,Y,Z}
    return IPOMDPs.transition(g.ipomdp, frame, from, action)
end

function IPOMDPs.observation(g::gIPOMDP{S,A,W,X,Y,Z}, frame::ipomdpFrame{S,A,W}, action::Vector{A}, to::S) where {S,A,W,X,Y,Z}
    return IPOMDPs.observation(g.ipomdp, frame, action, to)
end

function IPOMDPs.observation(g::gIPOMDP{S,A,W,X,Y,Z}, frame::pomdpFrame{S,A,W}, action::A, to::S) where {S,A,W,X,Y,Z}
    return IPOMDPs.observation(g.ipomdp, frame, action, to)
end

function IPOMDPs.reward(g::gIPOMDP{S,A,W,X,Y,Z}, frame::ipomdpFrame{S,A,W}, from::S, action::Vector{A}) where {S,A,W,X,Y,Z}
    return IPOMDPs.reward(g.ipomdp, frame, s, a)
end

function IPOMDPs.reward(g::gIPOMDP{S,A,W,X,Y,Z}, frame::pomdpFrame{S,A,W} from::S, action::A) where {S,A,W,X,Y,Z}
    return IPOMDPs.reward(g.ipomdp, frame, s, a)
end

function IPOMDPs.initialframe_distribution(g::gIPOMDP{S,A,W,X,Y,Z}, agent::Y) where {S,A,W,X,Y,Z}
    frames = [g.frame]
    probs = [1.0]
    return SparseCat(frames, probs)
end

function IPOMDPs.isterminal(g::gIPOMDP{S,A,W,X,Y,Z}, frame::Frame{S,A,W}, s::S) where {S,A,W,X,Y,Z}
    return IPOMDPs.isterminal(g.ipomdp, frame, s)
end

"""
    Generate a gIPOMDP{S,A,W,X,Y,Z} object from
        IPOMDP{S,}
        Agent{S,A,W}
        pomdpFrame{S,A,W}
"""
function generateIPOMDP(ipomdp::IPOMDP{S}, agent::Agent{S,A,W}, frame::ipomdpFrame{S,A,W}) where {S,A,W}
    return gIPOMDP(ipomdp, agent, frame)
end
