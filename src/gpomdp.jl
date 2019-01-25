"""
    Used in order to implement the IPOMDP to POMDP conversion.
    No conversion takes place at creation time, but it takes place in the moment the functions
    - initialstate_distribution
    - transition
    - observation
    - reward
    are called.
    See Julia.POMDPs for return values and usage.
    
    3rd version: Create transition, observation and reward functions on demand
"""
struct gPOMDP3{S,A,W} <: POMDPs.POMDP{S,A,W}
    belief::DiscreteInteractiveBelief{S,A,W}
    states
    observations
    transition
    observation
    reward
    
end

function gPOMDP3(b::DiscreteInteractiveBelief{S,A,W}) where {S,A,W}
        # Generate the objects
        # Transition
        problem_states = IPOMDPs.states(b.ipomdp)
        problem_actions = IPOMDPs.actions_agent(IPOMDPs.agent(b.ipomdp))
        problem_observations = IPOMDPs.observations_agent(IPOMDPs.agent(b.ipomdp))
        
        
        trans = Dict()
        rew = Dict()
        obs = Dict()
        
        return gPOMDP3(b, problem_states, problem_observations, trans, obs, rew)
    end

"""
    OC_pomdp = OC_ipomdp
"""
function POMDPs.discount(g::gPOMDP3)
    frame = g.belief.ipomdp
    return IPOMDPs.discount(frame)
end

"""
    S_pomdp = S_ipomdp
"""
function POMDPs.states(g::gPOMDP3)
    frame = g.belief.ipomdp
    return IPOMDPs.states(frame)
end

function POMDPs.n_states(g::gPOMDP3)
    l = size(POMDPs.states(g), 1)
    return l
end

function POMDPs.stateindex(g::gPOMDP3{S,A,W}, s::S) where {S,A,W}
    return IPOMDPs.stateindex(g.belief.ipomdp, s)
end

"""
    Marginalize the belief on the states S
"""
function POMDPs.initialstate_distribution(g::gPOMDP3)

    # Belief marginalization
    b = g.belief.dist
    states = POMDPs.states(g)
    probs = zeros(Float64, POMDPs.n_states(g))
    for (i,s) in enumerate(states)
        for (iS,p) in b
            if (iS.state == s)
                probs[i] = probs[i] + p
            end
        end
    end

    # The distribution should already be normalized, since it comes from b. Normalize it again anyway
    tot = 0
    for x in probs
        tot += x
    end
    for (i,v) in enumerate(probs)
        probs[i] = v/tot
    end
    b = SparseCat(states, probs)

    return b
end

function POMDPs.isterminal(g::gPOMDP3{S,A,W}, s::S) where {S,A,W}
    return IPOMDPs.isterminal(g.belief.ipomdp, s)
end

"""
    A_pomdp = Ai_ipomdp
"""
function POMDPs.actions(g::gPOMDP3)
    frame = g.belief.ipomdp
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.actions_agent(agent)
end

function POMDPs.n_actions(g::gPOMDP3{S,A,W}) where {S,A,W}
    l = size(POMDPs.actions(g), 1)
    return l
end

function POMDPs.actionindex(g::gPOMDP3{S,A,W}, action::A) where {S,A,W}
    frame = g.belief.ipomdp
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.actionindex_agent(agent, action)
end

"""
    W_pomdp = Wi_ipomdp
"""
function POMDPs.observations(g::gPOMDP3)
    frame = g.belief.ipomdp
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.observations_agent(agent)
end

function POMDPs.n_observations(g::gPOMDP3{S,A,W}) where {S,A,W}
    l = size(POMDPs.observations(g), 1)
    return l
end

function POMDPs.obsindex(g::gPOMDP3{S,A,W}, observation::W) where {S,A,W}
    frame = g.belief.ipomdp
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.obsindex_agent(agent, observation)
end

"""
    The transition function of the POMDP depends on the possible actions each model can take.
    Only actions of I are preserved
"""
function POMDPs.transition(g::gPOMDP3{S,A,W}, from::S, action::A) where {S,A,W}
    #O(1) search
    tra = get(g.transition, (from, action), nothing)
    if tra == nothing
        tra = make_transition(g.states, g.belief, from, action)
        g.transition[(from,action)] = tra
    end
    return tra
end

function make_transition(prob_states::Vector{S}, belief::DiscreteInteractiveBelief{S,A,W}, from::S, action::A) where {S,A,W}
    Ax = xAction(belief.ipomdp, action)
    states = Vector{S}()
    probs = Vector{Float64}()
    for s in prob_states
        push!(states, s)
        push!(probs,0.0)
    end

    # Perform the marginalization on all the actions of the other agents
    for (i,s) in enumerate(states)
        result = 0.0
        normal = 0.0
        for (iS, p) in belief.dist
            # Consider only the meaningful states is^{t-1} where s^{t-1} is the same as the function parameter
            if (iS.state == from)
                normal = normal + p
                for a in Ax
                    statesPDF = IPOMDPs.transition(belief.ipomdp, from, a)
                    aP = 1.0
                    # Weight depending on the probability of each action for each model
                    for m in iS.models
                        aP = aP * updatecache!(belief, m, a[IPOMDPs.agent(m.frame)])
                    end
                    result = result + POMDPModelTools.pdf(statesPDF, s) * aP * p
                end
            end
        end
        # Normalize
        probs[i] = result/normal
    end
    return SparseCat(states, probs)
end

"""
    The observation function of the POMDP depends on the possible actions each model can take.
"""
function POMDPs.observation(g::gPOMDP3{S,A,W}, action::A, to::S) where {S,A,W}
    #O(1) search
    obs = get(g.observation, (action,to), nothing)
    if obs == nothing
        obs = make_observation(g.observations, g.belief, action, to)
        g.observation[(action,to)] = obs
    end
    return obs
end

function make_observation(problem_observations::Vector{W}, belief::DiscreteInteractiveBelief{S,A,W}, action::A, to::S) where {S,A,W}
    Ax = xAction(belief.ipomdp, action)
    obs = Vector{W}()
    probs = Vector{Float64}()
    for s in problem_observations
        push!(obs, s)
        push!(probs,0.0)
    end

    # Perform the marginalization on all the actions of the other agents
    for a in Ax
        observationPDF = IPOMDPs.observation(belief.ipomdp, to, a)
        result = 0
        for (iS,p) in belief.dist
            aP = 1
            # Weight depending on the probability of each action for each model
            for m in iS.models
                aP = aP * updatecache!(belief, m, a[IPOMDPs.agent(m.frame)])
            end
            result = result + (aP * p)
        end
        for (i,o) in enumerate(obs)
            probs[i] = probs[i] + result * POMDPModelTools.pdf(observationPDF, o)
        end
    end
    return SparseCat(obs, probs)
end

"""
    The reward function of the POMDP depends on the possible actions each model can take and the physical state instead of the interactive.
    Only actions of I are preserved
"""
function POMDPs.reward(g::gPOMDP3{S,A,W}, from::S, action::A) where {S,A,W}
    #O(1) search
    rew = get(g.reward, (from, action), nothing)
    if rew == nothing
        rew = make_reward(g.belief, from, action)
        g.reward[(from,action)] = rew
    end
    return rew
end

function make_reward(belief::DiscreteInteractiveBelief{S,A,W}, from::S, action::A) where {S,A,W}
    Ax = xAction(belief.ipomdp, action)
    result = 0.0
    normal = 0.0

    for (iS, p) in belief.dist
        # Consider only the meaningful states is^{t-1} where s^{t-1} is the same as the function parameter
        if (iS.state == from)
            normal = normal + p
            # Perform the sum on all the actions of the other agents
            for a in Ax
                aP = 1.0
                # Weight depending on the probability of each action for each model
                for m in iS.models
                    aP = aP * updatecache!(belief, m, a[IPOMDPs.agent(m.frame)])
                end
                result = result + IPOMDPs.reward(belief.ipomdp, iS, a) * aP * p
            end
        end
    end

    # Normalize
    return result/normal
end
