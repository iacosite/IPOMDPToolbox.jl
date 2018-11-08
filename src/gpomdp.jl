"""
    Used in order to implement the IPOMDP to POMDP conversion.
    No conversion takes place at creation time, but it takes place in the moment the functions
    - initialstate_distribution
    - transition
    - observation
    - reward
    are called.
    See Julia.POMDPs for return values and usage.
"""
struct gPOMDP{S,A,W} <: POMDPs.POMDP{S,A,W}
    belief::DiscreteInteractiveBelief{S,A,W}
end

"""
    OC_pomdp = OC_ipomdp
"""
function POMDPs.discount(g::gPOMDP)
    frame = g.belief.ipomdp
    return IPOMDPs.discount(frame)
end

"""
    S_pomdp = S_ipomdp
"""
function POMDPs.states(g::gPOMDP)
    frame = g.belief.ipomdp
    return IPOMDPs.states(frame)
end

function POMDPs.n_states(g::gPOMDP)
    l = size(POMDPs.states(g), 1)
    return l
end

function POMDPs.stateindex(g::gPOMDP{S,A,W}, s::S) where {S,A,W}
    return IPOMDPs.stateindex(g.belief.ipomdp, s)
end

"""
    Marginalize the belief on the states S
"""
function POMDPs.initialstate_distribution(g::gPOMDP)

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

function POMDPs.isterminal(g::gPOMDP{S,A,W}, s::S) where {S,A,W}
    return IPOMDPs.isterminal(g.belief.ipomdp, s)
end

"""
    A_pomdp = Ai_ipomdp
"""
function POMDPs.actions(g::gPOMDP)
    frame = g.belief.ipomdp
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.actions_agent(agent)
end

function POMDPs.n_actions(g::gPOMDP{S,A,W}) where {S,A,W}
    l = size(POMDPs.actions(g), 1)
    return l
end

function POMDPs.actionindex(g::gPOMDP{S,A,W}, action::A) where {S,A,W}
    frame = g.belief.ipomdp
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.actionindex_agent(agent, action)
end

"""
    W_pomdp = Wi_ipomdp
"""
function POMDPs.observations(g::gPOMDP)
    frame = g.belief.ipomdp
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.observations_agent(agent)
end

function POMDPs.n_observations(g::gPOMDP{S,A,W}) where {S,A,W}
    l = size(POMDPs.observations(g), 1)
    return l
end

function POMDPs.obsindex(g::gPOMDP{S,A,W}, observation::W) where {S,A,W}
    frame = g.belief.ipomdp
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.obsindex_agent(agent, observation)
end

"""
    The transition function of the POMDP depends on the possible actions each model can take.
    Only actions of I are preserved
"""
function POMDPs.transition(g::gPOMDP{S,A,W}, from::S, action::A) where {S,A,W}

    Ax = xAction(g.belief.ipomdp, action)
    states = Vector{S}()
    probs = Vector{Float64}()
    for s in POMDPs.states(g)
        push!(states, s)
        push!(probs,0.0)
    end

    # Perform the marginalization on all the actions of the other agents
    for (i,s) in enumerate(states)
        result = 0.0
        normal = 0.0
        for (iS, p) in g.belief.dist
            # Consider only the meaningful states is^{t-1} where s^{t-1} is the same as the function parameter
            if (iS.state == from)
                normal = normal + p
                for a in Ax
                    statesPDF = IPOMDPs.transition(g.belief.ipomdp, from, a)
                    aP = 1.0
                    # Weight depending on the probability of each action for each model
                    for m in iS.models
                        tmp = actionP(m, a[IPOMDPs.agent(m.frame)])
                        aP = aP * tmp
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
function POMDPs.observation(g::gPOMDP{S,A,W}, action::A, to::S) where {S,A,W}
    Ax = xAction(g.belief.ipomdp, action)
    obs = Vector{W}()
    probs = Vector{Float64}()
    for s in POMDPs.observations(g)
        push!(obs, s)
        push!(probs,0.0)
    end

    # Perform the marginalization on all the actions of the other agents
    for a in Ax
        observationPDF = IPOMDPs.observation(g.belief.ipomdp, to, a)
        result = 0
        for (iS,p) in g.belief.dist
            aP = 1
            # Weight depending on the probability of each action for each model
            for m in iS.models
                aP = aP * actionP(m, a[IPOMDPs.agent(m.frame)])
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
function POMDPs.reward(g::gPOMDP{S,A,W}, from::S, action::A) where {S,A,W}

    Ax = xAction(g.belief.ipomdp, action)
    result = 0.0
    normal = 0.0

    for (iS, p) in g.belief.dist
        # Consider only the meaningful states is^{t-1} where s^{t-1} is the same as the function parameter
        if (iS.state == from)
            normal = normal + p
            # Perform the sum on all the actions of the other agents
            for a in Ax
                aP = 1.0
                # Weight depending on the probability of each action for each model
                for m in iS.models
                    tmp = actionP(m, a[IPOMDPs.agent(m.frame)])
                    aP = aP * tmp
                end
                result = result + IPOMDPs.reward(g.belief.ipomdp, iS, a) * aP * p
            end
        end
    end

    # Normalize
    return result/normal
end
