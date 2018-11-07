struct gPOMDP{S,A,W} <: POMDPs.POMDP{S,A,W}
    belief::DiscreteInteractiveBelief{S,A,W}
end

function POMDPs.discount(g::gPOMDP)
    frame = g.belief.ipomdp
    return IPOMDPs.discount(frame)
end

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

function POMDPs.initialstate_distribution(g::gPOMDP)
#    # Do not depend on belief
#    b = IPOMDPs.initialstate_distribution(g.belief.ipomdp)

    # Depends on belief
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

    # the distribution should already be normalized, since it comes from b. Normalize it again anyway
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

function POMDPs.transition(g::gPOMDP{S,A,W}, from::S, action::A) where {S,A,W}
    Ax = xAction(g.belief.ipomdp, action)
    states = Vector{S}()
    probs = Vector{Float64}()
    for s in POMDPs.states(g)
        push!(states, s)
        push!(probs,0.0)
    end

    Ax = xAction(g.belief.ipomdp, action)
    for (i,s) in enumerate(states)
        result = 0.0
        normal = 0.0
        for (iS, p) in g.belief.dist
            if (iS.state == from)
                normal = normal + p
                for a in Ax
                    statesPDF = IPOMDPs.transition(g.belief.ipomdp, from, a)
                    aP = 1.0
                    for m in iS.models
                        tmp = actionP(m, a[IPOMDPs.agent(m.frame)])
                     #   println("P($(IPOMDPs.agent(m.frame))->$(a[IPOMDPs.agent(m.frame)])): $tmp")
                        aP = aP * tmp
                    end
                    result = result + POMDPModelTools.pdf(statesPDF, s) * aP * p
                end
            end
        end

        probs[i] = result/normal
    end
    return SparseCat(states, probs)

end

function POMDPs.observation(g::gPOMDP{S,A,W}, action::A, to::S) where {S,A,W}
    Ax = xAction(g.belief.ipomdp, action)
    obs = Vector{W}()
    probs = Vector{Float64}()
    for s in POMDPs.observations(g)
        push!(obs, s)
        push!(probs,0.0)
    end

    for a in Ax
        observationPDF = IPOMDPs.observation(g.belief.ipomdp, to, a)
        result = 0
        for (iS,p) in g.belief.dist
            aP = 1
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

function POMDPs.reward(g::gPOMDP{S,A,W}, from::S, action::A) where {S,A,W}
    # Implements:
    # Sum_(Ax)( Sum_(Mx)( Prod_(m E mx)( P(am|m) )*P(mx) ) )/P(s)



    Ax = xAction(g.belief.ipomdp, action)
    result = 0.0
    normal = 0.0
    for (iS, p) in g.belief.dist
        if (iS.state == from)
            normal = normal + p
            for a in Ax
                aP = 1.0
                for m in iS.models
                    tmp = actionP(m, a[IPOMDPs.agent(m.frame)])
                 #   println("P($(IPOMDPs.agent(m.frame))->$(a[IPOMDPs.agent(m.frame)])): $tmp")
                    aP = aP * tmp
                end
                result = result + IPOMDPs.reward(g.belief.ipomdp, iS, a) * aP * p
            end
        end
    end

    return result/normal
end

