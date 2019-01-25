struct DiscreteInteractiveBelief{S,A,W}
    ipomdp::IPOMDP{S,A,W}
    dist::SparseCat{Vector{IS}, Vector{Float64}}
    depth::Int64
    actionProbs::Dict{Model,Dict{Any,Float64}}
end

struct DiscreteInteractiveUpdater{S,A,W}
    ipomdp::IPOMDP{S,A,W}
end

"""
    Initialize the belief for a given IPOMDP
    initialize_belief(up::DiscreteInteractiveUpdater)
Return:
    DiscreteInteractiveBelief{S,A,W}
"""
function IPOMDPs.initialize_belief(up::DiscreteInteractiveUpdater; depth=0)
    statedist = IPOMDPs.initialstate_distribution(up.ipomdp)
    framesdist = IPOMDPs.initialframe_distribution(up.ipomdp)
    # Call more general function
    return IPOMDPs.initialize_belief(up, statedist, framesdist;depth=depth)
end


"""
    Initialize the belief given the state and frame distributions
    initialize_belief(up::DiscreteInteractiveUpdater{S,A,W}, statedist::SparseCat{Vector{S},Vector{Float64}}, framesdist::SparseCat)
Return:
    DiscreteInteractiveBelief{S,A,W}
"""
function IPOMDPs.initialize_belief(up::DiscreteInteractiveUpdater{S,A,W}, statedist::SparseCat{Vector{S},Vector{Float64}}, framesdist::SparseCat;depth=0) where {S,A,W}

    # Divide all the frames depending on their agent and create the models
    aFrames = Dict{Agent, Vector{Model}}()
    for (f, p) in framesdist
        m = Model(f, depth = depth+1)
        if !haskey(aFrames, agent(f))
            # Insert
            aFrames[agent(f)] = [m]
        else
            # append
            push!(aFrames[agent(f)], m)
        end
    end
    # We now have all the models divided by their agent

    # Perform cartesian product between the models of all the agents
    partialBelief = SparseCat([Vector{Model}()], [1.0])
    for (a, fModels) in aFrames
        newP = Vector{Float64}()
        newM = Vector{Vector{Model}}()
        for (v,p) in partialBelief
            for m in fModels
                M = Vector{Model}()
                append!(M, deepcopy(v))
                P = p * POMDPModelTools.pdf(framesdist, m.frame)
                push!(M, m)
                newP = push!(newP, P)
                newM = push!(newM, M)
            end
        end
        partialBelief = SparseCat(newM, newP)
    end
    # We now have all the models combinations. One model per agent

    # Perform cartesian product between physical states and the models combinations, obtaining the Interactive states
    iStates = Vector{IS}()
    iProbs = Vector{Float64}()
    for (sv,sp) in statedist
        for (v,p) in partialBelief
            push!(iStates, IS(sv,v))
            push!(iProbs, sp * p)
        end
    end
    belief = SparseCat(iStates, iProbs)

    # Create and return the belief
    return DiscreteInteractiveBelief(up.ipomdp, belief, depth, Dict{Model,Dict{Any,Float64}}())
end


"""
    Performs IPOMDP belief update
    update(up::DiscreteInteractiveUpdater{S,A,W}, b::DiscreteInteractiveBelief{S,A,W}, a::A, o::W) where {S,A,W}
Return:
    DiscreteInteractiveBelief{S,A,W}
"""
function IPOMDPs.update(up::DiscreteInteractiveUpdater{S,A,W}, b::DiscreteInteractiveBelief{S,A,W}, a::A, o::W) where {S,A,W}

    # Determine cross product between all other agent actions and ai, all other agent observations
    Ax = xAction(up.ipomdp, a)
    Ox = xObservation(up.ipomdp)

    # Branch on the interactive state on possible:
    # - states
    # - agents actions
    # - agents observations
    newIS = Vector{IS}()
    newP = Vector{Float64}()
    for (Is, Pis) in b.dist
        # States to be propagated are chosen depending on their probability
        if true#rand() < (Pis*50.0/b.depth) #FIXME sampling breaks SARSOP
            for ax in Ax
                oi = POMDPModelTools.pdf(IPOMDPs.observation(up.ipomdp, Is.state, ax), o)
                if oi > 0.0
                    for s in IPOMDPs.states(up.ipomdp)
                        ti = POMDPModelTools.pdf(IPOMDPs.transition(up.ipomdp, Is.state, ax), s)
                        if ti > 0.0
                            for ox in Ox
                                newModels = Vector{Model}()
                                P = 1
                                # for the branch, calculate the probability for each to make a certain action and receive a certain observation
                                for m in Is.models
                                    ta = ax[agent(m.frame)]
                                    to = ox[agent(m.frame)]
                                    pa = updatecache!(b, m, ta)
                                    po = POMDPModelTools.pdf(IPOMDPs.model_observation(up.ipomdp, m.frame, s, ax), to)
                                    P = P * pa * po
                                    if P > 0.0
                                        tm = tau(m, ta, to)
                                        push!(newModels, tm)
                                    else
                                        break
                                    end
                                end
                                # Weight on I's transition and observation probabilities for s and oi
                                P = P * ti * oi * Pis
                                if P > 0.0
                                    tis = IS(s,newModels)
                                    push!(newIS, tis)
                                    push!(newP, P)
                                end
                            end
                        end
                    end
                end
            end
        end
        
    end
    # We now have all the combinations calculated before weighted for the relative probabilities. There are numerous duplicates

    # Remove duplicates
    #O(n^2) :( + myApprox is recursive! #TODO: figure out how to speed up
    seenIS = Vector{IS}()
    seenP = Vector{Float64}()
    tot = 0.0
    for (i,v) in enumerate(newIS)
        p = newP[i]
        tot = tot + p
        # Check if v is in seenIS
        present = false
        # Perform pruning again
        if true #rand() < (p*50.0/b.depth) #FIXME sampling breaks SARSOP
            for (j,vv) in enumerate(seenIS)
                if (myApprox(v,vv))
                    # Present, index = j
                    seenP[j] = seenP[j] + p
                    present = true
                    break
                end
            end
            if (!present)
                # Not present
                push!(seenIS, v)
                push!(seenP, p)
            end
        end
    end
    
    # Normalize
    for (i,v) in enumerate(seenP)
        seenP[i] = v/tot
    end

    # Create the new belief and return
    belief = DiscreteInteractiveBelief(b.ipomdp, SparseCat(seenIS, seenP), b.depth, b.actionProbs)
    #println("Belief updated")
    return belief
end


# Inner function used in order to cache the probability of actions of each model
#TODO: Update in order to consider actionP
function updatecache!(b::DiscreteInteractiveBelief, m::Model, action)
    
    #O(1) search
    aDist = get(b.actionProbs, m, nothing)
    
    if aDist == nothing
        # Not found
        aDist = Dict{Any,Float64}()
        # Calculate the probability for each action
        a = IPOMDPs.action(m)
        for ac in IPOMDPs.actions_agent(IPOMDPs.agent(m.frame))
            if ac == a
                aDist[ac] = 1.0
            else
                aDist[ac] = 0.0
            end 
        end
        b.actionProbs[m] = aDist
        return aDist[action]
    else
        return aDist[action]
    end
end


