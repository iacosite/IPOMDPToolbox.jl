struct DiscreteInteractiveBelief{S,A,W}
    ipomdp::IPOMDP{S,A,W}
    dist::SparseCat{Vector{IS}, Vector{Float64}}
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
function initialize_belief(up::DiscreteInteractiveUpdater)
    statedist = IPOMDPs.initialstate_distribution(up.ipomdp)
    framesdist = IPOMDPs.initialframe_distribution(up.ipomdp)
    # Call more general function
    return initialize_belief(up, statedist, framesdist)
end


"""
    Initialize the belief given the state and frame distributions
    initialize_belief(up::DiscreteInteractiveUpdater{S,A,W}, statedist::SparseCat{Vector{S},Vector{Float64}}, framesdist::SparseCat)
Return:
    DiscreteInteractiveBelief{S,A,W}
"""
function initialize_belief(up::DiscreteInteractiveUpdater{S,A,W}, statedist::SparseCat{Vector{S},Vector{Float64}}, framesdist::SparseCat) where {S,A,W}

    # Divide all the frames depending on their agent and create the models
    aFrames = Dict{Agent, Vector{Model}}()
    for (f, p) in framesdist
        if !haskey(aFrames, agent(f))
            # Insert
            aFrames[agent(f)] = [Model(f)]
        else
            # append
            push!(aFrames[agent(f)], Model(f))
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
    return DiscreteInteractiveBelief(up.ipomdp, belief)
end


"""
    Performs IPOMDP belief update
    update(up::DiscreteInteractiveUpdater{S,A,W}, b::DiscreteInteractiveBelief{S,A,W}, a::A, o::W) where {S,A,W}
Return:
    DiscreteInteractiveBelief{S,A,W}
"""
function update(up::DiscreteInteractiveUpdater{S,A,W}, b::DiscreteInteractiveBelief{S,A,W}, a::A, o::W) where {S,A,W}

    # Determine cross product between all other agent actions and ai, all other agent observations
    Ax = xAction(up.ipomdp, a)
    Ox = xObservation(up.ipomdp)

    # Branch on the interactive state on possible:
    # - states
    # - agents actions
    # - agents observations
    newIS = Vector{IS}()
    newP = Vector{Float64}()
    for (is, Pis) in b.dist
        for s in IPOMDPs.states(up.ipomdp)
            for ax in Ax
                for ox in Ox
                    newModels = Vector{Model}()
                    P = 1
                    # for the branch, calculate the probability for each to make a certain action and receive a certain observation
                    for m in is.models
                        ta = ax[agent(m.frame)]
                        to = ox[agent(m.frame)]
                        tm = tau(m, ta, to)
                        push!(newModels, tm)
                        pa = actionP(m, ta)
                        po = POMDPModelTools.pdf(IPOMDPs.model_observation(up.ipomdp, m.frame, s, ax), to)
                        P = P * pa * po
                    end
                    # Weight on I's transition and observation probabilities for s and oi
                    ti = POMDPModelTools.pdf(IPOMDPs.transition(up.ipomdp, is.state, ax), s)
                    oi = POMDPModelTools.pdf(IPOMDPs.observation(up.ipomdp, is.state, ax), o)
                    P = P * ti * oi * Pis
                    tis = IS(s,newModels)
                    if P > 0.0
                        push!(newIS, tis)
                        push!(newP, P)
                    end
                end
            end
        end
    end
    # We now have all the combinations calculated before weighted for the relative probabilities. There are numerous duplicates

    # Remove duplicates
    seenIS = Vector{IS}()
    seenP = Vector{Float64}()
    tot = 0
    for (i,v) in enumerate(newIS)
        p = newP[i]
        tot = tot + p
        # Check if v is in seenIS
        present = false
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

    # Normalize
    for (i,v) in enumerate(seenP)
        seenP[i] = v/tot
    end

    # Create the new belief and return
    belief = DiscreteInteractiveBelief(b.ipomdp, SparseCat(seenIS, seenP))
    return belief
end
