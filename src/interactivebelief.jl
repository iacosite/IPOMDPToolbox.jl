struct DiscreteInteractiveBelief{S,A,W}
    ipomdp::IPOMDP{S,A,W}
    dist::SparseCat{Vector{IS}, Vector{Float64}}
end

struct DiscreteInteractiveUpdater{S,A,W}
    ipomdp::IPOMDP{S,A,W}
end

function initialize_belief(up::DiscreteInteractiveUpdater)
    statedist = IPOMDPs.initialstate_distribution(up.ipomdp)
    framesdist = IPOMDPs.initialframe_distribution(up.ipomdp)
    return initialize_belief(up, statedist, framesdist)
end

function initialize_belief(up::DiscreteInteractiveUpdater{S,A,W}, statedist::SparseCat{Vector{S},Vector{Float64}}, framesdist::SparseCat) where {S,A,W}

    aFrames = Dict{Agent, Vector{Model}}()
    # Divide all the frames depending on their agent. Create the models
    for (f, p) in framesdist
        if !haskey(aFrames, agent(f))
            # Insert
            aFrames[agent(f)] = [Model(f)]
        else
            # append
            push!(aFrames[agent(f)], Model(f))
        end
    end
    # Perform cartesian product
    partialBelief = SparseCat([Vector{Model}()], [1.0])

   # Primo elemento: Symbol, altri Model -> Creare 2 array diversi per mantenere il tipo!
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
    # we have now the complete distribution among all the interactive states.
    # We need now to convert the array of state, models in an interactive stae itself

    iStates = Vector{IS}()
    iProbs = Vector{Float64}()
    for (sv,sp) in statedist
        for (v,p) in partialBelief
            push!(iStates, IS(sv,v))
            push!(iProbs, sp * p)
        end
    end
    belief = SparseCat(iStates, iProbs)

    return DiscreteInteractiveBelief(up.ipomdp, belief)
end

function update(up::DiscreteInteractiveUpdater{S,A,W}, b::DiscreteInteractiveBelief{S,A,W}, a::A, o::W) where {S,A,W}

    Ax = xAction(up.ipomdp, a)
    Ox = xObservation(up.ipomdp)

    newIS = Vector{IS}()
    newP = Vector{Float64}()
    for (is, Pis) in b.dist
        for s in IPOMDPs.states(up.ipomdp)
            for ax in Ax
                for ox in Ox
                    newModels = Vector{Model}()
                    P = 1
                    for m in is.models
                        ta = ax[agent(m.frame)]
                        to = ox[agent(m.frame)]
                        tm = tau(m, ta, to)
                        push!(newModels, tm)
                        pa = actionP(m, ta)
                        po = POMDPModelTools.pdf(IPOMDPs.model_observation(up.ipomdp, m.frame, s, ax), to)
                        P = P * pa * po
                        #  println("P($ta|J)=$pa")
                        #  println("Oj($s,$(values(ax)),$to)=$po")
                        #  println("Tau ($ta, $to)")
                    end
                    ti = POMDPModelTools.pdf(IPOMDPs.transition(up.ipomdp, is.state, ax), s)
                    oi = POMDPModelTools.pdf(IPOMDPs.observation(up.ipomdp, is.state, ax), o)
                    P = P * ti * oi * Pis
                    tis = IS(s,newModels)
                    if P > 0.0
                        push!(newIS, tis)
                        push!(newP, P)
                        #  println("Ti($(is.state),$(values(ax)),$s)= $ti")
                        #  println("Oi($s,$(values(ax)),$(o))=$oi")
                        #  println("P(is-1)=$Pis")
                        #  println("P(is) = $P")
                        #  dump(tis)
                    end
                    #  print("\n\n")
                end
            end
        end
    end

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

    # Normalize P
    for (i,v) in enumerate(seenP)
        seenP[i] = v/tot
    end

    belief = DiscreteInteractiveBelief(b.ipomdp, SparseCat(seenIS, seenP))
    return belief
end
