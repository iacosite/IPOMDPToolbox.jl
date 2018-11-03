#
# TYPES
#



struct pomdpModel{S,A,W} <: IPOMDPs.Model{A,W}
    history::DiscreteBelief

    # Immutable part of the structure! This is commo to all the models of the same frame!
    frame::POMDP{S,A,W}

    # Data
    updater::DiscreteUpdater
    policy::POMDPPolicy

end

struct ipomdpModel{S,A,W} <: IPOMDPs.Model{A,W}
	history
	frame::IPOMDP{S,A,W}
end

struct gPOMDP{S,A,W} <: POMDP{S, A, W}
    model::ipomdpModel{S,A,W}
end




#
#   POMDP MODEL
#

function IPOMDPs.Model(pomdp::POMDP)
#TODO: Get frame id from IPOMDPs
    policy = SARSOP.POMDPPolicy(pomdp, "test.policy")
    solver = SARSOP.SARSOPSolver()
    e_policy = POMDPs.solve(solver, pomdp, policy, pomdp_file_name="test.pomdpx")
    updater = SARSOP.updater(e_policy)
    belief = SARSOP.initialize_belief(updater, POMDPs.initialstate_distribution(pomdp))


    return pomdpModel(belief, pomdp, updater, e_policy)
end

function IPOMDPs.action(model::pomdpModel)
    return SARSOP.action(model.policy, model.history)
end

function IPOMDPs.tau(model::pomdpModel{A,W}, a::A, o::W) where {A,W}
    belief = update(model.updater, model.history, a, o)
    return pomdpModel(belief, model.frame, model.updater, model.policy)
end







#
#   IPOMDP MODEL
#

function IPOMDPs.Model(frame::IPOMDP)
    # Create the belief

    sDist = IPOMDPs.initialstate_distribution(frame)

    fDist = IPOMDPs.initialframe_distribution(frame)
    aFrames = Dict{Agent, Vector{Model}}()
    # Divide all the frames depending on their agent. Create the models
    for (f, p) in fDist
        if !haskey(aFrames, agent(f))
            # Insert
            aFrames[agent(f)] = [Model(f)]
        else
            # append
            push!(aFrames[agent(f)], Model(f))
        end
    end
    # Perform cartesian product
    frameProbs = IPOMDPs.initialframe_distribution(frame)
    partialBelief = SparseCat([Vector{Model}()], [1.0])

   # Primo elemento: Symbol, altri Model -> Creare 2 array diversi per mantenere il tipo!
    for (a, fModels) in aFrames
        newP = Vector{Float64}()
        newM = Vector{Vector{Model}}()
        for (v,p) in partialBelief
            for m in fModels
                M = Vector{Model}()
                append!(M, deepcopy(v))
                P = p * POMDPModelTools.pdf(frameProbs, m.frame)
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
    for (sv,sp) in sDist
        for (v,p) in partialBelief
            push!(iStates, IS(sv,v))
            push!(iProbs, sp * p)
        end
    end
    belief = SparseCat(iStates, iProbs)



    return ipomdpModel(belief, frame)
end

struct mData
    id
    e_policy
    belief
end

modelsData = Dict{gPOMDP, mData}()

function IPOMDPs.action(model::ipomdpModel)

    debug = true
    # Calculate the converted pomdp
    pomdp = gPOMDP(model)
    if debug
        printPOMDP(pomdp)
    end
    #Two ipomdpModel are equal if their frame is the same and their history is identical
    found = false
    e_policy = nothing
    belief = nothing
    if debug
        for (k,v) in modelsData
            println("K:")
            dump(k)
            println("Data:")
            dump(v)
        end
        println("To check with:")
        dump(pomdp)
    end
    for (g,v) in modelsData
        if isa(g.model.frame, typeof(pomdp.model.frame))
        if myApprox(g.model.history, pomdp.model.history)
            e_policy = v.e_policy
            belief = v.belief
            found = true
            if debug
                println("Already calculated!")
            end
            break
        end
        end
    end
    if !found
        #policy = SARSOP.POMDPPolicy(pomdp, "$(IPOMDPToolbox.counter).policy")
        policy = SARSOP.POMDPPolicy(pomdp, "test.policy")
        solver = SARSOP.SARSOPSolver()
        #e_policy = POMDPs.solve(solver, pomdp, policy, pomdp_file_name="$(IPOMDPToolbox.counter).pomdpx")
        e_policy = POMDPs.solve(solver, pomdp, policy, pomdp_file_name="test.pomdpx")

        # We obtain the belief state from the initial belief od the pomdp problem
        updater = SARSOP.updater(e_policy)
        belief = SARSOP.initialize_belief(updater, POMDPs.initialstate_distribution(pomdp))

        #modelsData[pomdp] = mData(IPOMDPToolbox.counter, e_policy, belief)
        modelsData[pomdp] = mData(1, e_policy, belief)
        #IPOMDPToolbox.counter = IPOMDPToolbox.counter + 1
        #println("Counter: $IPOMDPToolbox.counter")
    end
    if debug
        println("Policy:")
        dump(e_policy)
        println("Belief:")
        dump(belief)
    end
    return SARSOP.action(e_policy, belief)
end

function IPOMDPs.tau(model::ipomdpModel{S,A,W}, a::A, o::W) where {S,A,W}
    # A = ai x Aj x Ak x ...
    # O = Oj x Ok x ...
    # Get all the other agents in the problem (J,K,...)


    Ax = xAction(model, a)
    Ox = xObservation(model)

    newIS = Vector{IS}()
    newP = Vector{Float64}()
    for (is, Pis) in model.history
        for s in IPOMDPs.states(model.frame)
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
                        po = POMDPModelTools.pdf(IPOMDPs.model_observation(model.frame, m.frame, s, ax), to)
                        P = P * pa * po
                        #  println("P($ta|J)=$pa")
                        #  println("Oj($s,$(values(ax)),$to)=$po")
                        #  println("Tau ($ta, $to)")
                    end
                    ti = POMDPModelTools.pdf(IPOMDPs.transition(model.frame, is.state, ax), s)
                    oi = POMDPModelTools.pdf(IPOMDPs.observation(model.frame, is.state, ax), o)
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

    belief = SparseCat(seenIS, seenP)
    return ipomdpModel(belief, model.frame)
end









#
#   IPOMDP to POMDP conversion
#

#Conversion of a IPOMDP{S}, Agent{S,A,W} and pomdpFrame{S,A,W} in a POMDP{S,A,W}


function POMDPs.discount(g::gPOMDP)
    frame = g.model.frame
    return IPOMDPs.discount(frame)
end

function POMDPs.states(g::gPOMDP)
    frame = g.model.frame
    return IPOMDPs.states(frame)
end

function POMDPs.n_states(g::gPOMDP)
    l = size(POMDPs.states(g), 1)
    return l
end

function POMDPs.stateindex(g::gPOMDP{S,A,W}, s::S) where {S,A,W}
    return IPOMDPs.stateindex(g.model.frame, s)
end

function POMDPs.initialstate_distribution(g::gPOMDP)
#    # Do not depend on belief
#    b = IPOMDPs.initialstate_distribution(g.model.frame)

    # Depends on belief
    b = g.model.history
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
    return IPOMDPs.isterminal(g.model.frame, s)
end

function POMDPs.actions(g::gPOMDP)
    frame = g.model.frame
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.actions_agent(agent)
end

function POMDPs.n_actions(g::gPOMDP{S,A,W}) where {S,A,W}
    l = size(POMDPs.actions(g), 1)
    return l
end

function POMDPs.actionindex(g::gPOMDP{S,A,W}, action::A) where {S,A,W}
    frame = g.model.frame
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.actionindex_agent(agent, action)
end

function POMDPs.observations(g::gPOMDP)
    frame = g.model.frame
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.observations_agent(agent)
end

function POMDPs.n_observations(g::gPOMDP{S,A,W}) where {S,A,W}
    l = size(POMDPs.observations(g), 1)
    return l
end

function POMDPs.obsindex(g::gPOMDP{S,A,W}, observation::W) where {S,A,W}
    frame = g.model.frame
    agent = IPOMDPs.agent(frame)
    return IPOMDPs.obsindex_agent(agent, observation)
end

function POMDPs.transition(g::gPOMDP{S,A,W}, from::S, action::A) where {S,A,W}
    Ax = xAction(g.model, action)
    states = Vector{S}()
    probs = Vector{Float64}()
    for s in POMDPs.states(g)
        push!(states, s)
        push!(probs,0.0)
    end


    for a in Ax
        statesPDF = IPOMDPs.transition(g.model.frame, from, a)
        result = 0
        for (iS,p) in g.model.history
            aP = 1
            for m in iS.models
                aP = aP * actionP(m, a[IPOMDPs.agent(m.frame)])
            end
            result = result + (aP * p)
        end
        for (i,s) in enumerate(states)
            probs[i] = probs[i] + result * POMDPModelTools.pdf(statesPDF, s)
        end
    end
    return SparseCat(states, probs)
end

function POMDPs.observation(g::gPOMDP{S,A,W}, action::A, to::S) where {S,A,W}
    Ax = xAction(g.model, action)
    obs = Vector{W}()
    probs = Vector{Float64}()
    for s in POMDPs.observations(g)
        push!(obs, s)
        push!(probs,0.0)
    end


    for a in Ax
        observationPDF = IPOMDPs.observation(g.model.frame, to, a)
        result = 0
        for (iS,p) in g.model.history
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



    Ax = xAction(g.model, action)
    result = 0.0
    normal = 0.0
    for (iS, p) in g.model.history
        if (iS.state == from)
            normal = normal + p
            for a in Ax
                aP = 1.0
                for m in iS.models
                    tmp = actionP(m, a[IPOMDPs.agent(m.frame)])
                 #   println("P($(IPOMDPs.agent(m.frame))->$(a[IPOMDPs.agent(m.frame)])): $tmp")
                    aP = aP * tmp
                end
                result = result + IPOMDPs.reward(g.model.frame, iS, a) * aP * p
            end
        end
    end

    return result/normal
end





#
#   UTILITY FUNCTIONS
#


#
# Approximation function
#
appr(x::AbstractArray{X}) where {X <: Number} = true
appr(x::Number) = true
appr(x) = false

function myApprox(a,b; maxdepth=20, debug=false)
    # Returns wether two objects are equal:
    # Comparison is implemented: isequal for each field where isapprox is not applicable
    if (debug)
        println("Check depth: $maxdepth")
        dump(a)
        dump(b)
    end
    if (maxdepth > 0)

        # Check if the object is iterable: In case it is iterable MUST implement the function iterate
        if applicable(iterate, a) && applicable(iterate, b)
            result = true
            if (debug)
                println("Iterable!")
            end
            try
                c = zip(a,b)
                for x in c
                    result = result && myApprox(x[1], x[2]; maxdepth=maxdepth-1, debug=debug)
                end
            catch y
                if (debug)
                    println("Error!")
                    dump(y)
                end
                result = false
            end
            return result

        elseif(nfields(a) > 0 && nfields(b) > 0)
            # Not iterable but a struct or type, parse all the fields
            result = true
            nf = nfields(a)
            if (debug)
                println("Struct!")
            end
            for f in 1:nf
                try
                    result = result && myApprox(getfield(a,f), getfield(b,f); maxdepth=maxdepth-1, debug=debug)
                catch
                    result = false
                end
            end
            return result
        end
    end
    # We either cannot recurse anymore or the object is not iterable or we are at the bottom of the structure
    if (debug)
        println("Bottom of structure!")
    end
    if(appr(a) && appr(b))
        return isapprox(a,b)
    else
        return isequal(a,b)
    end
end



function xAction(model::ipomdpModel{S,A,W}, a::A) where {S,A,W}
    Agents = Vector{Agent}()
    for f in IPOMDPs.emulated_frames(model.frame)
        if !(IPOMDPs.agent(f) in Agents)
            push!(Agents, IPOMDPs.agent(f))
        end
    end

    Ax = Vector{Dict{Agent, Any}}()
    d = Dict{Agent, Any}()
    d[IPOMDPs.agent(model.frame)] = a
    push!(Ax, d)
    for ag in Agents
        newA = Vector{Dict}()
        for crossAction in Ax
            for agentAction in actions_agent(ag)
                tmp = deepcopy(crossAction)
                tmp[ag] = agentAction
                push!(newA, tmp)
            end
        end
        Ax = newA
    end

    return Ax
end

function xObservation(model::ipomdpModel)
    Agents = Vector{Agent}()
    for f in IPOMDPs.emulated_frames(model.frame)
        if !(IPOMDPs.agent(f) in Agents)
            push!(Agents, agent(f))
        end
    end

    Ox = Vector{Dict{Agent, Any}}()
    push!(Ox, Dict{Agent, Any}())
    for ag in Agents
        newO = Vector{Dict{Agent, Any}}()

        for crossObservation in Ox
            for agentObservation in IPOMDPs.observations_agent(ag)
                tmp = deepcopy(crossObservation)
                tmp[ag] = agentObservation
                push!(newO, tmp)
            end
        end
        Ox = newO
    end

    return Ox
end

function printPOMDP(pomdp::POMDP)
    println("States:")
    for s in POMDPs.states(pomdp)
        i = POMDPs.stateindex(pomdp, s)
        println("[$i] $s ")
    end
    println("")

    println("Actions:")
    for a in POMDPs.actions(pomdp)
        i = POMDPs.actionindex(pomdp, a)
        println("[$i] $a ")
    end
    println("")

    println("Observations")
    for o in POMDPs.observations(pomdp)
        i = POMDPs.obsindex(pomdp, o)
        println("[$i] $o ")
    end
    println("")

    println("Transition function:")
    print("T\t")
    for sp in POMDPs.states(pomdp)
        print("$sp\t")
    end
    println("")
    for s in POMDPs.states(pomdp)
        for a in POMDPs.actions(pomdp)
            print("[$s,$a]")
            dist = POMDPs.transition(pomdp, s, a)
            for sp in POMDPs.states(pomdp)
                p = POMDPModelTools.pdf(dist, sp)
                print("\t$p")
            end
            println("")
        end
    end
    println("")

    println("Observation function:")
    print("O")
    for o in POMDPs.observations(pomdp)
        print("\t$o")
    end
    println("")
    for sp in POMDPs.states(pomdp)
        for a in POMDPs.actions(pomdp)
            print("[$sp,$a]")
            dist = POMDPs.observation(pomdp, a, sp)
            for o in POMDPs.observations(pomdp)
                p = POMDPModelTools.pdf(dist, o)
                print("\t$p")
            end
            println("")
        end
    end
    println("")

    println("Reward function:")
    print("R")
    for sp in POMDPs.states(pomdp)
        print("\t$sp")
    end
    println("")
    for a in POMDPs.actions(pomdp)
        print("[$a]")
        for s in POMDPs.states(pomdp)
            r = POMDPs.reward(pomdp, s ,a)
            print("\t$r")
        end
        println("")
    end
    println("")
end
