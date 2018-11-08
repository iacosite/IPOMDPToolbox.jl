"""
    Custom approximation function:
    This is needed brecause, sometimes, belief differ for very small amounts making them basically identical.
    This is not intended to be used as a filter function, but as an approximation for equality check
"""
# Functions used in order to define whether isapprox() can be used or not
appr(x::AbstractArray{X}) where {X <: Number} = true
appr(x::Number) = true
appr(x) = false
function myApprox(a,b; maxdepth=20, debug=false)
    # Comparison is implemented: isequal for each field where isapprox is not applicable
    if (debug)
        println("Check depth: $maxdepth")
        dump(a)
        dump(b)
    end

    # Bound the depth
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
    # We either cannot recurse anymore or the object is a primitive object or we are at the bottom of the structure
    if (debug)
        println("Bottom of structure!")
    end
    if(appr(a) && appr(b))
        return isapprox(a,b)
    else
        return isequal(a,b)
    end
end

"""
    Return the cross product of one action with all the possible combinations of actions for the other agents
Return:
    Dict{Agent, Any}
"""
function xAction(frame::IPOMDP{S,A,W}, a::A) where {S,A,W}
    # Determine all the agents present in the IPOMDP
    Agents = Vector{Agent}()
    for f in IPOMDPs.emulated_frames(frame)
        if !(IPOMDPs.agent(f) in Agents)
            push!(Agents, IPOMDPs.agent(f))
        end
    end


    Ax = Vector{Dict{Agent, Any}}()
    d = Dict{Agent, Any}()
    d[IPOMDPs.agent(frame)] = a
    push!(Ax, d)
    # Cross product for all the actions of all the agents
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

"""
    Return the cross product of all the possible combinations of observations for the other agents
Return:
    Dict{Agent, Any}
"""
function xObservation(frame::IPOMDP)
    # Determine all the other agents present in the IPOMDP
    Agents = Vector{Agent}()
    for f in IPOMDPs.emulated_frames(frame)
        if !(IPOMDPs.agent(f) in Agents)
            push!(Agents, agent(f))
        end
    end

    # Cross product for all the observations of all the agents
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

"""
    Print all the POMDP elements in readable form
"""
function printPOMDP(pomdp::POMDP)
    println("Belief:")
    b = POMDPs.initialstate_distribution(pomdp)
    for (s,p) in b
        println("P($s) = $p")
    end
    println("")


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
