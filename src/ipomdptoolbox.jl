"""
    Explores a IPOMDP problem
"""

function exploreProblem(ipomdp::IPOMDP)
    i = IPOMDPs.agents(ipomdp)
    exploreAgent(ipomdp, i, "")
end

"""
    Explores an agent with its frames
"""

function exploreAgent(ipomdp::IPOMDP, a::Agent, suffix::String)
    println("Exploring")
    println(suffix * "Exploring agent: " * string(a.name))
    println(suffix * "- Actions: " * string(IPOMDPs.actions(ipomdp, a)))
    println(suffix * "- Observations: " * string(IPOMDPs.observations(ipomdp, a)))
    mod = IPOMDPs.frames(ipomdp, a)
    if isa(mod, Array)
        for m in mod
            exploreFrame(ipomdp, m, suffix)
        end
    else
        exploreFrame(ipomdp, mod, suffix)
    end
end

"""
    Explore the frame and all the contained agents
"""
function exploreFrame(ipomdp::IPOMDP, m::Model, suffix::String)
    if(isa(m, SubintentionalFrame))
        println(suffix * "- Subintentional frame")
    elseif(isa(m, IntentionalFrame))
        println(suffix * "- Intentional frame")
        if(isa(m, pomdpFrame))
            println(suffix * "- POMDP")
        elseif(isa(m, ipomdpFrame))
            println(suffix * "- I-POMDP, Exploring..")
            ags = IPOMDPs.agents(ipomdp, m)
            suffix = suffix * "    "
            if isa(ags, Array)
                for a in ags
                    exploreAgent(ipomdp, a, suffix)
                end
            else
                exploreAgent(ipomdp, ags, suffix)
            end
        else
            println(suffix * "Intentional frame not recognized")
        end
    else
        println(suffix * "Type not recognized: " * string(typeof(m)))
    end
end

# Find element X in the array V
function find(V::Vector{Agent}, X::Agent)
    for (i,e) in enumerate(X)
        if e == X
            return i
        end
    end
    return -1
end

function find(V::Vector{Frame}, X::Model)
    for (i,e) in enumerate(X)
        if e == X
            return i
        end
    end
    return -1
end


