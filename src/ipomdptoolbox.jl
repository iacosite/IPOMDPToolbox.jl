"""
    Explores a IPOMDP problem
"""

function exploreProblem(ipomdp::IPOMDP)
    i = IPOMDPs.agents(ipomdp)
    exploreAgent(ipomdp, i, "")
end

"""
    Explores an agent with its models
"""

function exploreAgent(ipomdp::IPOMDP, a::Agent, suffix::String)
    println("Exploring")
    println(suffix * "Exploring agent: " * string(a.name))
    println(suffix * "- Actions: " * string(IPOMDPs.actions(ipomdp, a)))
    println(suffix * "- Observations: " * string(IPOMDPs.observations(ipomdp, a)))
    mod = IPOMDPs.models(ipomdp, a)
    if isa(mod, Array)
        for m in mod
            exploreModel(ipomdp, m, suffix)
        end
    else
        exploreModel(ipomdp, mod, suffix)
    end
end

"""
    Explore the model and all the contained agents
"""
function exploreModel(ipomdp::IPOMDP, m::Model, suffix::String)
    if(isa(m, SubintentionalModel))
        println(suffix * "- Subintentional model")
    elseif(isa(m, IntentionalModel))
        println(suffix * "- Intentional model")
        if(isa(m, pomdpModel))
            println(suffix * "- POMDP")
        elseif(isa(m, ipomdpModel))
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
            println(suffix * "Intentional model not recognized")
        end
    else
        println(suffix * "Type not recognized: " * string(typeof(m)))
    end
end

"""
    Generate a GeneralPOMDP object from
        IPOMDP{S}
        Agent{S,A,W}
        pomdpModel{S,A,W}
"""
function generateGeneralPOMDP(ipomdp::IPOMDP, agent::Agent, model::pomdpModel)
    return null
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

function find(V::Vector{Model}, X::Model)
    for (i,e) in enumerate(X)
        if e == X
            return i
        end
    end
    return -1
end
