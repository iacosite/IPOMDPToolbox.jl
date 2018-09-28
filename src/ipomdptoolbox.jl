using IPOMDPs
"""
    Explores an agent with its models
"""

function exploreAgent(suffix::String, a::Agent)
    println("Exploring")
    println(suffix * "Exploring agent: " * string(a.name))
    println(suffix * "- Actions: " * string(actions(a)))
    println(suffix * "- Observations: " * string(observations(a)))
    mod = models(a)
    if isa(mod, Array)
        for m in mod
            exploreModel(suffix, m)
        end
    else
        exploreModel(suffix, mod)
    end
end

"""
    Explore the model and all the contained agents
"""
function exploreModel(suffix::String, m::Model)
    if(isa(m, SubintentionalModel))
        println(suffix * "- Subintentional model")
    elseif(isa(m, IntentionalModel))
        println(suffix * "- Intentional model")
        if(isa(m, pModel))
            println(suffix * "- POMDP")
        elseif(isa(m, iModel))
            println(suffix * "- I-POMDP, Exploring..")
            ags = agents(m)
            suffix = suffix * "    "
            if isa(ags, Array)
                for a in ags
                    exploreAgent(suffix, a)
                end
            else
                exploreAgent(suffix, ags)
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

