using IPOMDPs

using BeliefUpdaters, ARDESPOT, POMDPModelTools

"""
    The essential block containing all the data present in a frame.
    All objects extending the FrameData type must implement:
    function action{A}(belief::Belief)
    function update{Belief}(old::Belief, action, observation)
"""
struct interactiveBelief
    probs::Vector{Float64}
    states::Vector{IS}
    ModelFrameMap::Dict{Int64, FrameData}

end

struct interactiveBeliefUpdater{S}
    ipomdp::IPOMDP{S}
end

function initialize_belief(updater::interactiveBeliefUpdater{S}, probs::Vector{Float64}, states::Vector{IS}, map::Dict{Int64, FrameData}) where {S}
    return interactiveBelief(probs, states, map)
end

function initialize_belief(updater::interactiveBeliefUpdater{S}, ipomdp::IPOMDP{S}, agent::Agent{S,A,W}, frame::ipomdpFrame{S,A,W}) where {S,A,W}
    map = Dict{Int64, FrameData}()
    # Construct the probs, states and map vectors and return the belief

    stateDist = IPOMDPs.initialstate_distribution(ipomdp, ipomdpFrame)

    agentsFramesDists = Dict{Agent, SparseCat{Float64, Frame}}()
    for a in IPOMDPs.agents(ipomdp, ipomdpFrame)
        agentsFramesDists[a] = IPOMDPs.initialframe_distribution(ipomdp, a)
    end

    agentsModelsDists = Dict{Agent, SparseCat{Float64, Model}}()
    # Convert from Frames to Models in agentsFrameDists
    for a, dist in agentsFramesDists
        mProbs = Vector{Float64}()
        mMods = Vector{Model}()

        for p, f in dist
            # Retrieve needed data, if not present create it
            id = IPOMDPs.frameindex(ipomdp, f)
            if !haskey(map, id)
                map[id] = IPOMDPs.frameData(ipomdp, a, f)
            end
            data = map[id]

            # Create the model and save it
            push!(mProbs, p)
            push!(mMods, IPOMDPs.frameModel(data))
        end

        # Create the new distribution between models and their probability
        nDist = SparseCat(mMods, mProbs)
        agentsModelsDists[a] = nDist
    end

    # All the agents now have their own models and relative probabilities.
    # Perform cartesian products between SxMjxMkx... and their probabilities

    probVector = Vector{Float64}()
    stateVector = Vector{Vector{Any}}()

    for (s,p) in stateDist
        push!(probVector, p)
        push!(stateVector, [s])
    end

    for (a, modelDist) in agentsModelsDists
        newP = Vector{Float64}()
        newS = Vector{Vector{Any}}()

        for i in 1:length(stateVector)
            p = probVector[i]
            s = stateVector[i]

            for (m, pm) in modelDist
                tmp = deepcopy(s)
                push!(tmp, m)
                push!(newP, pm*p)
                push!(newS, tmp)
            end
        end
        probVector = newP
        stateVector = newS
    end

    # Now we can create all the interactive states
    iStates = Vector{IS}()
    for state in stateVector
        # state has shape [s, mI, mJ, mk...]
        push!(iStates, iState(state[1], state[2:end]))
    end


    return initialize_belief(updater, probVector, iStates, map)
end

function update(updater::interactiveBeliefUpdater{S}, b_old::interactiveBelief, action::A, observation::W) where {S,A,W}
    # Here goes the belief update function
end
