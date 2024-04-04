using PowerModelsDistribution, Ipopt, PowerModelsAnalytics, PowerModels, Distributions, BSON, JSON, HDF5

# Load the test case.
main_eng = PowerModels.parse_file("case5_risk.m")
eng = deepcopy(main_eng)
global keys_of_lines = keys(eng["branch"])
global keys_of_loads = keys(eng["load"])
plot_network(main_eng, label_nodes=true)

# Define the set of binary variables
binary_set = [0, 1]

# Generate all possible combinations
combinations = collect(Iterators.product(binary_set, binary_set, binary_set, binary_set, binary_set, binary_set))

# Define the dictionary to which all the values will be assigned.
all_data = Dict("load_shed" => [])
for i in keys_of_loads
    push!(all_data, "p$i" => [])
    push!(all_data, "q$i" => [])
end
for i in keys_of_lines
    push!(all_data, "z$i" => [])
end

# Define the number of samples for each configiration. 
num_cases = 2000                                                                            

# Begin variation in configurations and loads.
for c in combinations
    for case_idx in 1:num_cases
        eng = deepcopy(main_eng)
        # Manually varying the state of the switchable lines. Note: Closed line: 1, Open line: 0
        index = 1
        for i in keys_of_lines
            state = c[index]
            eng["branch"][i]["br_status"] = state
            push!(all_data["z$i"], state)
            index = index + 1
        end

        # Varying the load demand
        for i in keys(eng["load"])
            number_of_loads = count(>=(0), eng["load"][i]["pd"])
            nom_p = eng["load"][i]["pd"]
            nom_q = eng["load"][i]["qd"]
            uniform_distribution_range = Uniform(0.95, 1.05)
            scale = rand(uniform_distribution_range)
            if number_of_loads == 3
                eng["load"][i]["pd"] = nom_p * scale
                eng["load"][i]["qd"] = nom_q * scale
            else 
                eng["load"][i]["pd"] = nom_p * scale
                eng["load"][i]["qd"] = nom_q * scale
            end
            push!(all_data["p$i"], eng["load"][i]["pd"][1])
            push!(all_data["q$i"], eng["load"][i]["qd"][1])
        end

        loads = []
        keys_of_loads = keys(eng["load"])
        for i in keys_of_loads
            push!(loads, eng["load"][i]["pd"])
        end
        sum_load = sum(loads)

        # Begin solving the power flow
        pm = instantiate_model(eng, ACPPowerModel, PowerModels.build_opf)
        result = optimize_model!(pm, optimizer=Ipopt.Optimizer)

        gen = []
        keys_of_gen = keys(eng["gen"])
        for i in keys_of_gen
        push!(gen, result["solution"]["gen"][i]["pg"])
        end
        sum_gen = sum(gen)
        
        # Determing the load shed and saving to the dictionary
        if sum_gen > sum_load 
            S_loss = []
            for i in keys_of_lines
                if haskey(result["solution"]["branch"], "$i")
                    append!(S_loss, result["solution"]["branch"][i]["pt"] + result["solution"]["branch"][i]["pf"])
                else
                    append!(S_loss, 0)
                end
            end
            S_loss = sum(abs.(S_loss))
            load_shed = sum_gen - S_loss - sum_load
        else
            load_shed = abs(sum_load - sum_gen)
        end
        push!(all_data["load_shed"], load_shed/sum_load)
    end
end

# Save the dictionary to a BSON file.
bson("wildfire_training_data_$num_cases.bson", all_data)
