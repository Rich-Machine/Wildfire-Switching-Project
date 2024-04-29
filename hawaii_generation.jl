using Ipopt, PowerModelsAnalytics, PowerModels, Distributions, BSON

# network_type = "base_case"
# network_type = "sole_gen"
# network_type = "high_risk"
# network_type = "case14"
network_type = "hawaii"

# Load the test case.
main_eng = PowerModels.parse_file("case5_risk_$network_type.m")

global keys_of_lines = keys(main_eng["branch"])
global keys_of_loads = keys(main_eng["load"])
plot_network(main_eng, label_nodes=true)

# Define the set of binary variables
binary_set = [0, 1]

# # Generate all possible combinations
# combinations = collect(Iterators.product(binary_set, binary_set, binary_set, binary_set, binary_set, binary_set))
# if network_type == "case14"
#     combinations = collect(Iterators.product(binary_set, binary_set, binary_set, binary_set, binary_set, binary_set, binary_set, binary_set, binary_set, binary_set,binary_set, binary_set, binary_set, binary_set, binary_set,binary_set, binary_set, binary_set, binary_set, binary_set))
# end

# Define the dictionary to which all the values will be assigned.
all_data = Dict("load_shed" => [])
for i in keys_of_loads
    push!(all_data, "p$i" => [])
    push!(all_data, "q$i" => [])
end
for i in keys_of_lines
    push!(all_data, "y-$i" => [])
end

# Define the number of samples for each configiration. 
num_cases = 1000                                 

# Begin variation in configurations and loads.
for case_idx in 1:num_cases
    eng = deepcopy(main_eng)

    for i in keys(eng["branch"])
        r = eng["branch"][i]["br_r"]
        x = eng["branch"][i]["br_x"]

        a = 1.25 * rand()

        Y = 1/(r + x * im)
        Y = Y * a

        X = 1/(real(Y) + imag(Y) * im)

        eng["branch"][i]["br_r"] = real(X)
        eng["branch"][i]["br_x"] = imag(X)

        push!(all_data["y-$i"], a)
        # end
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
    global result = optimize_model!(pm, optimizer=Ipopt.Optimizer)

    gen = []
    keys_of_gen = keys(result["solution"]["gen"])
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
    push!(all_data["load_shed"], load_shed)
end

# Save the dictionary to a BSON file.
all_data = sort(all_data)
bson("wildfire_training_data_$network_type.bson", all_data)
