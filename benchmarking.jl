#This script is made to bench mark the wildfire nn optimization with the acopf and dcopf results of the ots problem
using PowerModels, JuMP, Ipopt, Gurobi, Cbc, Test, PowerPlots, JLD2, Plots

#import the results from the wildfire_nn_optimization bson file
network_type =  "high_risk" #"base_case" #"sole_gen" #"high_risk" 
nn_opt_results = load("nn_opt_results_$network_type.jld2")["nn_opt_results_$network_type"]

#AC
network_data = PowerModels.parse_file("case5_risk_$network_type.m")
pm = instantiate_model(network_data, ACPPowerModel, PowerModels.build_opf)
#import line statuses from the wildfire_nn_optimization results
#loop through the line statuses and set the line statuses in the network data
total_load_shed_nn_ac = []
total_load_shed_dc = []
total_load_shed_ac = []
line_statuses = []

loads = []
for i in keys(network_data["load"])
    push!(loads, network_data["load"][i]["pd"])
end
D_p = sum(loads)
# line statuses is a matrix of the  column wise concatonation of each line status vector in the nn_opt_results dictionary
for i in 1:6
    push!(line_statuses,nn_opt_results["line_$i"])
end
#line_statuses = [0, 1, 0, 1, 0, 1]

alpha = nn_opt_results["alpha"]
for j in 1:length(alpha)
    for i in 1:6
        #print(i)
        network_data["branch"]["$i"]["br_status"] = line_statuses[i][j]
    end
    pm = instantiate_model(network_data, ACPPowerModel, PowerModels.build_opf)
    result = optimize_model!(pm, optimizer=Ipopt.Optimizer)
    pm_dc = instantiate_model(network_data, DCPPowerModel, PowerModels.build_opf)
    result_dc = optimize_model!(pm_dc, optimizer=Ipopt.Optimizer)
    load_shed_units = sum(network_data["load"]["$i"]["pd"] for i in 1:length(network_data["load"])) - sum(result["solution"]["gen"]["$i"]["pg"] for i in 1:length(network_data["gen"]))
    push!(total_load_shed_ac, load_shed_units/D_p)
    
    load_shed_units_dc = sum(network_data["load"]["$i"]["pd"] for i in 1:length(network_data["load"])) - sum(result_dc["solution"]["gen"]["$i"]["pg"] for i in 1:length(network_data["gen"]))
    push!(total_load_shed_dc, load_shed_units_dc/D_p)

    push!(total_load_shed_nn_ac, nn_opt_results["load_shed_units"][j][1])

    # println(result["solution"]["gen"])
    # println(result_dc["solution"]["gen"])
    println("Load Shed NN_AC: $total_load_shed_nn_ac")
end


#plot wildfire risk vs load shed
risk = nn_opt_results["wildfire_risk"]
plot([total_load_shed_ac,total_load_shed_dc,total_load_shed_nn_ac],risk, label=["ACOPF" " DCOPF" "NN_ACOPF"], ylabel="Wildfire Risk", xlabel="% Load Shed", title="Wildfire Risk vs Load Shed for $network_type", legend=:topright)
savefig("benchmark_results/nn_acopf_ac_dc_opf_risk_vs_load_shed_$network_type.png")
