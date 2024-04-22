using PowerModels, PowerModelsWildfire
using Gurobi, Ipopt
using JLD2, Plots

# Load case data

network_type =  "sole_gen" #"high_risk"#"sole_gen" #"base_case"
# for network_type in ["base_case", "high_risk", "sole_gen"]

# network_type in ["base_case"]
    case = parse_file("case5_risk_$network_type.m")
    nn_opt_results = load("nn_opt_results_$network_type.jld2")["nn_opt_results_$network_type"]

    # Set risk parameter which determines trade-off between serving load and mitigating wildfire risk
    total_load_shed =[]
    wildfire_risk = []
    total_load_shed_nn_ac = []

    # Define alpha parameter
    # Smaller Values of Alpha ensure wildfire risk is zero and load shed is 0
    # alpha = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,  0.17, 0.18, 0.19, 0.2]
    # if network_type == "sole_gen"
    #     alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8, 0.9]
    # end

    alpha = (0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79, 0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99)

    for i in 1:length(alpha)
        case["risk_weight"] = alpha[i] # values between 0 and 1, smaller values emphasize load delivery

        # Run OPS problem
        solution = PowerModelsWildfire.run_ops(case, DCPPowerModel, Gurobi.Optimizer);

        #find the total active generation from the solution
        total_active_generation = sum(solution["solution"]["gen"]["$i"]["pg"] for i in 1:length(solution["solution"]["gen"]))

        #find the total load from the case file
        total_load = sum(case["load"]["$i"]["pd"] for i in 1:length(case["load"]))

        #find the total load shed
        push!(total_load_shed, (total_load - total_active_generation)*100/total_load)
        println("Total Load Shed: $(total_load_shed)")

        #find the total risk
        tot_risk = sum(case["branch"]["$i"]["power_risk"] for i in 1:length(case["branch"]))
        risk =0
        for i in 1:length(case["branch"])
            risk = sum(case["branch"]["$i"]["power_risk"]*solution["solution"]["branch"]["$i"]["br_status"] for i in 1:length(case["branch"]))
        end
        push!(wildfire_risk, (risk/tot_risk)*100)

        push!(total_load_shed_nn_ac, nn_opt_results["load_shed_units"][i][1])

    end
    #make a matrix to store the load shed and wildfire risk for each case into a single matrix
    
    #save the matrix to a file

    #plot wildfire risk vs load shed
    risk = nn_opt_results["wildfire_risk"]
    plot([total_load_shed, total_load_shed_nn_ac],[wildfire_risk,risk], label=[" DCOPS" "NN_ACOPS"], ylabel="Percentage of wildfire risk", xlabel="Percentage of Load Shed", title="Wildfire Risk vs Load Shed for $network_type", legend=:topright)
    savefig("benchmark_results/nn_acopf_dcops_risk_vs_load_shed_$network_type.png")

    # plot(total_load_shed, wildfire_risk, label="Total Load Shed  $network_type", ylabel="Wildfire Risk", xlabel="Total Load Shed", title="Total Load Shed vs Wildfire")
    # savefig("wildfire_ops_tests/total_load_shed_vs_wildfire_risk_$network_type.png")
   
    # plot(alpha,total_load_shed, label="Total Load Shed $network_type", xlabel="Alpha", ylabel="Total Load Shed", title="Total Load Shed vs Alpha")
    # savefig("wildfire_ops_tests/total_load_shed_vs_alpha_$network_type.png")
    # plot(alpha, wildfire_risk, label="Wildfire Risk $network_type", xlabel="Alpha", ylabel="Wildfire Risk", title="Wildfire Risk vs Alpha")
    # savefig("wildfire_ops_tests/wildfire_risk_vs_alpha_$network_type.png")



