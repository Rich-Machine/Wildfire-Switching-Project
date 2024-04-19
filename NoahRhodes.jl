using PowerModels, PowerModelsWildfire
using Gurobi, Ipopt

# Load case data
network_type =  "base_case" #"high_risk"#"sole_gen" #"base_case"
for network_type in ["base_case", "high_risk", "sole_gen"]
    case = parse_file("case5_risk_$network_type.m")

    # Set risk parameter which determines trade-off between serving load and mitigating wildfire risk
    total_load_shed =[]
    wildfire_risk = []
    # Define alpha parameter
    # Smaller Values of Alpha ensure wildfire risk is zero and load shed is 0
    #alpha = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,  0.17, 0.18, 0.19, 0.2]
    #if network_type == "sole_gen"
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8, 0.9]
    #end

    for i in alpha
        case["risk_weight"] = i # values between 0 and 1, smaller values emphasize load delivery

        # Run OPS problem
        solution = PowerModelsWildfire.run_ops(case, DCPPowerModel, Gurobi.Optimizer);

        #find the total active generation from the solution
        total_active_generation = sum(solution["solution"]["gen"]["$i"]["pg"] for i in 1:length(solution["solution"]["gen"]))

        #find the total load from the case file
        total_load = sum(case["load"]["$i"]["pd"] for i in 1:length(case["load"]))

        #find the total load shed
        push!(total_load_shed, (total_load-total_active_generation)/total_load)

        #find the total risk
        tot_risk = sum(case["branch"]["$i"]["power_risk"] for i in 1:length(case["branch"]))
        risk =0
        for i in 1:length(case["branch"])
            risk = sum(case["branch"]["$i"]["power_risk"]*solution["solution"]["branch"]["$i"]["br_status"] for i in 1:length(case["branch"]))
        end
        push!(wildfire_risk, risk/tot_risk)
    end
    
    plot(total_load_shed, wildfire_risk, label="Total Load Shed  $network_type", ylabel="Wildfire Risk", xlabel="Total Load Shed", title="Total Load Shed vs Wildfire")
    savefig("wildfire_ops_tests/total_load_shed_vs_wildfire_risk_$network_type.png")
    plot(alpha,total_load_shed, label="Total Load Shed $network_type", xlabel="Alpha", ylabel="Total Load Shed", title="Total Load Shed vs Alpha")
    savefig("wildfire_ops_tests/total_load_shed_vs_alpha_$network_type.png")
    plot(alpha, wildfire_risk, label="Wildfire Risk $network_type", xlabel="Alpha", ylabel="Wildfire Risk", title="Wildfire Risk vs Alpha")
    savefig("wildfire_ops_tests/wildfire_risk_vs_alpha_$network_type.png")
end