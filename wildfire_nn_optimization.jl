using JuMP
using HiGHS
using Gurobi
using Flux
using BSON
using Plots
using PowerModels
using HDF5
using JLD2


# network_type = "base_case"
 network_type = "sole_gen"
#network_type = "high_risk"

nn_model = BSON.load("wildfire_trained_model_$network_type.bson")
eng = PowerModels.parse_file("case5_risk_$network_type.m")

objective=[]
load_shed_units = []
wildfire_risk = []
line_1 = []
line_2 = []
line_3 = []
line_4 = []
line_5 = []
line_6 = []

# Define alpha parameter
alpha = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,  0.17, 0.18, 0.19, 0.2)
if network_type == "sole_gen"
    alpha = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8, 0.9)
end

for j in alpha
    print("This is $j\n \n")
    pd = []
    qd = []
    for i in 1:5
        push!(pd, eng["load"]["$i"]["pd"])
        push!(qd, eng["load"]["$i"]["qd"])
    end
    nominal_values = append!(pd, qd)

    # Define Big M vector
    if network_type == "base_case"
        u = fill(2.9504, 100)
        l = fill(-1.8339, 100)
    elseif network_type == "sole_gen"
        u = fill(2.7504, 100)
        l = fill(-1.863, 100)
    elseif network_type == "high_risk"
        u = fill(2.974, 100)
        l = fill(-1.81, 100)
    end

    loads = []
    for i in keys(eng["load"])
        push!(loads, eng["load"][i]["pd"])
    end
    D_p = sum(loads)

    risk = []
    for i in 1:6
        push!(risk, eng["branch"]["$i"]["power_risk"])
    end
    total_risk = sum(risk)

    W_1 = nn_model[:model][1].weight
    W_2 = nn_model[:model][2].weight

    B_1 = nn_model[:model][1].bias
    B_2 = nn_model[:model][2].bias

    #Optimization problem
    model = Model(Gurobi.Optimizer)

    @variable(model, line_status[1:length(eng["branch"])], Bin) 
    @variable(model, x2[1:length(nn_model[:model][1].bias)]) 
    @variable(model, x3[1:length(nn_model[:model][2].bias)])
    @variable(model, z2[1:length(nn_model[:model][1].bias)], Bin) 

    input_vector = append!(nominal_values, line_status)
    @constraint(model, x3 == W_2 * (x2) + B_2)
    @expression(model, without_bias, W_1 * input_vector)

    ## ReLu constraints
    for i in 1:100
        @constraint(model, x2[i] >= 0)
    end
    for i in 1:100
        @constraint(model, x2[i] >= without_bias[i] + B_1[i])
    end
    for i in 1:100
        @constraint(model, x2[i] <= u[i] * z2[i])
    end
    for i in 1:100
        @constraint(model, x2[i] <= without_bias[i] + B_1[i] - l[i] * (1 - z2[i]))
    end
    @constraint(model, x3 >= 0)
    @constraint( model, x3[1] <= j*D_p)

    display(j)

    # # ---Objective function
    # @objective(model, 
    #     Min, 
    #     (j * x3[1])/D_p
    #     + 
    #     ((1 - j) /total_risk)* sum(risk[i] * line_status[i] for i in 1:6)
    # )

    #---Objective function
    @objective(model, Min, sum(risk[i] * line_status[i] for i in 1:6)/total_risk
    )

    #--- Solve the model
    optimize!(model)    

    push!(objective, JuMP.objective_value(model))
    push!(load_shed_units, JuMP.value.(x3)/D_p)
    line_risk = sum(risk[i] * JuMP.value.(line_status)[i] for i in 1:6)
    push!(wildfire_risk, line_risk/total_risk)
    push!(line_1, JuMP.value.(line_status)[1])
    push!(line_2, JuMP.value.(line_status)[2])
    push!(line_3, JuMP.value.(line_status)[3])
    push!(line_4, JuMP.value.(line_status)[4])
    push!(line_5, JuMP.value.(line_status)[5])
    push!(line_6, JuMP.value.(line_status)[6])

    # Check the status of the optimization
    if termination_status(model) == MOI.OPTIMAL
        println("Optimal solution found.")
        println("Objective value: ", JuMP.objective_value(model))
        println("load_shed units: ", JuMP.value.(x3))
        println("line_status units: ", JuMP.value.(line_status))
        println("Binary Variables: ", JuMP.value.(z2))
    else
        println("Optimization problem failed to find an optimal solution.")
    end
end 

alpha = range(0.01, 0.2, length=20)
if network_type == "sole_gen"
    alpha = range(0.1, 1, length=9)
end

load_shed_units_combined = vcat(load_shed_units...)
plot(load_shed_units_combined, wildfire_risk, title = "Total Risk VS Total Load Shed for $network_type", xlabel = "% of load shed", ylabel = "Total Wildfire risk", label="Total Risk")

## Plotting the heat wave for the line statuses
# Define the lines
lines = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6"]

# Assuming line_1 to line_6 are defined as arrays
line_matrix = hcat(line_1, line_2, line_3, line_4, line_5, line_6)
line_matrix = line_matrix'

# Generate x-axis labels
xs = ["Step $i" for i in 1:20]

# Define a custom colormap for 0 and 1
custom_colors = Dict(0 => "blue", 1 => "red")

# Create the heatmap with custom colorscale
heatmap(xs, lines, line_matrix, aspect_ratio = 2, colorscale=custom_colors)

#create a dictionary to store the results
results = Dict("objective" => objective, "load_shed_units" => load_shed_units, "wildfire_risk" => wildfire_risk, "line_1" => line_1, "line_2" => line_2, "line_3" => line_3, "line_4" => line_4, "line_5" => line_5, "line_6" => line_6, "alpha" => alpha)
#save the results 
save("nn_opt_results_$network_type.jld2", "nn_opt_results_$network_type", results)
