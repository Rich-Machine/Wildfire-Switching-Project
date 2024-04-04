using JuMP
using HiGHS
using Gurobi
using Flux
using BSON
using PowerModels

nn_model = BSON.load("wildfire_trained_model.bson")
eng = PowerModels.parse_file("case5_risk.m")

objective=[]
load_shed_units = []
wildfire_risk = []

# Define alpha parameter
alpha = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8, 0.9, 1)
for j in alpha

    pd = []
    qd = []
    for i in 1:5
        push!(pd, eng["load"]["$i"]["pd"])
        push!(qd, eng["load"]["$i"]["qd"])
    end
    nominal_values = append!(pd, qd)
    # Define Big M vector
    u = fill(3.025, 100)
    l = fill(-1.865, 100)

    loads = []
    for i in keys(eng["load"])
        push!(loads, eng["load"][i]["pd"])
    end
    D_p = sum(loads)

    risk = []
    for i in keys(eng["branch"])
        push!(risk, eng["branch"][i]["power_risk"])
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

    ##ReLu constraints
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
    # @constraint( model, x3[1] <= 0.2 * D_p)


    # ---Objective function
    @objective(model, 
        Min, 
        (j * x3[1]) 
        + ((1 - j)/ total_risk) * sum(risk[i] * line_status[i] for i in 1:6)
    )
    #--- Solve the model
    optimize!(model)    

    push!(objective, JuMP.objective_value(model))
    push!(objective, JuMP.value.(x3))
    push!(objective, JuMP.objective_value(model))

    #---Objective function
    # @objective(model, 
    #     Min, sum(risk[i] * line_status[i] for i in 1:6)
    # )

    latex_formulation(model)
    # solution_summary(model, verbose=true)

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
#--- Figure out how to print JuMP results
solution_summary(model)
