using JuMP
using HiGHS
using Gurobi
using Flux
using BSON
using PowerModels

nn_model = BSON.load("wildfire_trained_model.bson")
eng = PowerModels.parse_file("case5_risk.m")

pd = []
qd = []
for i in keys(eng["load"])
    push!(pd, eng["load"][i]["pd"])
    push!(qd, eng["load"][i]["qd"])
end
nominal_values = append!(pd, qd)

# Define alpha parameter
alpha = 0.5

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
W_3 = nn_model[:model][3].weight
W_4 = nn_model[:model][4].weight

B_1 = nn_model[:model][1].bias
B_2 = nn_model[:model][2].bias
B_3 = nn_model[:model][3].bias
B_4 = nn_model[:model][4].bias


#Optimization problem
model = Model(Gurobi.Optimizer)

@variable(model, line_status[1:length(eng["branch"])], Bin) 
@variable(model, x2[1:length(nn_model[:model][1].bias)]) 
@variable(model, x3[1:length(nn_model[:model][2].bias)])
@variable(model, x4[1:length(nn_model[:model][3].bias)]) 
@variable(model, x5[1:length(nn_model[:model][4].bias)]) 


input_vector = append!(nominal_values, line_status)
@constraint(model, x2 == W_1 * input_vector + B_1)
@constraint(model, x3 == W_2 * (x2) + B_2)
@constraint(model, x4 == W_3 * (x3) + B_3)
@constraint(model, x5 == W_4 * (x4) + B_4)

@constraint( model, x5 <= 0.2 * D_p)

for i in [1:length(nn_model[:model][1].bias)]
    @constraint(model, 0 <= x2[i])
end
for i in [1:length(nn_model[:model][2].bias)]
    @constraint(model, 0 <= x3[i])
end

#---Objective function
# @objective(model, 
#     Min, 
#     (alpha * x5[1]) / D_p 
#     + ((1 - alpha)/ total_risk) * sum(risk[i] * line_status[i] for i in 1:6)
# )

#---Objective function
@objective(model, 
    Min, sum(risk[i] * line_status[i] for i in 1:6)
)
#--- Solve the model
optimize!(model)

latex_formulation(model)
# solution_summary(model, verbose=true)

# Check the status of the optimization
if termination_status(model) == MOI.OPTIMAL
    println("Optimal solution found.")
    println("Objective value: ", JuMP.objective_value(model))
    println("load_shed units: ", JuMP.value.(x5))
    println("line_status units: ", JuMP.value.(line_status))
else
    println("Optimization problem failed to find an optimal solution.")
end

#--- Figure out how to print JuMP results
solution_summary(model)
