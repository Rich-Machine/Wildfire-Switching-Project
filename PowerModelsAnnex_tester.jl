using PowerModelsAnnex
using JuMP
using HiGHS
using Gurobi
using Flux
using BSON
using Plots
using PowerModels
using HDF5
using Ipopt
using JLD2

eng = PowerModels.parse_file("case5_risk_sole_gen.m")

model = Model(Gurobi.Optimizer)
@variable(model, line_status[1:length(eng["branch"])], Bin) 
@constraint(model, line_status .<= 1)
@objective(model, Max, sum(line_status))

#create a dcopf using PowerModels 

for i in 1:6
    eng["branch"]["$i"]["br_status"] = line_status[i]
    println(i)
end
pm = PowerModelsAnnex.ACPPowerModel(eng)
result = optimize_model!(pm, optimizer = Ipopt.Optimizer)
#pm = instantiate_model(eng, ACPPowerModel, PowerModels.build_opf)
#result = optimize_model!(pm, optimizer=Ipopt.Optimizer)
total_load = sum(eng["load"]["$i"]["pd"] for i in 1:length(eng["load"]))
println(total_load)

total_solution_generation = sum(result["solution"]["gen"]["$i"]["pg"] for i in 1:length(eng["gen"]))
println(total_solution_generation)

load_shed_units = total_load - total_solution_generation
println(load_shed_units)
#load_shed_units = sum(eng["load"]["$i"]["pd"] for i in 1:length(eng["load"])) - sum(result["solution"]["gen"]["$i"]["pg"] for i in 1:length(eng["gen"]))


optimize!(model)

#save the results to a jld2 file
#save("dcopf_results.jld2", "dcopf_results", result)

#create a dcopf using the PowerModelsAnnex package
