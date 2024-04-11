using JuMP
using HiGHS
using Gurobi
using Flux
using BSON
using PowerPlots
using PowerModels

nn_model = BSON.load("wildfire_trained_model.bson")
eng = PowerModels.parse_file("case5_risk.m")

p = powerplot(eng,
# bus_data=:bus_status, bus_data_type=:quantitative,
branch_data=:power_risk, 
branch_data_type=:quantitative,
# branch_color=["#4169f1", "#4169f1","red"],
branch_color=["#4169f1","red"],
gen_data=:gen_status, gen_data_type=:ordinal, gen_color=["orange", "black"],
height=400, width=420, bus_size=150, gen_size=100, load_size=50,
)