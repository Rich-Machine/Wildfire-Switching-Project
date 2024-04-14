using PowerPlots
using PowerModels

network_type = "base_case"
# network_type = "sole_gen"
# network_type = "high_risk"

eng = PowerModels.parse_file("case5_risk_$network_type.m")

p = powerplot(eng,
branch_data=:power_risk, 
branch_data_type=:quantitative,
branch_color=["#4169f1","red"],
gen_data=:gen_status, gen_data_type=:ordinal, gen_color=["orange", "black"],
height=400, width=420, bus_size=150, gen_size=100, load_size=50,
)
