#### DC Optimal Power Flow ####

# This file provides a pedagogical example of modeling the DC Optimal Power
# Flow problem using the Julia Mathematical Programming package (JuMP) and the
# PowerModels package for data parsing.

# This file can be run by calling `include("dc-opf.jl")` from the Julia REPL or
# by calling `julia dc-opf.jl` in Julia v1.

# Developed by Line Roald (@lroald) and Carleton Coffrin (@ccoffrin)

#SOURCE: https://github.com/lanl-ansi/PowerModelsAnnex.jl/blob/master/src/model/dc-opf.jl

###############################################################################
# 0. Initialization
###############################################################################

# Load Julia Packages
#--------------------
using PowerModels
using Ipopt
using JuMP
using BilevelJuMP
using Gurobi
using Plots

# Load System Data
# ----------------
network_list = ["base_case" "sole_gen" "high_risk"]


function dcopf_wildfire_switching(network, alpha)
   
    powermodels_path = joinpath(dirname(pathof(PowerModels)), "..")

    file_name = "case5_risk_$network.m"#"$(powermodels_path)/test/data/matpower/case5.m"
    # note: change this string to modify the network data that will be loaded

    # load the data file
    data = PowerModels.parse_file(file_name)

    # Add zeros to turn linear objective functions into quadratic ones
    # so that additional parameter checks are not required
    PowerModels.standardize_cost_terms!(data, order=2)

    # Adds reasonable rate_a values to branches without them
    PowerModels.calc_thermal_limits!(data)

    # use build_ref to filter out inactive components
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    # Note: ref contains all the relevant system parameters needed to build the OPF model
    # When we introduce constraints and variable bounds below, we use the parameters in ref.

    ################################################################################################################
    # AW. This is the data that will be used to build the bilevel optimization problem from wildfire_nn_optimization
    ################################################################################################################
    # Get the wildfire risk
    eng = PowerModels.parse_file("case5_risk_$network.m")
    risk = []
    for i in 1:length(ref[:branch])
        push!(risk, eng["branch"]["$i"]["power_risk"])
    end
    total_risk = sum(risk)

    # Get the total load
    loads = []
    for i in keys(eng["load"])
        push!(loads, eng["load"][i]["pd"])
    end
    D_p = sum(loads)
    print(D_p)

    ###############################################################################
    # 1. Building the Optimal Power Flow Model
    ###############################################################################

    # Initialize a JuMP Optimization Model
    #-------------------------------------
    #model = Model(Ipopt.Optimizer)


    ##############################################################################################################
    # AW. Using BilevelJuMP to setup the 1st stage as the trade-off and the second stage as the conventional DCOPF
    ##############################################################################################################
    model = BilevelModel(
        Gurobi.Optimizer,
        mode = BilevelJuMP.ProductMode()#FortunyAmatMcCarlMode(primal_big_M = 100, dual_big_M = 100)
    )
    ##############################################################################################################
    # Define the upper level problems
    @variable(Upper(model), line_status[1:length(data["branch"])], Bin) 
    @variable(Upper(model), total_load_shed >= 0)
    @constraint(Upper(model), total_load_shed <= alpha*D_p)
    @objective(Upper(model), Min, sum(risk[i] * line_status[i] for i in 1:length(data["branch"]))/total_risk
    )


    # DCOPF Formulation
    set_optimizer_attribute(model, "NonConvex", 2)
    #set_optimizer_attribute(model, "print_level", 0)

    # note: print_level changes the amount of solver information printed to the terminal


    # Add Optimization and State Variables
    # ------------------------------------

    # Add voltage angles va for each bus
    @variable(Lower(model), va[i in keys(ref[:bus])])
    # note: [i in keys(ref[:bus])] adds one `va` variable for each bus in the network

    # Add active power generation variable pg for each generator (including limits)
    @variable(Lower(model), ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])

    # Add power flow variables p to represent the active power flow for each branch
    @variable(Lower(model), -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs_from]] <= ref[:branch][l]["rate_a"])

    #######################################################################################################
    # Insertion of line switching branch status as a variable in the DC OPF Lower(model), Amanda West
    #######################################################################################################
    for i in 1:6
        data["branch"]["$i"]["br_status"] = line_status[i]
        println(i)
    end
    ########################################################################################################

    # Build JuMP expressions for the value of p[(l,i,j)] and p[(l,j,i)] on the branches
    p_expr = Dict([((l,i,j), 1.0*p[(l,i,j)]) for (l,i,j) in ref[:arcs_from]])
    p_expr = merge(p_expr, Dict([((l,j,i), -1.0*p[(l,i,j)]) for (l,i,j) in ref[:arcs_from]]))
    # note: this is used to make the definition of nodal power balance simpler

    # Add power flow variables p_dc to represent the active power flow for each HVDC line
    @variable(Lower(model), p_dc[a in ref[:arcs_dc]])

    for (l,dcline) in ref[:dcline]
        f_idx = (l, dcline["f_bus"], dcline["t_bus"])
        t_idx = (l, dcline["t_bus"], dcline["f_bus"])

        JuMP.set_lower_bound(p_dc[f_idx], dcline["pminf"])
        JuMP.set_upper_bound(p_dc[f_idx], dcline["pmaxf"])

        JuMP.set_lower_bound(p_dc[t_idx], dcline["pmint"])
        JuMP.set_upper_bound(p_dc[t_idx], dcline["pmaxt"])
    end


    # Add Objective Function
    # ----------------------

    # index representing which side the HVDC line is starting
    from_idx = Dict(arc[1] => arc for arc in ref[:arcs_from_dc])

    # Minimize the cost of active power generation and cost of HVDC line usage
    # assumes costs are given as quadratic functions
    @objective(Lower(model), Min,
        sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]) +
        sum(dcline["cost"][1]*p_dc[from_idx[i]]^2 + dcline["cost"][2]*p_dc[from_idx[i]] + dcline["cost"][3] for (i,dcline) in ref[:dcline])
    )


    # Add Constraints
    # ---------------

    # Fix the voltage angle to zero at the reference bus
    for (i,bus) in ref[:ref_buses]
        @constraint(Lower(model), va[i] == 0)
    end

    # Nodal power balance constraints
    for (i,bus) in ref[:bus]
        # Build a list of the loads and shunt elements connected to the bus i
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        # Active power balance at node i
        @constraint(Lower(model),
            sum(p_expr[a] for a in ref[:bus_arcs][i]) +                  # sum of active power flow on lines from bus i +
            sum(p_dc[a_dc] for a_dc in ref[:bus_arcs_dc][i]) ==     # sum of active power flow on HVDC lines from bus i =
            sum(pg[g] for g in ref[:bus_gens][i]) -                 # sum of active power generation at bus i -
            sum(load["pd"] for load in bus_loads) -                 # sum of active load consumption at bus i -
            sum(shunt["gs"] for shunt in bus_shunts)*1.0^2          # sum of active shunt element injections at bus i
        )
    end

    # Branch power flow physics and limit constraints
    for (i,branch) in ref[:branch]
        # Build the from variable id of the i-th branch, which is a tuple given by (branch id, from bus, to bus)
        f_idx = (i, branch["f_bus"], branch["t_bus"])

        p_fr = p[f_idx]                     # p_fr is a reference to the optimization variable p[f_idx]

        va_fr = va[branch["f_bus"]]         # va_fr is a reference to the optimization variable va on the from side of the branch
        va_to = va[branch["t_bus"]]         # va_fr is a reference to the optimization variable va on the to side of the branch

        # Compute the branch parameters and transformer ratios from the data
        g, b = PowerModels.calc_branch_y(branch)

        # DC Power Flow Constraint
        @constraint(Lower(model), p_fr == -b*(va_fr - va_to))
        # note: that upper and lower limits on the power flow (i.e. p_fr) are not included here.
        #   these limits were already enforced for p (which is the same as p_fr) when
        #   the optimization variable p was defined (around line 65).


        # Voltage angle difference limit
        @constraint(Lower(model), va_fr - va_to <= branch["angmax"])
        @constraint(Lower(model), va_fr - va_to >= branch["angmin"])
    end

    # HVDC line constraints
    for (i,dcline) in ref[:dcline]
        # Build the from variable id of the i-th HVDC line, which is a tuple given by (hvdc line id, from bus, to bus)
        f_idx = (i, dcline["f_bus"], dcline["t_bus"])
        # Build the to variable id of the i-th HVDC line, which is a tuple given by (hvdc line id, to bus, from bus)
        t_idx = (i, dcline["t_bus"], dcline["f_bus"])   # index of the ith HVDC line which is a tuple given by (line number, to bus, from bus)
        # note: it is necessary to distinguish between the from and to sides of a HVDC line due to power losses

        # Constraint defining the power flow and losses over the HVDC line
        @constraint(Lower(model), (1-dcline["loss1"])*p_dc[f_idx] + (p_dc[t_idx] - dcline["loss0"]) == 0)
    end

    # Load Shedding Constraint
    total_generation = []
    for i in 1:length(ref[:gen])
        push!(total_generation, pg[i])
    end
    @constraint(Lower(model), total_load_shed == D_p - sum(total_generation))




    ###############################################################################
    # 3. Solve the Optimal Power Flow Model and Review the Results
    ###############################################################################

    # Solve the optimization problem
    optimize!(model)


    ###############################################################################
    # 4. Export the Results
    ###############################################################################


    # Check that the solver terminated without an error
    println("The solver termination status is $(termination_status(model))")

    # Check the value of the objective function
    risk_percentage = objective_value(Upper(model))
    cost = objective_value(Lower(model))
    println("The cost of generation is $(cost).")

    # Check the value of an optimization variable
    # Example: Active power generated at generator 1
    pg1 = value(pg[1])
    println("The active power generated at generator 1 is $(pg1*ref[:baseMVA]) MW.")
    # note: the optimization model is in per unit, so the baseMVA value is used to restore the physical units

    total_gen = sum(value.(pg))
    ls = value(total_load_shed)
    println("The load shed is $(ls*ref[:baseMVA]) MW.")
    #println("The total load is $(D_p*ref[:baseMVA]) MW.")
    println("The total generation is $(total_gen*ref[:baseMVA]) MW.")

    # Return the results
    return cost, risk_percentage, ls, total_gen, D_p, value.(line_status), risk

end

# Run the DCOPF
#--------------
ls_vec = []
total_gen_vec = []
risk_percent_vec = []
line_status_vec = []


network_ls_percent_vec = []
net_list = ["base_case" "high_risk"]
for i = 1:length(net_list)
    network = net_list[i]
    alpha = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,  0.17, 0.18, 0.19, 0.2]
    #if network == "sole_gen"
     #   alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8, 0.9]
    #end
    for alpha in alpha
        cost, risk_percentage, ls, total_gen, D_p, line_status, risk = dcopf_wildfire_switching(network, alpha)
        push!(ls_vec, ls)
        push!(total_gen_vec, total_gen)
        push!(risk_percent_vec, risk_percentage)
        push!(line_status_vec, line_status)
    end

    D_p = 14

    hcat(network_ls_percent_vec,ls_vec/D_p)
end

    plot([network_ls_percent_vec[1],network_ls_percent_vec[2]],risk_percent_vec, label=["base_case" "high_risk"], ylabel="Wildfire Risk", xlabel="% Load Shed", title="Wildfire Risk vs Load Shed", legend=:topright)
    savefig("conventional_opf_tests/dc_opf_risk_vs_load_shed_all_networks.png")