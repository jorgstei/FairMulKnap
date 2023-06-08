struct IterationCount <: MathOptInterface.AbstractModelAttribute end
MathOptInterface.is_set_by_optimize(::IterationCount) = true
MathOptInterface.get(model::Cbc.Optimizer, ::IterationCount) = Cbc.Cbc_getIterationCount(model.inner)
# Linear Model
function linear_model_solver(values, weights, capacities, n_agents, n_items)::CBC_return
    m = Model(Cbc.Optimizer)
    set_silent(m)
    # Binary solution variable
    @variable(m, x[1:n_agents, 1:n_items], binary = true)
    #display(x)

    # Maximize sum of values of knapsacks
    @objective(m, Max, dot(values, x))

    # Ensure no bin exceeds its capacity
    for i in 1:n_agents
        @constraint(m, dot(weights[i, :], x[i, :]) <= capacities[i])
    end

    # Ensure no item is picked twice
    for i in 1:n_items
        @constraint(m, sum(x[:, i]) .<= 1)
    end

    optimize!(m)
    return CBC_return(m, floor(Int, JuMP.objective_value(m)), value.(x), MathOptInterface.get(m, MathOptInterface.NodeCount()))
end