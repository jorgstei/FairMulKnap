# Linear Model
function linear_model_solver(values, weights, capacities, n_agents, n_items)
    m = Model(Cbc.Optimizer)
    set_silent(m)
    # Binary solution variable
    @variable(m, x[1:n_agents, 1:n_items], binary = true)

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
end