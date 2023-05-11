function convert_from_weight_and_value_array_to_item_list(weights::Vector, values::Vector, n_agents::Int)::Vector{Item}
    list::Vector{Item} = []
    for i in 1:length(weights)
        push!(list, Item(i, weights[i], fill(values[i], n_agents)))
    end
    return list
end

function generate_items(n_items::Int, min_cost::Int, max_cost::Int, n_agents::Int, min_val::Int, max_val::Int)
    items = Vector{Item}()
    for i in range(1, n_items)
        push!(items, Item(i, rand(min_cost:max_cost), rand(min_val:max_val, n_agents)))
    end
    #println("generated", items)
    return items
end

function generate_knapsacks(n_knapsacks, min_capacity, max_capacity)
    knapsacks = Vector{Knapsack}()
    for i in range(1, n_knapsacks)
        push!(knapsacks, Knapsack(i, [], rand(min_capacity:max_capacity), 0))
    end
    return knapsacks
end

function generate_identical_items(n_items::Int, min_cost::Int, max_cost::Int, n_agents::Int, min_val::Int, max_val::Int)
    items = Vector{Item}()
    for i in range(1, n_items)
        val = rand(min_val:max_val)
        push!(items, Item(i, rand(min_cost:max_cost), repeat([val], n_agents)))
    end
    #println("generated", items)
    return items
end

function sum_items(items::Vector{Item})
    load = 0
    for item in items
        load += item.cost
    end
    return load
end

function find_heaviest_item(items::Vector{Item})
    max_id, max_cost = (1, items[1])
    for (id, item) in enumerate(items)
        if (item.cost > max_cost)
            max_id, max_cost = id, item.cost
        end
    end
    return max_id, max_cost
end

# Ensures that no items in the instance have costs above the maximum capacity of the knapsacks
function remove_infeasible_items!(bins::Vector{Knapsack}, items::Vector{Item})
    removed_items = Vector{Item}()
    indeces_to_remove = Vector{Int}()

    max_capacity = -1
    for knap in bins
        if knap.capacity > max_capacity
            max_capacity = knap.capacity
        end
    end

    for item_idx in range(1, length(items))
        if items[item_idx].cost > max_capacity
            push!(removed_items, copy(items[item_idx]))
            pushfirst!(indeces_to_remove, item_idx)
        end
    end

    for item_index in indeces_to_remove
        deleteat!(items, item_index)
    end
    return removed_items
end

# Ensures that no knapsacks in the instance have capacities below the minimum cost of the items
function remove_infeasible_knapsacks!(bins::Vector{Knapsack}, items::Vector{Item})
    removed_bins = Vector{Knapsack}()
    indeces_to_remove = Vector{Int}()

    min_cost = Inf64
    for item in items
        if item.cost < min_cost
            min_cost = item.cost
        end
    end

    for bin_idx in range(1, length(bins))
        if bins[bin_idx].capacity < min_cost
            push!(removed_bins, copy(bins[bin_idx]))
            pushfirst!(indeces_to_remove, bin_idx)
        end
    end

    for bin_index in indeces_to_remove
        deleteat!(bins, bin_index)
    end
    return removed_bins
end

function print_solution(solution::MKP_return)
    println("\nObtained solution with profit: ", solution.best_profit)

    println("With bins\n", solution.best_assignment)
    for bin in solution.best_assignment
        println("\nBin ", bin.id, "     ", bin)
        for item in bin.items
            println(item)
        end
        println("Total capacity filled: ", sum(item -> item.cost, bin.items; init=0))
        println("Total profit: ", sum(item -> item.valuations[bin.id], bin.items; init=0))
    end

end

# Util for the linear model
function duplicate_vals(array, duplicates)
    new_arr = zeros((length(array), duplicates))
    for (i, el) in enumerate(array)
        new_arr[i, 1:duplicates] = fill(el, duplicates)
    end
    return new_arr'
end