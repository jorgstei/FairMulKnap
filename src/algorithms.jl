
#=
    The choose_bin function described in "A branch and bound algorithm for hard multiple knapsack problems" - Fukunaga
=#
function get_bin_with_least_capacity(bins::Vector{Knapsack}, items::Vector{Item})
    smallest_remaining_bin = findmin((bin) -> bin.capacity, bins)
    return bins[smallest_remaining_bin[2]]
end
#=
    An alternative to get_bin_with_least_remaining_capacity, which instead takes the bin with the most max valuations
=#
function get_bin_with_most_max_valuations(bins::Vector{Knapsack}, items::Vector{Item})
    scores = zeros(length(bins))
    ids_of_remaining_bins = map((bin) -> bin.id, bins)
    for item in items
        max = findmax([item.valuations[i] for i in ids_of_remaining_bins])
        scores[max[2]] += 1
    end
    return bins[argmax(scores)]
end

function get_bin_with_most_capacity(bins::Vector{Knapsack}, items::Vector{Item})
    largest_remaining_bin = findmax((bin) -> bin.capacity, bins)
    return bins[largest_remaining_bin[2]]
end

function get_bin_with_smallest_capacity_divided_by_n_max_valuations(bins::Vector{Knapsack}, items::Vector{Item})
    scores = zeros(length(bins))
    ids_of_remaining_bins = map((bin) -> bin.id, bins)
    for item in items
        max = findmax([item.valuations[i] for i in ids_of_remaining_bins])
        scores[max[2]] += 1
    end
    final = map(bin -> bin[2].capacity / scores[bin[1]], enumerate(bins))
    return bins[argmin(final)]
end

#=
    Dynamic programming solution to the 0-1 knapsack problem
    Used to compute the upper bound in MKP
    Inspired by: https://en.wikipedia.org/wiki/Knapsack_problem#Solving
=#
function solve_binary_knapsack(bin::Knapsack, items::Vector{Item}, valuation_index::Int, return_assignment::Bool)
    # Add filler item for the algorithm to work properly
    items_copy = copy(items)
    pushfirst!(items_copy, Item(-1, 0, [0]))
    # Init values
    profits = fill(-1, (length(items_copy), bin.capacity + 1))
    profits[1, 1] = 0
    # Return profit of cell i, j
    function get_profit!(i, j)
        if (i == 1 || j == 1)
            profits[i, j] = 0
        else
            if (profits[i-1, j] == -1)
                profits[i-1, j] = get_profit!(i - 1, j)
            end
            if (items_copy[i].cost > j - 1)
                profits[i, j] = profits[i-1, j]
            else
                if (j - items_copy[i].cost >= 1)

                    if (profits[i-1, j-items_copy[i].cost] == -1)
                        profits[i-1, j-items_copy[i].cost] = get_profit!(i - 1, j - items_copy[i].cost)
                    end
                    profits[i, j] = max(profits[i-1, j], profits[i-1, j-items_copy[i].cost] + items_copy[i].valuations[valuation_index])
                else
                    profits[i, j] = profits[i-1, j]
                end
            end
        end
    end
    # Get profit of full problem
    get_profit!(length(items_copy), bin.capacity + 1)

    # For returning the solution instead of just the value.
    if return_assignment
        optimal_solution_index = findmax(profits)[2]
        if optimal_solution_index[1] < -1 || optimal_solution_index[2] < -1
            println("Profits:")
            display(profits)
            println("optimal", optimal_solution_index)
        end

        function get_binary_knapsack_solution(i, j)
            # not sure abount j == 1
            if (i == 1 || j == 1)
                return []
            end
            if (profits[i, j] > profits[i-1, j])
                if j - items_copy[i].cost < 0
                    println("Profits:")
                    display(profits)
                    println("optimal", optimal_solution_index)
                    println("i: ", i, " j: ", j)
                    println("Adding item: ", items_copy[i])
                end
                push!(get_binary_knapsack_solution(i - 1, j - items_copy[i].cost), items_copy[i])
            else
                return get_binary_knapsack_solution(i - 1, j)
            end
        end
        #println("optimal", optimal_solution_index)
        solution = get_binary_knapsack_solution(optimal_solution_index[1], optimal_solution_index[2])
        #println("Solution: ", solution)
        return (profits[length(items_copy), bin.capacity+1], solution)
    end

    return profits[length(items_copy), bin.capacity+1]
end

# solve 0-1 knapsack problem on the aggregate knapsack and return the value
# Works for items with a single valuation
function compute_surrogate_upper_bound(bins::Vector{Knapsack}, items::Vector{Item})
    aggregate_knapsack = Knapsack(0, [], sum((bin) -> bin.capacity, bins; init=0), 0)
    return solve_binary_knapsack(aggregate_knapsack, items, 1, false)
end

# the idea:
# Surrogate sucks cause max of all valuations, but LP-relaxed can respect the valuation of the current agent
# If every knapsack gets it's most efficient item, with ties broken by 

#=


ERROR: RESULTS DIFFER BETWEEN MODELS:

4-element Vector{Main.FairMulKnap.Knapsack}:
 Main.FairMulKnap.Knapsack(1, Main.FairMulKnap.Item[], 111, 0)
 Main.FairMulKnap.Knapsack(2, Main.FairMulKnap.Item[], 60, 0)
 Main.FairMulKnap.Knapsack(3, Main.FairMulKnap.Item[], 101, 0)
 Main.FairMulKnap.Knapsack(4, Main.FairMulKnap.Item[], 100, 0)
12-element Vector{Main.FairMulKnap.Item}:
 Main.FairMulKnap.Item(1, 19, [20, 13, 20, 2])
 Main.FairMulKnap.Item(2, 57, [12, 13, 1, 7])
 Main.FairMulKnap.Item(3, 41, [16, 12, 2, 5])
 Main.FairMulKnap.Item(4, 20, [13, 18, 10, 11])
 Main.FairMulKnap.Item(5, 17, [18, 5, 1, 8])
 Main.FairMulKnap.Item(6, 11, [4, 1, 17, 11])
 Main.FairMulKnap.Item(7, 21, [12, 20, 20, 17])
 Main.FairMulKnap.Item(8, 34, [16, 10, 18, 19])
 Main.FairMulKnap.Item(9, 59, [6, 10, 7, 8])
 Main.FairMulKnap.Item(10, 48, [4, 10, 9, 11])
 Main.FairMulKnap.Item(11, 43, [20, 3, 13, 18])
 Main.FairMulKnap.Item(12, 10, [19, 18, 7, 18])


undominated_r2 = BenchmarkResults("R2", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, false, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
Obtained solution with profit: 185
With bins
Main.FairMulKnap.Knapsack[Main.FairMulKnap.Knapsack(1, Main.FairMulKnap.Item[Main.FairMulKnap.Item(12, 10, [19, 18, 7, 18]), Main.FairMulKnap.Item(5, 17, [18, 5, 1, 8]), Main.FairMulKnap.Item(11, 43, [20, 3, 13, 18]), Main.FairMulKnap.Item(3, 41, [16, 12, 2, 5])], 111, 0), Main.FairMulKnap.Knapsack(3, Main.FairMulKnap.Item[Main.FairMulKnap.Item(6, 11, [4, 1, 17, 11]), Main.FairMulKnap.Item(1, 19, [20, 13, 20, 2]), Main.FairMulKnap.Item(9, 59, [6, 10, 7, 8])], 101, 0), Main.FairMulKnap.Knapsack(4, Main.FairMulKnap.Item[Main.FairMulKnap.Item(8, 34, [16, 10, 18, 19]), Main.FairMulKnap.Item(10, 48, [4, 10, 9, 11])], 100, 0), Main.FairMulKnap.Knapsack(2, Main.FairMulKnap.Item[Main.FairMulKnap.Item(7, 21, [12, 20, 20, 17]), Main.FairMulKnap.Item(4, 20, [13, 18, 10, 11])], 60, 0)]

Bin 1     Main.FairMulKnap.Knapsack(1, Main.FairMulKnap.Item[Main.FairMulKnap.Item(12, 10, [19, 18, 7, 18]), Main.FairMulKnap.Item(5, 17, [18, 5, 1, 8]), Main.FairMulKnap.Item(11, 43, [20, 3, 13, 18]), Main.FairMulKnap.Item(3, 41, [16, 12, 2, 5])], 111, 0)
Main.FairMulKnap.Item(12, 10, [19, 18, 7, 18])
Main.FairMulKnap.Item(5, 17, [18, 5, 1, 8])
Main.FairMulKnap.Item(11, 43, [20, 3, 13, 18])
Main.FairMulKnap.Item(3, 41, [16, 12, 2, 5])
Total capacity filled: 111
Total profit: 73

Bin 3     Main.FairMulKnap.Knapsack(3, Main.FairMulKnap.Item[Main.FairMulKnap.Item(6, 11, [4, 1, 17, 11]), Main.FairMulKnap.Item(1, 19, [20, 13, 20, 2]), Main.FairMulKnap.Item(9, 59, [6, 10, 7, 8])], 101, 0)
Main.FairMulKnap.Item(6, 11, [4, 1, 17, 11])
Main.FairMulKnap.Item(1, 19, [20, 13, 20, 2])
Main.FairMulKnap.Item(9, 59, [6, 10, 7, 8])
Total capacity filled: 89
Total profit: 44

Bin 4     Main.FairMulKnap.Knapsack(4, Main.FairMulKnap.Item[Main.FairMulKnap.Item(8, 34, [16, 10, 18, 19]), Main.FairMulKnap.Item(10, 48, [4, 10, 9, 11])], 100, 0)
Main.FairMulKnap.Item(8, 34, [16, 10, 18, 19])
Main.FairMulKnap.Item(10, 48, [4, 10, 9, 11])
Total capacity filled: 82
Total profit: 30

Bin 2     Main.FairMulKnap.Knapsack(2, Main.FairMulKnap.Item[Main.FairMulKnap.Item(7, 21, [12, 20, 20, 17]), Main.FairMulKnap.Item(4, 20, [13, 18, 10, 11])], 60, 0)
Main.FairMulKnap.Item(7, 21, [12, 20, 20, 17])
Main.FairMulKnap.Item(4, 20, [13, 18, 10, 11])
Total capacity filled: 41
Total profit: 38


undominated_lp_bound = BenchmarkResults("LP-relaxed", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], lp_relaxed_upper_bound, false, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
Obtained solution with profit: 184
With bins
Main.FairMulKnap.Knapsack[Main.FairMulKnap.Knapsack(1, Main.FairMulKnap.Item[Main.FairMulKnap.Item(5, 17, [18, 5, 1, 8]), Main.FairMulKnap.Item(11, 43, [20, 3, 13, 18]), Main.FairMulKnap.Item(3, 41, [16, 12, 2, 5])], 111, 0), Main.FairMulKnap.Knapsack(3, Main.FairMulKnap.Item[Main.FairMulKnap.Item(6, 11, [4, 1, 17, 11]), Main.FairMulKnap.Item(1, 19, [20, 13, 20, 2]), Main.FairMulKnap.Item(9, 59, [6, 10, 7, 8])], 101, 0), Main.FairMulKnap.Knapsack(4, Main.FairMulKnap.Item[Main.FairMulKnap.Item(12, 10, [19, 18, 7, 18]), Main.FairMulKnap.Item(8, 34, [16, 10, 18, 19]), Main.FairMulKnap.Item(10, 48, [4, 10, 9, 11])], 100, 0), Main.FairMulKnap.Knapsack(2, Main.FairMulKnap.Item[Main.FairMulKnap.Item(7, 21, [12, 20, 20, 17]), Main.FairMulKnap.Item(4, 20, [13, 18, 10, 11])], 60, 0)]

Bin 1     Main.FairMulKnap.Knapsack(1, Main.FairMulKnap.Item[Main.FairMulKnap.Item(5, 17, [18, 5, 1, 8]), Main.FairMulKnap.Item(11, 43, [20, 3, 13, 18]), Main.FairMulKnap.Item(3, 41, [16, 12, 2, 5])], 111, 0)
Main.FairMulKnap.Item(5, 17, [18, 5, 1, 8])
Main.FairMulKnap.Item(11, 43, [20, 3, 13, 18])
Main.FairMulKnap.Item(3, 41, [16, 12, 2, 5])
Total capacity filled: 101
Total profit: 54

Bin 3     Main.FairMulKnap.Knapsack(3, Main.FairMulKnap.Item[Main.FairMulKnap.Item(6, 11, [4, 1, 17, 11]), Main.FairMulKnap.Item(1, 19, [20, 13, 20, 2]), Main.FairMulKnap.Item(9, 59, [6, 10, 7, 8])], 101, 0)
Main.FairMulKnap.Item(6, 11, [4, 1, 17, 11])
Main.FairMulKnap.Item(1, 19, [20, 13, 20, 2])
Main.FairMulKnap.Item(9, 59, [6, 10, 7, 8])
Total capacity filled: 89
Total profit: 44

Bin 4     Main.FairMulKnap.Knapsack(4, Main.FairMulKnap.Item[Main.FairMulKnap.Item(12, 10, [19, 18, 7, 18]), Main.FairMulKnap.Item(8, 34, [16, 10, 18, 19]), Main.FairMulKnap.Item(10, 48, [4, 10, 9, 11])], 100, 0)
Main.FairMulKnap.Item(12, 10, [19, 18, 7, 18])
Main.FairMulKnap.Item(8, 34, [16, 10, 18, 19])
Main.FairMulKnap.Item(10, 48, [4, 10, 9, 11])
Total capacity filled: 92
Total profit: 48

Bin 2     Main.FairMulKnap.Knapsack(2, Main.FairMulKnap.Item[Main.FairMulKnap.Item(7, 21, [12, 20, 20, 17]), Main.FairMulKnap.Item(4, 20, [13, 18, 10, 11])], 60, 0)
Main.FairMulKnap.Item(7, 21, [12, 20, 20, 17])
Main.FairMulKnap.Item(4, 20, [13, 18, 10, 11])
Total capacity filled: 41
Total profit: 38


ERROR: RESULTS DIFFER BETWEEN MODELS:
=#
# LP-relaxed - THIS DOES NOT WORK, SEE ABOVE EXAMPLE
function lp_relaxed_upper_bound(bins::Vector{Knapsack}, items::Vector{Item})
    remaining_bins = deepcopy(bins)
    filled_bins = []
    ids_of_remaining_bins = map((bin) -> bin.id, remaining_bins)
    while length(remaining_bins) > 0
        items = sort(items, lt=(a, b) -> is_smaller_max_profit_divided_by_weight(a, b, ids_of_remaining_bins), rev=true)
        #println("Got items: ", items)

        for (idx, item) in enumerate(items)
            max_remaining_valuation = maximum(item.valuations[i] for i in ids_of_remaining_bins)
            most_benefitted_knap_id = intersect(findall((val) -> val == max_remaining_valuation, item.valuations), ids_of_remaining_bins)[1]
            #println("Max val of item ", item, " is ", max_remaining_valuation, " id: ", most_benefitted_knap_id, " knaps ", remaining_bins, "\n excluding ", filled_bins)
            #println("ids of remaining ", ids_of_remaining_bins)
            most_benefitted_knap = remaining_bins[findfirst((bin) -> bin.id == most_benefitted_knap_id, remaining_bins)]
            #println("Max val of item ", item, " is ", max_remaining_valuation, " id: ", most_benefitted_knap_id, " knap ", most_benefitted_knap)

            remaining_capacity = most_benefitted_knap.capacity - most_benefitted_knap.load
            #println("remaining cap: ", remaining_capacity, " vs ", item.cost)
            if item.cost <= remaining_capacity
                #println("Item ", item, " fits in knap ", most_benefitted_knap)
                push!(most_benefitted_knap.items, item)
                most_benefitted_knap.load += item.cost
                if most_benefitted_knap.capacity == most_benefitted_knap.load
                    #println("It fit perfectly")
                    remaining_bins = setdiff(remaining_bins, [most_benefitted_knap])
                    ids_of_remaining_bins = setdiff(ids_of_remaining_bins, [most_benefitted_knap_id])
                    push!(filled_bins, most_benefitted_knap)
                    break
                end
            else
                push!(most_benefitted_knap.items, Item(-1, remaining_capacity, fill(ceil(Int, max_remaining_valuation * remaining_capacity / item.cost), length(item.valuations))))
                # Add the remaining part of the item to our items
                valuations_excluding_bin = copy(item.valuations)
                #println("valuations: ", item.valuations)
                valuations_excluding_bin[most_benefitted_knap.id] = 0
                rest_of_item = Item(-1, item.cost - remaining_capacity, [ceil(Int, val * (item.cost - remaining_capacity) / item.cost) for val in valuations_excluding_bin])
                # Add the rest of the item to items
                insert!(items, idx + 1, rest_of_item)
                #println("Added item ", rest_of_item, " to ", items[idx:end], " with remaining bins ", remaining_bins)
                remaining_bins = setdiff(remaining_bins, [most_benefitted_knap])
                ids_of_remaining_bins = setdiff(ids_of_remaining_bins, [most_benefitted_knap_id])
                push!(filled_bins, most_benefitted_knap)
                # Remove all allocated items and break to re-sort the rest
                items = items[idx:end]
                break
            end
        end
    end
    total_profit = 0
    for bin in filled_bins
        #println(bin)
        total_profit += sum(item -> item.valuations[bin.id], bin.items; init=0)
    end
    #println("Got bound", total_profit)
    return total_profit
end
# Return the upper bound considering individual valuations
# Compute aggregate knapsack, and assign each item it's maximum valuation before solving binary knapsack
function compute_max_upper_bound_individual_vals(bins::Vector{Knapsack}, items::Vector{Item})
    aggregate_knapsack = Knapsack(0, [], sum((bin) -> bin.capacity, bins; init=0), 0)
    adjusted_items = Array{Item}(undef, 0)
    ids_of_remaining_bins = map((bin) -> bin.id, bins)
    for item in items
        relevant_valuations = [item.valuations[i] for i in ids_of_remaining_bins]
        itemcpy = Item(item.id, item.cost, fill(maximum(relevant_valuations), length(item.valuations)))
        push!(adjusted_items, itemcpy)
    end
    return solve_binary_knapsack(aggregate_knapsack, adjusted_items, 1, false)
end

# Bound-and-Bound
# Computes a lower bound by first filling knap 1 optimally, then reducing and filling knap 2 optimally ...
function bound_and_bound(bins::Vector{Knapsack}, items::Vector{Item})
    items::Vector{Item} = copy(items)
    assignments = []
    profits = []
    for bin in bins
        res = solve_binary_knapsack(bin, items, bin.id, true)
        push!(assignments, Knapsack(bin.id, res[2], bin.capacity, bin.load))
        items = setdiff(items, res[2])
        push!(profits, res[1])
    end
    return (sum(profits), assignments)
end

# The program should keep the items sorted throughout so long as the items were sorted by profit/weight initially.
# This happens automatically in solve_multiple_knapsack_problem
# Works for identical valuations
function pisinger_r2_reduction(bins::Vector{Knapsack}, sorted_items::Vector{Item}, sum_profit, lower_bound)
    # SMKP
    aggregate_knapsack = Knapsack(0, [], sum((bin) -> bin.capacity, bins; init=0), 0)
    # Find break item
    break_item = Nothing
    break_item_idx = -1

    profits_of_efficient_bundle = 0
    weights_of_efficient_bundle = 0
    for (idx, item) in enumerate(sorted_items)
        weights_of_efficient_bundle += item.cost
        profits_of_efficient_bundle += item.valuations[1]

        if weights_of_efficient_bundle >= aggregate_knapsack.capacity
            break_item = item
            break_item_idx = idx
            weights_of_efficient_bundle -= item.cost
            profits_of_efficient_bundle -= item.valuations[1]
            break
        end
    end

    # If all items fit in the combined knapsack none can be removed
    if break_item_idx == -1
        return Item[]
    end

    items_to_remove = Item[]

    for item in sorted_items[break_item_idx:length(sorted_items)]
        upper_bound = profits_of_efficient_bundle + item.valuations[1] +
                      floor(
                          (aggregate_knapsack.capacity - weights_of_efficient_bundle - item.cost) *
                          break_item.valuations[1] / break_item.cost
                      )
        if upper_bound + sum_profit <= lower_bound
            push!(items_to_remove, item)
        end
    end
    return items_to_remove
end

# TODO what do we do with sum_profits and individual vals?
function pisinger_r2_reduction_individual_valuations(bins::Vector{Knapsack}, items::Vector{Item}, sum_profit, lower_bound)
    # SMKP
    ids_of_remaining_bins = map((bin) -> bin.id, bins)
    sorted_items = sort(items, lt=(a, b) -> is_smaller_max_profit_divided_by_weight(a, b, ids_of_remaining_bins), rev=true)
    aggregate_knapsack = Knapsack(0, [], sum((bin) -> bin.capacity, bins; init=0), 0)

    # Find break item
    break_item = Nothing
    break_item_idx = -1
    profits_of_efficient_bundle = 0
    weights_of_efficient_bundle = 0
    for (idx, item) in enumerate(sorted_items)
        weights_of_efficient_bundle += item.cost
        profits_of_efficient_bundle += maximum([item.valuations[i] for i in ids_of_remaining_bins])

        if weights_of_efficient_bundle >= aggregate_knapsack.capacity
            break_item = item
            break_item_idx = idx
            weights_of_efficient_bundle -= item.cost
            profits_of_efficient_bundle -= maximum([item.valuations[i] for i in ids_of_remaining_bins])
            break
        end
    end

    # If all items fit in the combined knapsack none can be removed
    if break_item_idx == -1
        return Item[]
    end

    items_to_remove = Item[]

    for item in sorted_items[break_item_idx:length(sorted_items)]
        upper_bound = profits_of_efficient_bundle + maximum([item.valuations[i] for i in ids_of_remaining_bins]) +
                      floor(
                          (aggregate_knapsack.capacity - weights_of_efficient_bundle - item.cost) *
                          maximum([break_item.valuations[i] for i in ids_of_remaining_bins]) / break_item.cost
                      )
        if upper_bound + sum_profit <= lower_bound
            push!(items_to_remove, item)
        end
    end
    return items_to_remove
end

# TODO what do we do with sum_profits and individual vals?
function pisinger_r2_reduction_no_sort_individual_valuations(bins::Vector{Knapsack}, items::Vector{Item}, sum_profit, lower_bound)
    # SMKP
    aggregate_knapsack = Knapsack(0, [], sum((bin) -> bin.capacity, bins; init=0), 0)

    # Find break item
    break_item = Nothing
    break_item_idx = -1
    profits_of_efficient_bundle = 0
    weights_of_efficient_bundle = 0
    for (idx, item) in enumerate(items)
        weights_of_efficient_bundle += item.cost
        profits_of_efficient_bundle += maximum(item.valuations)

        if weights_of_efficient_bundle >= aggregate_knapsack.capacity
            break_item = item
            break_item_idx = idx
            weights_of_efficient_bundle -= item.cost
            profits_of_efficient_bundle -= maximum(item.valuations)
            break
        end
    end

    # If all items fit in the combined knapsack none can be removed
    if break_item_idx == -1
        return Item[]
    end

    items_to_remove = Item[]

    for item in items[break_item_idx:length(items)]
        upper_bound = profits_of_efficient_bundle + maximum(item.valuations) +
                      floor(
                          (aggregate_knapsack.capacity - weights_of_efficient_bundle - item.cost) *
                          maximum(break_item.valuations) / break_item.cost
                      )
        if upper_bound + sum_profit <= lower_bound
            push!(items_to_remove, item)
        end
    end
    return items_to_remove
end

# Works for individual valuations
# Jørgen Steig criterion 2023 :))
function is_undominated_individual_vals(bin::Knapsack, path::Vector{Item}, remaining_bins::Vector{Knapsack}, excluded_items::Vector{Item})

    ids_of_remaining_bins = map((bin) -> bin.id, remaining_bins)
    assignment_cost = sum((item) -> item.cost, path; init=0)

    subsets = combinations(path)
    for subset in subsets
        subset_cost = sum((item) -> item.cost, subset)
        subset_profit = sum((item) -> item.valuations[bin.id], subset)
        for item in excluded_items
            # If feasible and apparently profitable to swap subset with excluded item
            #println(assignment_cost - subset_cost + item.cost, bin.capacity, item.cost, subset_cost)
            if (assignment_cost - subset_cost + item.cost <= bin.capacity && item.cost >= subset_cost && item.valuations[bin.id] >= subset_profit)

                relevant_valuations = [item.valuations[i] for i in ids_of_remaining_bins]
                # If the value gained for current agent by swapping the subset with the item is greater than
                # the difference in value from giving the item to the person who appreciates it the most
                # and forcing the person who appreciates the subset the least to get the subset
                # then giving the agent the item cannot yield a worse optimal solution, therefore the assignment is dominated
                if (item.valuations[bin.id] + sum((subset_item) -> minimum([subset_item.valuations[i] for i in ids_of_remaining_bins]), subset) >= maximum(relevant_valuations) + subset_profit)
                    #println("Subset: ", subset, " is dominated by ", item, " in assignment ", path, "\nFor bin ", bin.id)
                    #println(item.valuations[bin.id] + sum((subset_item) -> minimum([subset_item.valuations[i] for i in ids_of_remaining_bins]), subset), " vs ", maximum(relevant_valuations) + subset_profit)
                    return false
                end
            end
        end
    end
    return true
end


# Used to generate feasible and undominated assignments
function generate_undominated_bin_assignments(bin::Knapsack, remaining_bins::Vector{Knapsack}, items::Vector{Item}, is_undominated::Function)
    undominated_assignments::Vector{Vector{Item}} = []
    # If the supplied bin is the last bin, we don't need to test it against the dominance criteria for individual valuations
    # We just return every assignment and let the next level in the recursion of search_MKP discard it if not better than best_solution
    # Consider just returning max assignment here instead
    should_check_for_dominance = length(remaining_bins) > 0
    # Manually insert empty set if not last bin
    if should_check_for_dominance
        push!(undominated_assignments, Vector{Item}())
    end
    # Include empty set if other bins can possibly get the items
    function traverse_binary_tree(remaining_capacity, path::Vector{Item}, remaining_items::Vector{Item}, excluded_items::Vector{Item})
        if length(remaining_items) == 0
            if should_check_for_dominance
                if length(path) > 0 && is_undominated(bin, path, remaining_bins, excluded_items)
                    push!(undominated_assignments, path)
                end
            else
                push!(undominated_assignments, path)
            end
            return
        end
        # Don't include item
        traverse_binary_tree(remaining_capacity, path, remaining_items[2:end], vcat(remaining_items[1], excluded_items))
        # If feasible to include the item, do that
        if remaining_capacity - remaining_items[1].cost >= 0
            traverse_binary_tree(remaining_capacity - remaining_items[1].cost, vcat(path, remaining_items[1]), remaining_items[2:end], excluded_items)
        end
    end

    traverse_binary_tree(bin.capacity, Item[], items, Item[])
    return undominated_assignments
end


# Make new version with up to k combinations by sorting them and finding the combination with most items
function generate_all_feasible_maximal_bin_assignments(bin::Knapsack, remaining_bins::Vector{Knapsack}, items::Vector{Item}, is_undominated::Function)
    all_combinations = combinations(items)
    maximal_assignments = []
    for comb in all_combinations
        unallocated_items = filter(item -> !issubset([item], comb), items)
        current_weight = sum(item -> item.cost, comb)
        # If the assignment is infeasible
        if (current_weight > bin.capacity)
            continue
        end
        # If the assignment is maximal (contains as many items as possible)
        assignment_is_maximal = true
        for item in unallocated_items
            if (current_weight + item.cost <= bin.capacity)
                assignment_is_maximal = false
                break
            end
        end
        if (assignment_is_maximal)
            push!(maximal_assignments, comb)
        end
    end
    if length(maximal_assignments) == 0
        push!(maximal_assignments, Item[])
    end
    return maximal_assignments
end

# Generates all feasible, maximal bin assignments
# Sorts items and finds the maximum number of items that an assignment can contain, k
# Finds all combinations up to k items and removed non-maximal ones
# ONLY APPLICABLE TO IDENTICAL VALUATIONS WITHOUT FAIRNESS CRITERIA
function generate_all_feasible_maximal_bin_assignments_using_up_to_k_combs(bin::Knapsack, remaining_bins::Vector{Knapsack}, items::Vector{Item}, is_undominated::Function)
    # Sort items by increasing cost
    sorted_items = sort(items, by=item -> item.cost)
    # Greedily sum items until we hit capacity, the number of items is the max we need to calculate combinations for.
    sum_of_cost = 0
    n_items = 0
    for item in sorted_items
        if (sum_of_cost + item.cost > bin.capacity)
            break
        else
            sum_of_cost += item.cost
            n_items += 1
        end
    end
    all_combinations = []
    for i in range(1, n_items)
        combinations_of_k_items = combinations(items, i)
        for comb in combinations_of_k_items
            push!(all_combinations, comb)
        end
    end
    maximal_assignments = []
    for comb in all_combinations
        unallocated_items = filter(item -> !issubset([item], comb), items)
        current_weight = sum(item -> item.cost, comb)
        # If the assignment is infeasible
        if (current_weight > bin.capacity)
            continue
        end
        # If the assignment is maximal (contains as many items as possible)
        assignment_is_maximal = true
        for item in unallocated_items
            if (current_weight + item.cost <= bin.capacity)
                assignment_is_maximal = false
                break
            end
        end
        if (assignment_is_maximal)
            push!(maximal_assignments, comb)
        end
    end
    if length(maximal_assignments) == 0
        push!(maximal_assignments, Item[])
    end
    #println(maximal_assignments)
    return maximal_assignments
end

# Suitable for cases with individual valuations of fairness criteria
function generate_all_feasible_bin_assignments_using_up_to_k_combs(bin::Knapsack, remaining_bins::Vector{Knapsack}, items::Vector{Item}, is_undominated::Function)
    # Sort items by increasing cost
    sorted_items = sort(items, by=item -> item.cost)
    # Greedily sum items until we hit capacity, the number of items is the max we need to calculate combinations for.
    sum_of_cost = 0
    n_items = 0
    for item in sorted_items
        if (sum_of_cost + item.cost > bin.capacity)
            break
        else
            sum_of_cost += item.cost
            n_items += 1
        end
    end
    all_combinations = []
    for i in range(1, n_items)
        combinations_of_k_items = combinations(items, i)
        for comb in combinations_of_k_items
            push!(all_combinations, comb)
        end
    end
    # Add the empty set since we need that for individual valuations to work
    feasible_assignments = [Vector{Item}()]
    for comb in all_combinations
        current_weight = sum(item -> item.cost, comb; init=0)
        # If the assignment is infeasible
        if (current_weight > bin.capacity)
            continue
        end
        push!(feasible_assignments, comb)
    end
    return feasible_assignments
end

# Individual valuations
function generate_all_feasible_combinations_bin_assignments(bin::Knapsack, remaining_bins::Vector{Knapsack}, items::Vector{Item}, is_undominated::Function)
    all_combinations = combinations(items)
    feasible_assignments = [Vector{Item}()]
    for comb in all_combinations
        current_weight = sum(item -> item.cost, comb)
        # If the assignment is infeasible skip
        if (current_weight > bin.capacity)
            continue
        end
        push!(feasible_assignments, comb)
    end
    return feasible_assignments
end

# Individual valuations
function no_collect_generate_all_feasible_combinations_bin_assignments(bin::Knapsack, remaining_bins::Vector{Knapsack}, items::Vector{Item}, is_undominated::Function)
    all_combinations = combinations(items)
    feasible_assignments = [Vector{Item}()]
    for comb in all_combinations
        current_weight = sum(item -> item.cost, comb)
        # If the assignment is infeasible skip
        if (current_weight > bin.capacity)
            continue
        end
        push!(feasible_assignments, comb)
    end
    return feasible_assignments
end

global best_profit = 0

function search_MKP(bins::Vector{Knapsack}, all_items::Vector{Item}, is_individual_vals::Bool, sum_profit::Int, reductions::Vector, compute_upper_bound::Function, validate_upper_bound::Bool, choose_bin::Function, value_ordering_heuristic::Function, reverse_value_ordering_heuristic::Bool, generate_assignments::Function, is_undominated::Function)::MKP_return
    #println("Subproblem called with\n", bins, "\n", all_items, "\nProfit:", sum_profit, "\nBest profit:", best_profit)
    if (length(bins) == 0 || length(all_items) == 0)
        if (sum_profit > best_profit)
            global best_profit = sum_profit
            #println("Set new best: ", sum_profit, ", ", all_items)
        end
        return MKP_return(sum_profit, bins, 1)
    end

    all_reduced_items::Vector{Item} = []
    for reduction in reductions
        reduced_items = reduction(bins, all_items, sum_profit, best_profit)
        unique_removed_items = setdiff(reduced_items, all_reduced_items)
        all_reduced_items = vcat(all_reduced_items, unique_removed_items)
    end

    if length(all_reduced_items) > 0
        #println("Reduced items:\n", all_reduced_items, "\n from instance with bins:\n", bins, "\nAnd items: ", all_items)
        #println("Reduced ", length(all_reduced_items), " with ", length(bins), " knapsacks left")
        return search_MKP(bins, setdiff(all_items, all_reduced_items), is_individual_vals, sum_profit, reductions, compute_upper_bound, validate_upper_bound, choose_bin, value_ordering_heuristic, reverse_value_ordering_heuristic, generate_assignments, is_undominated)
    end

    upper_bound = compute_upper_bound(bins, all_items)

    if (sum_profit + upper_bound <= best_profit)
        #println("Pruned ", bins, " and ", all_items, " sum ", sum_profit, " upper bound", upper_bound, ", best_profit ", best_profit)
        return MKP_return(-1, [], 1) # Prune because this branch cannot possibly acheive the value of best_profit
    end
    if validate_upper_bound
        lower_bound = bound_and_bound(bins, all_items)
        if lower_bound[1] == upper_bound
            #println("LOWER BOUND MATCHED UPPER, FOUND BEST SOLUTION", upper_bound, " with profit ", sum_profit, " vs ", lower_bound)
            if sum_profit + lower_bound[1] > best_profit
                global best_profit = lower_bound[1] + sum_profit
            end
            return MKP_return(lower_bound[1] + sum_profit, lower_bound[2], 1)
        end
    end
    bin = choose_bin(bins, all_items)
    #println("Selected ", bin, " and got bound ", upper_bound)
    # We evaluated 11 different value ordering heuristics and found that the best performing heuristic overall was the min-cardinality-max-profit ordering, where candidate bin assignments are sorted in order of non-decreasing cardinality and ties are broken according to non-increasing order of profit
    remaining_bins = setdiff(bins, [bin])
    assignments = sort(generate_assignments(bin, remaining_bins, all_items, is_undominated), lt=(a, b) -> value_ordering_heuristic(a, b), rev=reverse_value_ordering_heuristic)
    best_assignment = MKP_return(-1, [], 0)
    #println("Assignments:")
    #display(assignments)
    sum_nodes_explored = 1
    for assignment in assignments
        bin.items = assignment
        remaining_items = setdiff(all_items, assignment)
        #println("Gave knapsack ", bin.id, " the assignment ", assignment)
        #println("Items: ", all_items, "\nItems to remove: ", assignment, "\nRemaining items: ", remaining_items)
        #println("Bin: ", bin, "\n\nBins: ", bins, "\n\nremaining: ", remaining_bins)
        subproblem = search_MKP(remaining_bins, remaining_items, is_individual_vals, sum_profit + sum((item) -> item.valuations[is_individual_vals ? bin.id : 1], assignment; init=0), reductions, compute_upper_bound, validate_upper_bound, choose_bin, value_ordering_heuristic, reverse_value_ordering_heuristic, generate_assignments, is_undominated)
        sum_nodes_explored += subproblem.nodes_explored
        #println("Subproblem:\n", subproblem)
        if (subproblem.best_profit > best_assignment.best_profit)
            best_assignment = MKP_return(subproblem.best_profit, subproblem.best_assignment + copy(bin), 0)
            #println("Set best assignment to: ", best_assignment.best_profit)
        end
        bin.items = []
    end
    best_assignment.nodes_explored = sum_nodes_explored

    return best_assignment
end

function solve_multiple_knapsack_problem(bins::Vector{Knapsack}, items::Vector{Item}, options::MulKnapOptions, print_solution_after::Bool)::MKP_return
    if (options.individual_vals)
        items = sort(items, lt=(a, b) -> is_smaller_max_profit_divided_by_weight(a, b, [bin.id for bin in bins]), rev=true)
    else
        items = sort(items, lt=(a, b) -> is_smaller_profit_divided_by_weight(a, b), rev=true)
    end

    if (options.preprocess)
        remove_infeasible_knapsacks!(bins, items)
        remove_infeasible_items!(bins, items)
    end

    global best_profit = 0
    solution = search_MKP(bins, items, options.individual_vals, 0, options.reductions, options.compute_upper_bound, options.validate_upper_bound, options.choose_bin, options.value_ordering_heuristic, options.reverse_value_ordering_heuristic, options.generate_assignments, options.is_undominated)

    if (print_solution_after)
        print_solution(solution, options.individual_vals)
    end
    return solution
end



#=
search MKP(bins, items, sumProfit) 

    if bins==∅ or items == ∅ 
        if sumProfit > bestProfit then bestProfit = sumProfit; return 
    ri = reduce(bins,items) /* Pisinger’s R2 reduction */ 
    if ri 6= ∅ 
        search MKP(bins, items \ ri, sumProfit) 
        return 
    upperBound = compute upper bound(items,bins) 
    if (sumProfit + upperBound ≤ bestProfit 
        return /* upper-bound based pruning using SMKP bound */ 
    if (validate upper bound(upperBound))
        bestProfit = upperbound
        return /* bound-and-bound */ 
    bin = choose bin(bins) 
    undominatedAssignments = generate undominated(items,capacity(bin)) 
    foreach A ∈ sort assignments(undominatedAssignments) 
    if not(symmetric(A)) 
        assign A to bin 
        search MKP(bins \ bin, items \ A, sumProfit+P j∈A pj )
=#






#=
### BASIC BRUTE FORCE

# Each node is a possible item in a knapsack
function use_brute_force!(items::Vector{Item}, bins::Vector{Knapsack})

    function dive_deeper!(items::Vector{Item}, bins::Vector{Knapsack})
        total_value = 0
        for bin in bins
            for item in bin.items
                total_value += item.valuations[1]
            end
        end
        println("Total value of ", bins, " is ", total_value)

        if (length(items) == 0)
            return (bins, total_value)

        else
            max_val = (bins, total_value)
            init_bins = copy(bins)
            init_items = copy(items)
            bins = deepcopy(bins)
            # check for whether bins are filled or no more items fit, then return the current val instead of max_val
            println("Top ", bins, items)
            for bin in bins
                if (bin.load + init_items[1].cost <= bin.capacity)
                    # Assign item to bin
                    items = copy(init_items)
                    println("123 ", items)
                    push!(bin.items, items[1])
                    bin.load += items[1].cost
                    deleteat!(items, 1)
                    println("Bin after push ", bin)
                    # Branch further
                    branch_val = dive_deeper!(items, bins)
                    if (branch_val[2] > max_val[2])
                        max_val = branch_val
                    end
                end
            end
            # Check for the case where we discard items
            if (length(init_items) > 1)
                println("Init bins is 3  ", init_bins)
                println("Testing with removing: ", init_items[1], " from ", init_items)
                deleteat!(init_items, 1)
                branch_val = dive_deeper!(init_items, init_bins)
                println("Got branch val", branch_val)
                if (branch_val[2] > max_val[2])
                    max_val = branch_val
                end
            end

            return max_val
        end
    end
    return dive_deeper!(items, bins)
end

#test_items = [Item(1, 4, [5]), Item(2, 9, [11]), Item(3, 6, [5])]
#test_bins = [Knapsack(1, [], 10, 0)]

test_items = [Item(4, 2, [1]), Item(1, 4, [5]), Item(2, 9, [14]), Item(3, 6, [5])]
test_bins = [Knapsack(1, [], 10, 0), Knapsack(2, [], 10, 0)]
#println(use_brute_force!(items_sorted_by_efficiency, knapsacks))
#println(use_brute_force!(test_items, test_bins))
=#