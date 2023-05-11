
#=
    The choose_bin function described in "A branch and bound algorithm for hard multiple knapsack problems" - Fukunaga
=#
function get_bin_with_least_remaining_capacity(bins::Vector{Knapsack})
    smallest_remaining_bin = bins[1]
    smallest_remaining_capacity = bins[1].capacity
    for bin in bins
        remaining_capacity = bin.capacity - sum((item) -> item.cost, bin.items; init=0)
        if (remaining_capacity < smallest_remaining_capacity)
            smallest_remaining_bin = bin
            smallest_remaining_capacity = remaining_capacity
        end
    end
    return smallest_remaining_bin
end

#=
    Dynamic programming solution to the 0-1 knapsack problem
    Used to compute the upper bound in MKP
    Inspired by: https://en.wikipedia.org/wiki/Knapsack_problem#Solving
=#
function solve_binary_knapsack(bin::Knapsack, items::Vector{Item})
    # Add filler item for the algorithm to work properly
    items_copy = deepcopy(items)
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
                    profits[i, j] = max(profits[i-1, j], profits[i-1, j-items_copy[i].cost] + items_copy[i].valuations[1])
                else
                    profits[i, j] = profits[i-1, j]
                end
            end
        end
    end
    # Get profit of full problem
    get_profit!(length(items_copy), bin.capacity + 1)

    # For printing the table and actual solution instead of just the value.

    #println("Profits:")
    #display(profits)
    #=
    optimal_solution_index = findmax(profits)[2]
    function get_binary_knapsack_solution(i, j)
        if (i == 1)
            return []
        end
        if (profits[i, j] > profits[i-1, j])
            push!(get_binary_knapsack_solution(i - 1, j - items_copy[i].cost), items_copy[i])
        else
            return get_binary_knapsack_solution(i - 1, j)
        end
    end
    #println("optimal", optimal_solution_index)
    solution = get_binary_knapsack_solution(optimal_solution_index[1], optimal_solution_index[2])
    println("Solution: ", solution)
    =#

    return profits[length(items_copy), bin.capacity+1]
end

# solve 0-1 knapsack problem on the aggregate knapsack and return the value
# Works for items with a single valuation
function compute_upper_bound(bins::Vector{Knapsack}, items::Vector{Item})
    aggregate_knapsack = Knapsack(0, [], sum((bin) -> bin.capacity, bins; init=0), 0)
    return solve_binary_knapsack(aggregate_knapsack, items)
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
    return solve_binary_knapsack(aggregate_knapsack, adjusted_items)
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



# Works for MKP with identical valuations
# MKP dominance criterion Fukunaga & Korf 2007
# MAPPED ONE TO ONE :DDDDDDDDDDDDDDDDD TODO
function is_undominated_identical_vals(path::Vector{Item}, excluded_items::Vector{Item})
    assignment_cost = sum((item) -> item.cost, path)
    subsets = collect(combinations(path))
    for item in excluded_items
        for subset in subsets
            subset_cost = sum((item) -> item.cost, subset)
            subset_profit = sum((item) -> item.valuations[bin.id], subset)
            # If feasible to swap subset with excluded item
            if (assignment_cost - subset_cost + item.cost <= bin.capacity)
                # If profitable to swap subset with excluded item
                if (item.valuations[bin.id] >= subset_profit)
                    # Then the assignment is dominated
                    return false
                end
            end
        end
    end
    return true
end

# Works for individual valuations
# Jørgen Steig criterion 2023 :)) 
function is_undominated_individual_vals(bin::Knapsack, path::Vector{Item}, remaining_bins::Vector{Knapsack}, excluded_items::Vector{Item})

    ids_of_remaining_bins = map((bin) -> bin.id, remaining_bins)
    assignment_cost = sum((item) -> item.cost, path; init=0)

    subsets = collect(combinations(path))
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
                if (item.valuations[bin.id] - subset_profit >= maximum(relevant_valuations) - sum((subset_item) -> minimum([subset_item.valuations[i] for i in ids_of_remaining_bins]), subset))
                    #println("Vals: ", ids_of_remaining_bins, ", ", relevant_valuations)
                    #println("Subset: ", subset, " is dominated by ", item, " in assignment ", path, "\nFor bin ", bin.id)
                    #println(item.valuations[bin.id] - subset_profit, " vs ", maximum(relevant_valuations) - sum((subset_item) -> minimum([subset_item.valuations[i] for i in ids_of_remaining_bins]), subset))
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
    all_combinations = collect(combinations(items))
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
    #println(maximal_assignments)
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
        combinations_of_k_items = collect(combinations(items, i))
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
        combinations_of_k_items = collect(combinations(items, i))
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

global best_profit = 1

function search_MKP(bins::Vector{Knapsack}, all_items::Vector{Item}, sum_profit::Int, reductions::Vector, compute_upper_bound::Function, generate_assignments::Function, is_undominated::Function)::MKP_return
    #println("Subproblem called with\n", bins, "\n", all_items, "\nProfit:", sum_profit)
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
        return search_MKP(bins, setdiff(all_items, all_reduced_items), sum_profit, reductions, compute_upper_bound, generate_assignments, is_undominated)
    end

    upper_bound = compute_upper_bound(bins, all_items)

    if (sum_profit + upper_bound <= best_profit)
        #println("Pruned ", bins, " and ", all_items, " sum ", sum_profit, " upper bound", upper_bound, ", best_profit ", best_profit)
        return MKP_return(-1, [], 1) # Prune because this branch cannot possibly acheive the value of best_profit
    end
    # upper bound validation - TODO
    bin = get_bin_with_least_remaining_capacity(bins)
    #println("Selected ", bin, " and got bound ", upper_bound)
    # We evaluated 11 different value ordering heuristics and found that the best performing heuristic overall was the min-cardinality-max-profit ordering, where candidate bin assignments are sorted in order of non-decreasing cardinality and ties are broken according to non-increasing order of profit
    remaining_bins = setdiff(bins, [bin])
    assignments = sort(generate_assignments(bin, remaining_bins, all_items, is_undominated), rev=true)
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
        subproblem = search_MKP(remaining_bins, remaining_items, sum_profit + sum((item) -> item.valuations[bin.id], assignment; init=0), reductions, compute_upper_bound, generate_assignments, is_undominated)
        sum_nodes_explored += subproblem.nodes_explored
        #println("Subproblem:\n", subproblem)
        if (subproblem.best_profit > best_assignment.best_profit)
            best_assignment = MKP_return(subproblem.best_profit, subproblem.best_assignment + copy(bin), 0)
            #println("Set best assignment to: ", best_assignment)
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
    solution = search_MKP(bins, items, 0, options.reductions, options.compute_upper_bound, options.generate_assignments, options.is_undominated)

    if (print_solution_after)
        print_solution(solution)
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