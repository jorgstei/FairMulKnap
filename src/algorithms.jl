
#=
    The choose_bin function described in "A branch and bound algorithm for hard multiple knapsack problems" - Fukunaga
=#
function get_bin_with_least_remaining_capacity(bins::Vector{Knapsack})
    smallest_remaining_bin = (bins[1], 1)
    smallest_remaining_capacity = bins[1].capacity
    for (idx, bin) in enumerate(bins)
        remaining_capacity = bin.capacity - sum((item) -> item.cost, bin.items; init=0)
        if (remaining_capacity < smallest_remaining_capacity)
            smallest_remaining_bin = (bins[idx], idx)
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
    ids_of_remaining_bins = map((bin) -> bin.id, remaining_bins)
    for item in items
        relevant_valuations = [item.valuations[i] for i in ids_of_remaining_bins]
        itemcpy = Item(item.id, item.cost, fill(maximum(relevant_valuations), length(item.valuations)))
        push!(adjusted_items, itemcpy)
    end
    return solve_binary_knapsack(aggregate_knapsack, adjusted_items)
end

# The program should keep the items sorted throughout so long as the items were sorted by profit/weight initially.
# This happens automatically in solve_multiple_knapsack_problem
# TODO what do we do with sum_profits and individual vals?
function pisinger_r2_reduction(bins::Vector{Knapsack}, sorted_items::Vector{Item}, lower_bound)
    # SMKP
    aggregate_knapsack = Knapsack(0, [], sum((bin) -> bin.capacity, bins; init=0), 0)
    # Find break item
    break_item = Nothing
    break_item_idx = -1
    total_sum = 0
    for (idx, item) in enumerate(sorted_items)
        total_sum += item.cost
        if total_sum > aggregate_knapsack.capacity
            break_item = item
            break_item_idx = idx
            break
        end
    end

    # If all items fit in the combined knapsack none can be removed
    if break_item_idx == -1
        return Item[]
    end

    sum_profits = 0
    sum_weights = 0
    for item in sorted_items[1:break_item_idx]
        sum_profits += item.valuations[1]
        sum_weights += item.cost
    end

    items_to_remove = Item[]

    for item in sorted_items[break_item_idx:length(sorted_items)]
        upper_bound = sum_profits + item.valuations[1] +
                      floor(
                          (aggregate_knapsack.capacity - sum_weights - item.cost) *
                          break_item.valuations[1] / break_item.cost
                      )
        if upper_bound <= lower_bound
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
function is_undominated_individual_vals(bin::Knapsack, remaining_bins::Knapsack, path::Vector{Item}, excluded_items::Vector{Item})
    assignment_cost = sum((item) -> item.cost, path; init=0)
    subsets = collect(combinations(path))
    for item in excluded_items
        for subset in subsets
            subset_cost = sum((item) -> item.cost, subset)
            subset_profit = sum((item) -> item.valuations[bin.id], subset)
            # If feasible to swap subset with excluded item
            if (assignment_cost - subset_cost + item.cost <= bin.capacity && item.cost >= subset_cost)
                ids_of_remaining_bins = map((bin) -> bin.id, remaining_bins)
                relevant_valuations = [item.valuations[i] for i in ids_of_remaining_bins]
                # If the value gained for current agent by swapping the subset with the item is greater than
                # the difference in value from giving the item to the person who appreciates it the most
                # and forcing the person who appreciates the subset the least to get the subset
                # then giving the agent the item cannot yield a worse optimal solution, therefore the assignment is dominated
                if (item.valuations[bin.id] - subset_profit >= max(relevant_valuations) - sum((item) -> minimum(relevant_valuations), subset))
                    return false
                end
            end
        end
    end
    return true
end


# Used to generate feasible and undominated assignments
# Add empty set somehow TODO
function generate_undominated_bin_assignments(bin::Knapsack, items::Vector{Item}, is_undominated::Function)
    undominated_assignments::Vector{Vector{Item}} = []

    function traverse_binary_tree(remaining_capacity, path::Vector{Item}, remaining_items::Vector{Item}, excluded_items::Vector{Item})
        if length(remaining_items) == 0
            if length(path) > 0 && is_undominated(path, excluded_items)
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

#


# Make new version with up to k combinations by sorting them and finding the combination with most items
function generate_all_feasible_maximal_bin_assignments(bin::Knapsack, items::Vector{Item})
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
function generate_all_feasible_maximal_bin_assignments_using_up_to_k_combs(bin::Knapsack, items::Vector{Item})
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
function generate_all_feasible_bin_assignments_using_up_to_k_combs(bin::Knapsack, items::Vector{Item})
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

function search_MKP(bins::Vector{Knapsack}, all_items::Vector{Item}, sum_profit::Int, reductions::Vector, generate_assignments::Function, items_diff::Bool)
    println("Subproblem called with\n", bins, "\n", all_items, "\nProfit:", sum_profit)
    if (length(bins) == 0 || length(all_items) == 0)
        if (sum_profit > best_profit)
            global best_profit = sum_profit
            println("Set new best: ", sum_profit, ", ", all_items)
        end
        return (sum_profit, bins)
    end

    all_reduced_items::Vector{Item} = []
    for reduction in reductions
        reduced_items = reduction(bins, all_items, best_profit)
        unique_removed_items = setdiff(reduced_items, all_reduced_items)
        all_reduced_items = vcat(all_reduced_items, unique_removed_items)
    end

    if length(all_reduced_items) > 0
        #println("Current items:\n", items)
        #println("Reduced items:\n", reduced_items, "\n from instance with bins:\n", bins)
        return search_MKP(bins, setdiff(all_items, all_reduced_items), sum_profit, reductions, generate_assignments, items_diff)
    end

    upper_bound = compute_max_upper_bound_individual_vals(bins, all_items)

    if (sum_profit + upper_bound <= best_profit)
        #println("Pruned ", bins, " and ", items, " sum ", sum_profit, " best ", best_profit)
        return (-1, []) # Prune because this branch cannot possibly acheive the value of best_profit
    end
    # upper bound validation - TODO
    (bin, bin_index) = get_bin_with_least_remaining_capacity(bins)
    println("Selected ", bin, "and got bound ", upper_bound)
    #We evaluated 11 different value ordering heuristics and found that the best performing heuristic overall was the min-cardinality-max-profit ordering, where candidate bin assignments are sorted in order of non-decreasing cardinality and ties are broken according to non-increasing order of profit
    assignments = sort(generate_assignments(bin, all_items), rev=true)
    best_assignment = (-1, [])
    bin_copy = copy(bin)
    println("Assignments: ", assignments)
    for assignment in assignments
        bin_copy.items = assignment
        #println("Gave knapsack ", bins[bin_index].id, " == ", bin_index, " the assignment ", assignment)
        remaining_items = setdiff(all_items, assignment)
        #println("Items: ", all_items, "\nItems to remove: ", assignment, "\nRemaining items: ", remaining_items)
        remaining_bins = setdiff(bins, [bin])
        #println("Bin: ", bin_copy, "\n\nBins: ", bins, "\n\nremaining: ", remaining_bins)
        subproblem = search_MKP(remaining_bins, remaining_items, sum_profit + sum((item) -> item.valuations[bin.id], assignment; init=0), reductions, generate_assignments, items_diff)
        #println("Subproblem:\n", subproblem)
        if (subproblem[1] > best_assignment[1])
            best_assignment = (subproblem[1], subproblem[2] + copy(bin_copy))
            #println("Set best assignment to: ", best_assignment)
        end
    end
    return best_assignment
end

function solve_multiple_knapsack_problem(bins::Vector{Knapsack}, items::Vector{Item}, preprocess_items_and_bins::Bool, print_solution_after::Bool, reductions::Vector, generate_assignments::Function, items_diff::Bool)
    # TODO maximum(valuations) ? 
    items = sort(items, by=item -> item.valuations[1] / item.cost, rev=true)

    if (preprocess_items_and_bins)
        remove_infeasible_knapsacks!(bins, items)
        remove_infeasible_items!(bins, items)
    end

    global best_profit = 0
    solution = search_MKP(bins, items, 0, reductions, generate_assignments, items_diff)

    if (print_solution_after)
        print_solution(solution)
    end
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