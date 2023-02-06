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
    println("optimal", optimal_solution_index)
    solution = get_binary_knapsack_solution(optimal_solution_index[1], optimal_solution_index[2])
    println("Solution: ", solution)
    =#
    return profits[length(items_copy), bin.capacity+1]
end

# solve 0-1 knapsack problem on the aggregate knapsack and return the value
function compute_upper_bound(bins::Vector{Knapsack}, items::Vector{Item})
    aggregate_knapsack = Knapsack(0, [], sum((bin) -> bin.capacity, bins), 0)
    return solve_binary_knapsack(aggregate_knapsack, items)
end

# Return the upper bound considering individual valuations
function compute_upper_bound_individual_vals(bins::Vector{Knapsack}, items::Vector{Item})
    for item in items

    end
end


# Tests whether the assignment is dominated by making sure x-s > r-t
# Where x is 
function is_assignment_dominated(included, excluded, residual_capacity)
    # The residual capacity r of a bin is the bin capacity c minus the largest number in the bin, 
    # which is common to all feasible completions of a given bin.

    #sum_of_weights = sum_item_cost(included \ ) # t in the paper

end

# Path symmetry
function is_assignment_symmetric()
    # If a sibling of an ancestor of current node has already been expanded, that sibling is a nogood with respect to current node.
    # Since DFS, a nogood is a node who's descendants have been exhaustively searched already. 
    # The union of these nogoods is a list fo the entire part of the search tree we've searched.
    # Can prune based if symmetric 

end

# R. Korf 2003 - An improved Algorithm for Optimal Bin packing
# Used to generate undominated sets
# Q: What does an undominated set mean in the context of multiple knapsack as opposed to bin packing?
# Current understanding: Still means the same thing, we deal with the value/profit later.
# But maybe we could include some notion of dominance based on profit as well?
function feasible(included, excluded, remaining::Vector{Item}, lower_bound, upper_bound)
    if (length(remaining) == 0 || upper_bound == 0)
        if (is_assignment_dominated(included, excluded, residual_capacity))
            return
        else
            return (included, excluded)
        end
    else
        max = find_heaviest_item(remaining)
        if (max[2] > upper_bound)
            remaining_copy = copy(remaining)
            deleteat!(remaining_copy, max[1])
            feasible(included, excluded, remaining_copy, lower_bound, upper_bound)
        elseif (max[2] == upper_bound)
            included_copy = copy(included)
            push!(included_copy, remaining[max[1]])
            remaining_copy = copy(remaining)
            deleteat!(remaining_copy, max[1])
            feasible(included_copy, excluded, remaining_copy, lower_bound - max[2], upper_bound - max[2])
        else
            included_copy = copy(included)
            push!(included_copy, remaining[max[1]])
            excluded_copy = copy(excluded)
            push!(excluded_copy, remaining[max[1]])
            remaining_copy = copy(remaining)
            deleteat!(remaining_copy, max[1])

            feasible(included_copy, excluded, remaining_copy, lower_bound - max[2], upper_bound - max[2])
            feasible(included, excluded_copy, remaining_copy, max(lower_bound, lower_bound + max[2]), upper_bound)
        end
    end
end

function generate_undominated_bin_assignments(bin::Knapsack, items::Vector{Item})
    #max = find_max_item(items)
    #feasible([], [], items, max[2] + 1, bin.capacity)

end

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

global best_profit = 1

function search_MKP(bins::Vector{Knapsack}, items::Vector{Item}, sum_profit::Int)
    upper_bound = 0
    if (length(bins) == 0 || length(items) == 0)
        if (sum_profit > best_profit)
            global best_profit = sum_profit
            #println("Set new best: ", sum_profit, ", ", items)
            return (sum_profit, bins)
        end
        #println("Didn't ", sum_profit, " vs ", best_profit)
    else
        upper_bound = compute_upper_bound(bins, items)
    end

    if (sum_profit + upper_bound <= best_profit)
        #println("Pruned ", bins, " and ", items, " sum ", sum_profit, " best ", best_profit)
        return (-1, []) # Prune because this branch cannot possibly acheive the value of best_profit
    end
    # upper bound validation - TODO
    (bin, bin_index) = get_bin_with_least_remaining_capacity(bins)
    #println("Selected ", bin, bin_index, " bound ", upper_bound)
    assignments = sort(generate_all_feasible_maximal_bin_assignments(bin, items), rev=true)
    best_assignment = (-1, [])
    #println("Assignments: ", assignments)
    for assignment in assignments
        #if not symmetric - missing pruning step here, TODO
        bins[bin_index].items = assignment
        #println("Gave knapsack ", bins[bin_index].id, " == ", bin_index, " the assignment ", assignment)
        items_copy = deepcopy(items)
        items_copy \ assignment
        #println("Items: ", items, "\nItems to remove: ", assignment, "\nRemaining items: ", items_copy)
        bins_copy = deepcopy(bins)
        bins_copy \ bin
        subproblem = search_MKP(bins_copy, items_copy, sum_profit + sum((item) -> item.valuations[1], assignment))
        if (subproblem[1] > best_assignment[1])
            best_assignment = (subproblem[1], subproblem[2] + deepcopy(bins[bin_index]))
            #println("Set best assignment to: ", best_assignment)
        end
    end
    if (length(best_assignment[2]) > 0)
        items \ last(best_assignment[2]).items
    end
    return best_assignment
end

function solve_multiple_knapsack_problem(bins::Vector{Knapsack}, items::Vector{Item}, sort_items_by_efficiency::Bool, preprocess_items_and_bins::Bool, print_solution_after::Bool)
    if (sort_items_by_efficiency)
        items = sort(items, by=item -> item.valuations[1] / item.cost, rev=true)
    end
    if (preprocess_items_and_bins)
        remove_infeasible_knapsacks!(bins, items)
        remove_infeasible_items!(bins, items)
    end

    global best_profit = 0
    solution = search_MKP(bins, items, 0)

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