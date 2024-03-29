# Benchmarking and testing
#=
thief_items = [Item(1, 2, [3]), Item(2, 3, [4]), Item(3, 4, [5]), Item(4, 5, [6])]
thief_knap = [Knapsack(1, [], 5, 0), Knapsack(2, [], 4, 0)]
simple_example_benchmark = @benchmark solve_multiple_knapsack_problem(thief_knap, thief_items, false, false)
show(stdout, MIME("text/plain"), simple_example_benchmark)
=#
#=
testing_items = [Item(1, 4, [2]), Item(2, 7, [5]), Item(3, 2, [6]), Item(4, 4, [7]), Item(5, 3, [1]), Item(6, 2, [3])]
testing_knap = [Knapsack(1, [], 5, 0), Knapsack(2, [], 4, 0), Knapsack(3, [], 7, 0)]
=#
# Example from google: https://developers.google.com/optimization/bin/multiple_knapsack
item_weights = [48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36]
item_vals = [10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25]
test_items = convert_from_weight_and_value_array_to_item_list(item_weights, item_vals, 5)
test_knaps = [Knapsack(1, [], 100, 0), Knapsack(2, [], 100, 0), Knapsack(3, [], 100, 0), Knapsack(4, [], 100, 0), Knapsack(5, [], 100, 0)]
#bin_completion_benchmark = @benchmark solve_multiple_knapsack_problem(test_knaps, test_items, false, false)
#show(stdout, MIME("text/plain"), bin_completion_benchmark)
#bin_completion_benchmark_nosort = @benchmark solve_multiple_knapsack_problem(test_knaps, test_items,  false, true)
#show(stdout, MIME("text/plain"), bin_completion_benchmark_nosort)
#solve_multiple_knapsack_problem(test_knaps, test_items, true, true, [pisinger_r2_reduction], generate_all_feasible_bin_assignments_using_up_to_k_combs)

# Individual values 
indiv_knaps = [Knapsack(1, [], 50, 0), Knapsack(2, [], 30, 0)]
indiv_items = [Item(1, 10, [1, 5]), Item(2, 20, [2, 14]), Item(3, 40, [4, 8]), Item(4, 9, [5, 1]), Item(5, 9, [6, 2])]
#=
test_bench = @benchmark generate_all_feasible_bin_assignments_using_up_to_k_combs(copy($indiv_knaps[1]), copy($indiv_items))
new_bench = @benchmark generate_undominated_bin_assignments(copy($indiv_knaps[1]), copy($indiv_items))
println("Res: ", test_bench, " vs ", new_bench)
println("K combs:\n", generate_all_feasible_bin_assignments_using_up_to_k_combs(indiv_knaps[1], indiv_items), "\n\n", length(generate_all_feasible_bin_assignments_using_up_to_k_combs(indiv_knaps[1], indiv_items)))
println("\n\nUndominated:\n", generate_undominated_bin_assignments(indiv_knaps[1], indiv_items), "\n\n", length(generate_undominated_bin_assignments(indiv_knaps[1], indiv_items)))
print(@benchmark solve_multiple_knapsack_problem(copy($indiv_knaps), copy($indiv_items), true, false, [pisinger_r2_reduction], generate_all_feasible_bin_assignments_using_up_to_k_combs))
print(@benchmark solve_multiple_knapsack_problem(copy($indiv_knaps), copy($indiv_items), true, false, [pisinger_r2_reduction], generate_undominated_bin_assignments))
=#
#test_bench = @benchmark generate_all_feasible_maximal_bin_assignments(copy($test_knaps[1]), copy($test_items))
#new_bench = @benchmark generate_all_feasible_maximal_bin_assignments_two(copy($test_knaps[1]), copy($test_items))
#println("Res: ", test_bench, " vs ", new_bench)
#=
no_reductions = @benchmark solve_multiple_knapsack_problem(indiv_knaps, indiv_items, true, false, [], generate_all_feasible_maximal_bin_assignments_using_up_to_k_combs)
normal = @benchmark solve_multiple_knapsack_problem(indiv_knaps, indiv_items, true, false, [pisinger_r2_reduction], generate_all_feasible_maximal_bin_assignments_using_up_to_k_combs)
reduce_all_combs = @benchmark solve_multiple_knapsack_problem(indiv_knaps, indiv_items, true, false, [pisinger_r2_reduction], generate_all_feasible_maximal_bin_assignments)
no_reductions_and_generate_all = @benchmark solve_multiple_knapsack_problem(indiv_knaps, indiv_items, true, false, [], generate_all_feasible_maximal_bin_assignments)
println("No reduction:", no_reductions, "\n\nNormal:", normal, "\n\nReduce all combs:", reduce_all_combs, "\n\nNada:", no_reductions_and_generate_all)
=#

mutable struct BenchmarkResults
    solve_method::String
    mean_values::Matrix{Float64}
    nodes_explored::Matrix{Int}
    agent_number_list::Vector{Int}
    items_number_list::Vector{Int}
    options::MulKnapOptions
end



function plot_results(models::Vector{BenchmarkResults}, n_items::Int, n_items_index::Int)
    x = models[1].agent_number_list

    max_time = -1
    for model in models
        model_max = maximum(model.mean_values[n_items_index, :])
        if model_max > max_time
            max_time = model_max
        end
    end

    scaling = 1
    y_label = "Time (ns)"

    if (max_time >= 1e9)
        y_label = "Time (s)"
        scaling = 1e9

    elseif (max_time >= 1e6)
        y_label = "Time (ms)"
        scaling = 1e6

    elseif (max_time >= 1e3)
        y_label = "Time (μs)"
        scaling = 1e3
    end

    plot_title = "Mean times - " * string(n_items) * " items"
    p = plot(x, [res.mean_values[n_items_index, :] / scaling for res in models], xlabel="Agents", ylabel=y_label, title=plot_title, label=reshape(map((result) -> result.solve_method, models), 1, length(models)), linewidth=3)
    savefig(p, "../results/mean_times/mean_" * string(models[1].agent_number_list[1]) * "-" * string(models[1].agent_number_list[end]) * "agents_" * string(n_items) * "items.png")

    nodes_title = "Nodes explored - " * string(n_items) * " items"
    node_plot = plot(x, [res.nodes_explored[n_items_index, :] for res in models], xlabel="Agents", ylabel="Nodes", title=nodes_title, label=reshape(map((result) -> result.solve_method, models), 1, length(models)), linewidth=3)
    savefig(node_plot, "../results/nodes/nodes_" * string(models[1].agent_number_list[1]) * "-" * string(models[1].agent_number_list[end]) * "agents_" * string(n_items) * "items.png")

end

#=
knapsss = Knapsack(1, [], 16, 0)
my_items = [Item(i, i, [i]) for i in range(1, 20)]
k = @benchmark generate_all_feasible_bin_assignments_using_up_to_k_combs($knapsss, Vector{Knapsack}(), $my_items, () -> ())
all = @benchmark generate_all_feasible_combinations_bin_assignments($knapsss, Vector{Knapsack}(), $my_items, () -> ())
no_collect = @benchmark no_collect_generate_all_feasible_combinations_bin_assignments($knapsss, Vector{Knapsack}(), $my_items, () -> ())


show(stdout, MIME("text/plain"), k)
show(stdout, MIME("text/plain"), all)
show(stdout, MIME("text/plain"), no_collect)
exit(0)
=#
# How big the step between n items should be, e.g. 2 makes 1, 3, 5, 7 ...
item_list_step = 2
#=
n_items_list = collect(range(1, 1, step=item_list_step))
# How many knapsacks should be generated tested, (from, to)
n_agents_tuple = (1, 1)
=#
n_items_list = collect(range(60, 60, step=item_list_step))
n_agents_tuple = (10, 10)

function init_items_x_agents_matrix()
    return zeros((length(n_items_list), n_agents_tuple[2] - n_agents_tuple[1] + 1))
end
#display(items_x_agents[1, :])
agent_range = range(n_agents_tuple[1], n_agents_tuple[2])
# Objective valuations
identical = BenchmarkResults("MKP Maximal", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(false, true, [], compute_surrogate_upper_bound, false, get_bin_with_least_capacity, is_smaller_cardinally_with_objective_value_tie_break, false, generate_all_feasible_maximal_bin_assignments_using_up_to_k_combs, () -> ()))
nonmaximal = BenchmarkResults("MKP Nonmaximal", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(false, true, [], compute_surrogate_upper_bound, false, get_bin_with_least_capacity, is_smaller_cardinally_with_objective_value_tie_break, false, generate_all_feasible_bin_assignments_using_up_to_k_combs, () -> ()))
r2 = BenchmarkResults("MKP + R2", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(false, true, [pisinger_r2_reduction], compute_surrogate_upper_bound, false, get_bin_with_least_capacity, is_smaller_cardinally_with_objective_value_tie_break, false, generate_all_feasible_maximal_bin_assignments_using_up_to_k_combs, () -> ()))

# Subjective valuations
all_combs = BenchmarkResults("All combinations", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [], compute_max_upper_bound_individual_vals, false, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_all_feasible_combinations_bin_assignments, () -> ()))
up_to_k = BenchmarkResults("Up-to-k", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [], compute_max_upper_bound_individual_vals, false, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_all_feasible_bin_assignments_using_up_to_k_combs, () -> ()))

always_undominated = BenchmarkResults("Always undominated", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [], compute_max_upper_bound_individual_vals, false, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, always_undominated_fun))
undominated = BenchmarkResults("Normal", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [], compute_max_upper_bound_individual_vals, false, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
undominated_r2 = BenchmarkResults("R2", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, false, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
undominated_r2_no_sort = BenchmarkResults("R2 nosort", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_no_sort_individual_valuations], compute_max_upper_bound_individual_vals, false, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
#undominated_lp_bound = BenchmarkResults("LP-relaxed", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], lp_relaxed_upper_bound, false, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
undominated_bb_r2 = BenchmarkResults("BB + Undom + R2", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))

undominated_bb_r2_min_capacity_ordering = BenchmarkResults("Smallest capacity", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
undominated_bb_r2_max_val_ordering = BenchmarkResults("Max val ordering", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_most_max_valuations, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
undominated_bb_r2_max_val_and_least_capacity_ordering = BenchmarkResults("Smallest c/n_max_vals", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_smallest_capacity_divided_by_n_max_valuations, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
undominated_bb_r2_max_capacity_ordering = BenchmarkResults("Largest capacity", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_most_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))

# Value ordering heuristics
min_cardinality = BenchmarkResults("Min Cardinality", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
max_cardinality = BenchmarkResults("Max Cardinality", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, true, generate_undominated_bin_assignments, is_undominated_individual_vals))
min_weight = BenchmarkResults("Min Weight", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_least_capacity, is_smaller_by_weight, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
max_weight = BenchmarkResults("Max Weight", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_least_capacity, is_smaller_by_weight, true, generate_undominated_bin_assignments, is_undominated_individual_vals))
min_profit = BenchmarkResults("Min Profit", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_least_capacity, is_smaller_by_max_value, false, generate_undominated_bin_assignments, is_undominated_individual_vals))
max_profit = BenchmarkResults("Max Profit", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_least_capacity, is_smaller_by_max_value, true, generate_undominated_bin_assignments, is_undominated_individual_vals))

best_i_got = BenchmarkResults("MOKP", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, true, get_bin_with_least_capacity, is_smaller_cardinally_with_value_tie_break, true, generate_undominated_bin_assignments, is_undominated_individual_vals))


cbc_res = BenchmarkResults("CBC", init_items_x_agents_matrix(), init_items_x_agents_matrix(), agent_range, n_items_list, MulKnapOptions(false, false, [], () -> (), false, () -> (), () -> (), false, () -> (), () -> ()))
#
test_cases::Vector{BenchmarkResults} = [cbc_res]

#=
el_bins = [
    Knapsack(1, Item[], 10, 0),
    Knapsack(2, Item[], 10, 0)
]
el_items = [
    Item(1, 5, [6, 1]),
    Item(2, 5, [1, 6]),
    Item(3, 9, [2, 2])
]
=#

#=
el_bins = [
    Knapsack(1, Item[], 44, 0),
    Knapsack(2, Item[], 47, 0),
    Knapsack(3, Item[], 97, 0)]
el_items = [
    Item(1, 17, [19, 4, 12]),
    Item(2, 51, [16, 20, 13]),
    Item(3, 57, [4, 14, 20])
]

solve_multiple_knapsack_problem(el_bins, el_items, identical.options, true)
solve_multiple_knapsack_problem(el_bins, el_items, r2.options, true)
exit(0)
normal_matrix = init_items_x_agents_matrix()
individual_matrix = init_items_x_agents_matrix()

for (item_index, n_items) in enumerate(n_items_list)
    for n_agents in n_agents_tuple[1]:n_agents_tuple[2]

        ratios_normal = zeros(10)
        ratios_individual = zeros(10)

        for i in 1:10
            bench_items = generate_items(n_items, 1, 60, n_agents, 1, 20)
            bench_bins = generate_knapsacks(n_agents, 30, 120)

            normal_upper_bound = compute_surrogate_upper_bound(bench_bins, bench_items)
            normal_result = solve_multiple_knapsack_problem(deepcopy(bench_bins), deepcopy(bench_items), identical.options, false)
            if normal_upper_bound < normal_result.best_profit || normal_upper_bound < 0 || normal_result.best_profit < 0
                println("MKP : Upper: ", normal_upper_bound, " vs res: ", normal_result.best_profit)
                display(bench_bins)
                display(normal_result.best_assignment)
                display(bench_items)
            end
            ratios_normal[i] = normal_upper_bound / normal_result.best_profit

            individual_upper_bound = compute_max_upper_bound_individual_vals(bench_bins, bench_items)
            individual_result = solve_multiple_knapsack_problem(deepcopy(bench_bins), deepcopy(bench_items), undominated_bb_r2_min_capacity_ordering.options, false).best_profit
            if individual_upper_bound < individual_result
                println("MKP indiv : Upper: ", individual_upper_bound, " vs res: ", individual_result)
                println(bench_bins, bench_items)
            end
            ratios_individual[i] = individual_upper_bound / individual_result
        end

        normal_matrix[item_index, n_agents-n_agents_tuple[1]+1] = mean(ratios_normal)
        individual_matrix[item_index, n_agents-n_agents_tuple[1]+1] = mean(ratios_individual)

    end
end
println("MKP:")
display(normal_matrix)
println("Individual vals")
display(individual_matrix)
exit(0)
=#

for (item_index, n_items) in enumerate(n_items_list)
    for n_agents in n_agents_tuple[1]:n_agents_tuple[2]

        bench_items = generate_items(n_items, 1, 60, n_agents, 1, 20)
        bench_bins = generate_knapsacks(n_agents, 30, 120)

        #=
                bench_bins = [
                    Knapsack(1, Item[], 43, 0),
                    Knapsack(2, Item[], 108, 0)
                    ]
                    bench_items = [
                        Item(1, 57, [20, 19]),
                        Item(2, 29, [10, 14]),
                        Item(3, 23, [4, 8]),
                        Item(4, 9, [16, 9]),
                        Item(5, 11, [13, 13]),
                        Item(6, 20, [18, 14]),
                        Item(7, 18, [15, 4]),
                        Item(8, 52, [8, 10]),
                        Item(9, 25, [15, 7])
                        ]
                        =#
        #issue was: we didnt check if profit was higher on excluded item vs subset

        println("\nTesting on instance with ", n_agents, " agents and ", n_items, " items:")
        println(compute_max_upper_bound_individual_vals(bench_bins, bench_items), " vs ", lp_relaxed_upper_bound(bench_bins, bench_items))
        combined_capacity = sum((bin) -> bin.capacity, bench_bins)
        items_sorted_by_asc_cost = sort(bench_items, by=(item) -> item.cost)
        sum_cost = 0
        count = 0
        for item in items_sorted_by_asc_cost
            if sum_cost + item.cost > combined_capacity
                break
            end
            sum_cost += item.cost
            count += 1
        end
        println("A maximum of ", count, " items fit in the combined knapsack")
        #display(bench_bins)
        #println("\n\n")
        #display(bench_items)
        results = []
        for test_case in test_cases
            if test_case.solve_method == "CBC"
                values = zeros(Int8, n_agents, n_items)
                weights = []
                capacities = []
                for (j, item) in enumerate(bench_items)
                    push!(weights, item.cost)
                    values[:, j] = item.valuations
                end
                for bin in bench_bins
                    push!(capacities, bin.capacity)
                end
                weights = duplicate_vals(weights, n_agents)
                println("\n\nCBC")
                res = linear_model_solver(values, weights, capacities, n_agents, n_items)
                cbc_bench = @benchmark linear_model_solver($values, $weights, $capacities, $n_agents, $n_items)
                println(res.nodes_explored, " nodes explored in ", mean(cbc_bench.times) / 1e6, " ms")
                #show(stdout, MIME("text/plain"), cbc_bench)
                test_case.mean_values[item_index, n_agents-n_agents_tuple[1]+1] = mean(cbc_bench.times)
                test_case.nodes_explored[item_index, n_agents-n_agents_tuple[1]+1] = res.nodes_explored
                push!(results, res)
            else
                println("\n", test_case.solve_method)
                result = solve_multiple_knapsack_problem(deepcopy(bench_bins), deepcopy(bench_items), test_case.options, false)
                bench = @benchmark solve_multiple_knapsack_problem(copy($bench_bins), copy($bench_items), $test_case.options, false)
                println(result.nodes_explored, " nodes explored in ", mean(bench.times) / 1e6, " ms")
                #show(stdout, MIME("text/plain"), bench)
                test_case.mean_values[item_index, n_agents-n_agents_tuple[1]+1] = mean(bench.times)
                test_case.nodes_explored[item_index, n_agents-n_agents_tuple[1]+1] = result.nodes_explored
                push!(results, result)
            end
            for bin in bench_bins
                bin.items = []
            end
        end
        if !all(res -> res.best_profit == results[1].best_profit, results)
            println("\n\nERROR: RESULTS DIFFER BETWEEN MODELS:\n")
            display(bench_bins)
            display(bench_items)
            for res in results
                print_solution(res, test_cases[1].options.individual_vals)
            end
            println("\n\nERROR: RESULTS DIFFER BETWEEN MODELS:\n")
            exit(1)
        end
    end
    plot_results(test_cases, n_items, item_index)
end

