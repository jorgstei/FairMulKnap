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



function plot_results(models::Vector{BenchmarkResults}, n_agents_tuple::Tuple{Int,Int}, n_items_list::Vector{Int})
    x = models[1].agent_number_list

    max_time = -1
    for model in models
        model_max = maximum(model.mean_values)
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


    for (index, n_items) in enumerate(n_items_list)
        plot_title = "Mean times - " * string(n_items) * " items"
        p = plot(x, [res.mean_values[index, :] / scaling for res in models], xlabel="Agents", ylabel=y_label, title=plot_title, label=reshape(map((result) -> result.solve_method, models), 1, length(models)), linewidth=3)
        savefig(p, "../results/mean_times/mean_" * string(models[1].agent_number_list[1]) * "-" * string(models[1].agent_number_list[end]) * "agents_" * string(models[1].items_number_list[index]) * "items.png")

        nodes_title = "Nodes explored - " * string(n_items) * " items"
        node_plot = plot(x, [res.nodes_explored[index, :] for res in models], xlabel="Agents", ylabel="Nodes", title=nodes_title, label=reshape(map((result) -> result.solve_method, models), 1, length(models)), linewidth=3)
        savefig(node_plot, "../results/nodes/nodes_" * string(models[1].agent_number_list[1]) * "-" * string(models[1].agent_number_list[end]) * "agents_" * string(models[1].items_number_list[index]) * "items.png")
    end
end



# How big the step between n items should be, e.g. 2 makes 1, 3, 5, 7 ...
item_list_step = 2
n_items_list = collect(range(1, 1, step=item_list_step))
# How many knapsacks should be generated tested, (from, to)
n_agents_tuple = (1, 1)
#=
=#
n_items_list = collect(range(2, 16, step=item_list_step))
n_agents_tuple = (2, 7)

items_x_agents = zeros((length(n_items_list), n_agents_tuple[2] - n_agents_tuple[1] + 1))
agent_range = range(n_agents_tuple[1], n_agents_tuple[2])

identical = BenchmarkResults("MKP", items_x_agents, items_x_agents, agent_range, n_items_list, MulKnapOptions(false, true, [], compute_upper_bound, generate_all_feasible_maximal_bin_assignments_using_up_to_k_combs, () -> ()))
r2 = BenchmarkResults("MKP + R2", items_x_agents, items_x_agents, agent_range, n_items_list, MulKnapOptions(false, true, [pisinger_r2_reduction], compute_upper_bound, generate_all_feasible_maximal_bin_assignments_using_up_to_k_combs, () -> ()))

basic_res = BenchmarkResults("Up-to-k", items_x_agents, items_x_agents, agent_range, n_items_list, MulKnapOptions(true, true, [], compute_max_upper_bound_individual_vals, generate_all_feasible_bin_assignments_using_up_to_k_combs, () -> ()))
undominated = BenchmarkResults("Undominated", items_x_agents, items_x_agents, agent_range, n_items_list, MulKnapOptions(true, true, [], compute_max_upper_bound_individual_vals, generate_undominated_bin_assignments, is_undominated_individual_vals))
undominated_r2 = BenchmarkResults("Undominated+R2", items_x_agents, items_x_agents, agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_individual_valuations], compute_max_upper_bound_individual_vals, generate_undominated_bin_assignments, is_undominated_individual_vals))
undominated_r2_no_sort = BenchmarkResults("Undominated+R2nosort", items_x_agents, items_x_agents, agent_range, n_items_list, MulKnapOptions(true, true, [pisinger_r2_reduction_no_sort_individual_valuations], compute_max_upper_bound_individual_vals, generate_undominated_bin_assignments, is_undominated_individual_vals))

cbc_res = BenchmarkResults("CBC", items_x_agents, items_x_agents, agent_range, n_items_list, MulKnapOptions(false, false, [], () -> (), () -> (), () -> ()))

test_cases::Vector{BenchmarkResults} = [basic_res, undominated, undominated_r2, undominated_r2_no_sort]


for (item_index, n_items) in enumerate(n_items_list)
    for n_agents in n_agents_tuple[1]:n_agents_tuple[2]


        bench_items = generate_items(n_items, 1, 60, n_agents, 1, 20)
        bench_bins = generate_knapsacks(n_agents, 30, 120)
        #=bench_bins = [
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
        issue was: we didnt check if profit was higher on excluded item vs subset
        =#
        println("\nTesting on instance with ", n_agents, " agents and ", n_items, " items:")
        #display(bench_bins)
        #println("\n\n")
        #display(bench_items)
        results = []
        for test_case in test_cases
            if test_case.solve_method == "CBC"
                #=
                values = []
                weights = []
                capacities = []
                for item in bench_items
                    push!(values, item.valuations[1])
                    push!(weights, item.cost)
                end
                for bin in bench_bins
                    push!(capacities, bin.capacity)
                end
                values = duplicate_vals(values, n_agents)
                weights = duplicate_vals(weights, n_agents)
                println("\n\nCBC")
                cbc_bench = @benchmark linear_model_solver($values, $weights, $capacities, $n_agents, $n_items)
                show(stdout, MIME("text/plain"), cbc_bench)
                =#
                break
            else
                println("\n", test_case.solve_method)
                result = solve_multiple_knapsack_problem(deepcopy(bench_bins), deepcopy(bench_items), test_case.options, false)
                bench = @benchmark solve_multiple_knapsack_problem(copy($bench_bins), copy($bench_items), $test_case.options, false)
                #println(result.nodes_explored, " nodes explored in ", mean(bench.times) / 1e6, " ms")
                #show(stdout, MIME("text/plain"), bench)
                test_case.mean_values[item_index, n_agents-n_agents_tuple[1]+1] = mean(bench.times)
                test_case.nodes_explored[item_index, n_agents-n_agents_tuple[1]+1] = result.nodes_explored
                push!(results, result)
            end
            for bin in bench_bins
                bin.items = []
            end

            #solve_multiple_knapsack_problem(deepcopy(bench_bins), deepcopy(bench_items), test_case.preprocess, true, test_case.reductions, test_case.compute_upper_bound, test_case.generate_assignments, test_case.is_undominated, test_case.items_diff)
        end
        if !all(res -> res.best_profit == results[1].best_profit, results)
            println("\n\nERROR: RESULTS DIFFER BETWEEN MODELS:\n")
            display(bench_bins)
            display(bench_items)
            for res in results
                print_solution(res)
            end
            println("\n\nERROR: RESULTS DIFFER BETWEEN MODELS:\n")
            exit(1)
        end
    end
end

plot_results(test_cases, n_agents_tuple, n_items_list)