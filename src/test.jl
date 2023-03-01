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
#solve_multiple_knapsack_problem(test_knaps, test_items, true, true)

#=
# Individual values 
=#
indiv_knaps = [Knapsack(1, [], 50, 0), Knapsack(2, [], 30, 0)]
indiv_items = [Item(1, 10, [1, 5]), Item(2, 20, [2, 14]), Item(3, 40, [4, 8]), Item(4, 9, [5, 1]), Item(5, 9, [6, 2])]
#test_bench = @benchmark generate_all_feasible_maximal_bin_assignments(copy($test_knaps[1]), copy($test_items))
#new_bench = @benchmark generate_all_feasible_maximal_bin_assignments_two(copy($test_knaps[1]), copy($test_items))
#println("Res: ", test_bench, " vs ", new_bench)
#solve_multiple_knapsack_problem(indiv_knaps, indiv_items, true, true, true)



mutable struct BenchmarkResults
    solve_method::String
    mean_values::Matrix{Float64}
    agent_number_list::Vector{Int}
    items_number_list::Vector{Int}
end

function plot_results(models::Vector{BenchmarkResults}, test_info::Vector{Int}, scaling::Int)
    x = models[1].agent_number_list
    x_label = "Trial number"
    y_label = "Time (ns)"
    if (scaling == 1000)
        y_label = "Time (μs)"
    elseif (scaling == 1000000)
        y_label = "Time (ms)"
    elseif (scaling == 1000000000)
        y_label = "Time (s)"
    end

    plot_title_mean = "Mean values, " * string(test_info[1]) * "-" * string(test_info[2]) * " agents, " * string(test_info[3]) * "-" * string(test_info[4]) * " items.png"

    for i in 1:size(models[1].mean_values)[1]
        p = plot(x, [models[1].mean_values[i, :] / scaling models[2].mean_values[i, :] / scaling models[3].mean_values[i, :] / scaling], xlabel="Number of agents", ylabel=y_label, title=plot_title_mean, label=["BC" "BC no pp" "JuMP - CBC"], linewidth=3)
        savefig(p, "../results/mean_" * string(models[1].agent_number_list[1]) * "-" * string(models[1].agent_number_list[end]) * "agents_" * string(models[1].items_number_list[i]) * "items.png")
    end
end




item_list_step = 4
n_items_list = collect(range(4, 12, step=item_list_step))
n_agents_tuple = (4, 8)

fukunaga_res = BenchmarkResults("Bin completion", zeros((length(n_items_list), n_agents_tuple[2] - n_agents_tuple[1] + 1)), range(n_agents_tuple[1], n_agents_tuple[2]), n_items_list)
fukunaga_without_res = BenchmarkResults("Bin completion without sorting & preprocessing", zeros((length(n_items_list), n_agents_tuple[2] - n_agents_tuple[1] + 1)), range(n_agents_tuple[1], n_agents_tuple[2]), n_items_list)
cbc_res = BenchmarkResults("CBC", zeros((length(n_items_list), n_agents_tuple[2] - n_agents_tuple[1] + 1)), range(n_agents_tuple[1], n_agents_tuple[2]), n_items_list)
test_info = [5, 10, 10, 20]

for n_agents in n_agents_tuple[1]:n_agents_tuple[2]

    for n_items in n_items_list

        bench_items = generate_items(n_items, 1, 30, n_agents, 1, 20)
        bench_bins = generate_knapsacks(n_agents, 30, 120)
        #=
        println("\nBenchmarking with items: \n", bench_items)
        println("\nBenchmarking with bins: \n", bench_bins)
        println(
            "\nRes1: ", compute_max_upper_bound_individual_vals(bench_bins, bench_items),
            "\nRes2: ", compute_upper_bound(bench_bins, bench_items),
        )

        for (idx, bin) in enumerate(bench_bins)
            adjusted_items = Array{Item}(undef, 0)
            for item in bench_items
                itemcpy = Item(item.id, item.cost, fill(item.valuations[idx], length(item.valuations)))
                push!(adjusted_items, itemcpy)
            end
            println("Bin ", bin.id, " with c=", bin.capacity, ": ", solve_binary_knapsack(bin, adjusted_items))
        end
        =#

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
        #println("\n\n----------------------------------------------------------------------\n----------------------------------------------------------------------\n----------------------------------------------------------------------")
        println("\nTesting on instance with ", n_agents, " agents and ", n_items, " items:")

        fukunaga_temp = []
        fukunaga_without_temp = []
        cbc_temp = []

        for i in 1:3
            #println("\nBin completion:")
            fukunaga_bench = @benchmark solve_multiple_knapsack_problem(copy($bench_bins), copy($bench_items), true, false)
            #show(stdout, MIME("text/plain"), fukunaga_bench)

            #println("\n\nBin completion without and preprocessing")
            fukunaga_bench_without = @benchmark solve_multiple_knapsack_problem(copy($bench_bins), copy($bench_items), false, false)
            #show(stdout, MIME("text/plain"), fukunaga_bench_without)

            #println("\n\nCBC")
            cbc_bench = @benchmark linear_model_solver($values, $weights, $capacities, $n_agents, $n_items)
            #show(stdout, MIME("text/plain"), cbc_bench)

            push!(fukunaga_temp, mean(fukunaga_bench.times))
            push!(fukunaga_without_temp, mean(fukunaga_bench_without.times))
            push!(cbc_temp, mean(cbc_bench.times))
        end
        fukunaga_res.mean_values[div(n_items, item_list_step), n_agents] = mean(fukunaga_temp)
        fukunaga_without_res.mean_values[div(n_items, item_list_step), n_agents] = mean(fukunaga_without_temp)
        cbc_res.mean_values[div(n_items, item_list_step), n_agents] = mean(cbc_temp)
    end
end

plot_results([fukunaga_res, fukunaga_without_res, cbc_res], test_info, 1000000)