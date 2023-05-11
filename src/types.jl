# For immutable items er equality field-basert, trenger ikke overwrite
struct Item
    id::Int
    cost::Int
    valuations::Vector{Int}
end
# Knapsack.id needs to correspond with the index of the valuation of that knapsack in Item.valuations
mutable struct Knapsack
    id::Int
    items::Vector{Item}
    capacity::Int
    load::Int
end

mutable struct MKP_return
    best_profit::Int
    best_assignment::Vector{Knapsack}
    nodes_explored::Int
end

mutable struct MulKnapOptions
    individual_vals::Bool
    preprocess::Bool
    reductions::Vector
    compute_upper_bound::Function
    generate_assignments::Function
    is_undominated::Function
end

function add_knapsack_to_vector_and_return_new_vector(knapsacks::Vector{Knapsack}, knapsack_to_add::Knapsack)
    my_copy = copy(knapsacks)
    push!(my_copy, knapsack_to_add)
    return my_copy
end

function is_smaller_cardinally_with_value_tie_break(items_a::Vector{Item}, items_b::Vector{Item})
    if (length(items_a) == length(items_b))
        return sum((item) -> maximum(item.valuations), items_a) < sum((item) -> maximum(item.valuations), items_b)
    else
        return length(items_a) < length(items_b)
    end
end

import Base: ==
==(x::Knapsack, y::Knapsack) = x.id == y.id

Base.copy(item::Item) = Item(item.id, item.cost, item.valuations)
Base.copy(knap::Knapsack) = Knapsack(knap.id, knap.items, knap.capacity, knap.load)

Base.isless(items_a::Vector{Item}, items_b::Vector{Item}) = is_smaller_cardinally_with_value_tie_break(items_a, items_b)

Base.union(knap_a::Knapsack, knap_b::Knapsack) = Knapsack(-knap_a.id - knap_b, vcat(knap_a.items + knap_b.items), knap_a.capacity + knap_b.capacity, knap_a.load + knap_b.load)
Base.:+(list::Vector{Knapsack}, knap::Knapsack) = add_knapsack_to_vector_and_return_new_vector(list, knap)

function is_smaller_profit_divided_by_weight(a::Item, b::Item)
    return a.valuations[1] / a.cost < b.valuations[1] / b.cost
end

function is_smaller_max_profit_divided_by_weight(a::Item, b::Item, relevant_indices::Vector{Int})
    return maximum([a.valuations[i] for i in relevant_indices]) / a.cost < maximum([b.valuations[i] for i in relevant_indices]) / b.cost
end



