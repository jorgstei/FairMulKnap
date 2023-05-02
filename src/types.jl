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
# TODO - unused?
Base.isless(items_a::Vector{Item}, items_b::Vector{Item}) = is_smaller_cardinally_with_value_tie_break(items_a, items_b)

Base.union(knap_a::Knapsack, knap_b::Knapsack) = Knapsack(-knap_a.id - knap_b, vcat(knap_a.items + knap_b.items), knap_a.capacity + knap_b.capacity, knap_a.load + knap_b.load)
Base.:+(list::Vector{Knapsack}, knap::Knapsack) = add_knapsack_to_vector_and_return_new_vector(list, knap)

#setdiff(x, y) instead of 
#vcat istedet of + maybe?
# maybe definer egen container-type i stedet for å overwrite metoder for 

