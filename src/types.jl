struct Item
    id::Int
    cost::Int
    valuations::Vector{Int}
end

mutable struct Knapsack
    id::Int
    items::Vector{Item}
    capacity::Int
    load::Int
end

function remove_item_from_vector(items::Vector{Item}, item::Item)
    deleteat!(items, findfirst(x -> x.id == item.id, items))
    return items
end

function remove_knapsack_from_vector!(knapsacks::Vector{Knapsack}, knapsack_to_remove::Knapsack)
    deleteat!(knapsacks, findfirst(x -> x.id == knapsack_to_remove.id, knapsacks))
    return knapsacks
end

function add_knapsack_to_vector_and_return_new_vector(knapsacks::Vector{Knapsack}, knapsack_to_add::Knapsack)
    my_copy = copy(knapsacks)
    push!(my_copy, knapsack_to_add)
    return my_copy
end

function is_smaller_cardinally_with_value_tie_break(items_a::Vector{Item}, items_b::Vector{Item})
    if (length(items_a) == length(items_b))
        return sum((item) -> item.valuations[1], items_a) < sum((item) -> item.valuations[1], items_b)
    else
        return length(items_a) < length(items_b)
    end
end

Base.copy(item::Item) = Item(item.id, item.cost, item.valuations)
Base.:\(items::Vector{Item}, item::Item) = remove_item_from_vector(items, item)
Base.:\(items::Vector{Item}, items_to_remove::Vector{Item}) = foreach(i -> remove_item_from_vector(items, i), items_to_remove)
Base.isless(items_a::Vector{Item}, items_b::Vector{Item}) = is_smaller_cardinally_with_value_tie_break(items_a, items_b)

Base.copy(knap::Knapsack) = Knapsack(knap.id, knap.items, knap.capacity, knap.load)
Base.union(knap_a::Knapsack, knap_b::Knapsack) = Knapsack(-knap_a.id - knap_b, vcat(knap_a.items + knap_b.items), knap_a.capacity + knap_b.capacity, knap_a.load + knap_b.load)
Base.:\(knaps::Vector{Knapsack}, knap_to_remove::Knapsack) = remove_knapsack_from_vector!(knaps, knap_to_remove)
Base.:+(list::Vector{Knapsack}, knap::Knapsack) = add_knapsack_to_vector_and_return_new_vector(list, knap)

