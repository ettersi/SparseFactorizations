"""
    IntSet(n)

Subset of `{1,...,n}` with `O(1)` `in()` and `push!()`.
"""
struct IntSet
    set::Vector{Int}
    iselement::Vector{Bool}
end

IntSet(n) = IntSet(sizehint!(Int[],n), fill(false,n))

limit(s::IntSet) = length(s.iselement)
Base.length(s::IntSet) = length(s.set)
Base.iterate(s::IntSet, args...) = iterate(s.set, args...)
Base.@propagate_inbounds Base.in(v::Int, s::IntSet) = s.iselement[v]

function Base.push!(s::IntSet, v::Int)
    @boundscheck @assert 1 <= v <= limit(s)
    @inbounds begin
        if !s.iselement[v]
            push!(s.set,v)
            s.iselement[v] = true
        end
        return s
    end
end

Base.@propagate_inbounds function Base.union!(s::IntSet, itr)
    Base.haslength(itr) && sizehint!(s.set,length(s)+length(itr))
    for v in itr
        push!(s,v)
    end
    return s
end

function Base.empty!(s::IntSet)
    @inbounds @. s.iselement[s.set] = false
    empty!(s.set)
    return s
end

Base.sort!(s::IntSet) = sort!(s.set)