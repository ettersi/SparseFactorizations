"""
    iterate_jkp(Fp,Fi, top_k = Val(false))

Iterable such that
```
for (j,kps) in iterate_jkp(Fp,Fi)
    for (k,ps) in kps
        for i in Fi[ps]
```
is equivalent to
```
for j = 1:n
    for k = 1:j-1 if F[k,j] != 0 && (!top_k || all(F[k+1:j-1,k] .== 0))
        for i = j:n if F[i,k] != 0
```
The body of the `j`-loop may update the sparsity pattern of `F[:,j:n]`.

Iterator creation requires `O(n)` runtime and allocates two `Vector{Int}(n)`.
Each call to `iterate()` requires `O(1)` runtime and is allocation-free.

`issorted(Fi[Fp[j]:Fp[j+1]-1])` must hold true for all `j in 1:n`. The
iterator may produce incorrect results if this property is violated.
"""
iterate_jkp(Fp,Fi, top_k = Val(false)) = Iterate_jkp(Fp,Fi, top_k)



#=
This implementation is based on

  [ESS81]
    Eisenstat, Schultz, Sherman
    Algorithms and Data Structures for Sparse Symmetric Gaussian Elimination
    SIAM Journal on Scientific and Statistical Computing (1981)

Meaning of variables:

  kl[i] with i >= j points to a linked-list representation of the set
    { k < j | F[i,k] != 0 && all(F[j+1:i-1,k] .== 0) }
  Note that this set simplifies to { k < j | F[j,k] != 0 } if i == j.
  End of list is indicated by kl[i] = 0.

  rp is a vector of length n such that
    ( rp[k] : Fp[k+1]-1 ) = nonzeros in F[j:n,k] for all k < j

The correspondences to the variables in [ESS81] are kl -> JL and rp -> IL.
=#

struct Iterate_jkp{TopK}
    Fp::Vector{Int}
    Fi::Vector{Int}
    kl::Vector{Int}
    rp::Vector{Int}
end

Base.length(jkp::Iterate_jkp) = length(jkp.Fp)-1

struct Iterate_kp{TopK}
    Fp::Vector{Int}
    Fi::Vector{Int}
    kl::Vector{Int}
    rp::Vector{Int}
    j::Int
end

Base.IteratorSize(::Type{<:Iterate_kp}) = Base.SizeUnknown()

function Iterate_jkp(Fp,Fi, top_k = Val(false))
    n = length(Fp)-1

    # Allocate workspace
    kl = fill(0,n)
    rp = Vector{Int}(undef,n)

    return Iterate_jkp{top_k}(Fp,Fi,kl,rp)
end

function Base.iterate(jkp::Iterate_jkp{T}) where {T}
    Fp = jkp.Fp
    Fi = jkp.Fi
    kl = jkp.kl
    rp = jkp.rp
    length(Fp) <= 1 && return nothing

    # Iterate over first j
    ks = Iterate_kp{T}(Fp,Fi,kl,rp,1)
    return (1,ks), 1
end

function Base.iterate(jkp::Iterate_jkp{T},j) where {T}
    Fp = jkp.Fp
    Fi = jkp.Fi
    kl = jkp.kl
    rp = jkp.rp
    length(Fp)-1 == j && return nothing

    # Update kl and rp for finished j and move to next j
    for p in @inbounds(Fp[j]:Fp[j+1]-1)
        i = Fi[p]
        if i > j
            @inbounds kl[i],kl[j] = j,kl[i]
            @inbounds rp[j] = p
            break
        end
    end
    j += 1

    # Iterate over next j
    ks = Iterate_kp{T}(Fp,Fi,kl,rp,j)
    return (j,ks),j
end

function Base.iterate(kp::Iterate_kp)
    Fp = kp.Fp
    Fi = kp.Fi
    kl = kp.kl
    rp = kp.rp
     j = kp.j

    # Iterate over first k
    @inbounds k = kl[j]
    k == 0 && return nothing
    return (k, @inbounds(rp[k]:Fp[k+1]-1)),k
end

function Base.iterate(kp::Iterate_kp{Val(false)}, k)
    Fp = kp.Fp
    Fi = kp.Fi
    kl = kp.kl
    rp = kp.rp
    j = kp.j
    n = length(Fp)-1

    # Update kl and rp for finished k and move to next k
    rpk = @inbounds( rp[k] += 1 )
    kk = @inbounds kl[k]
    if rpk < @inbounds Fp[k+1]
        i = Fi[rpk]
        @assert 1 <= i <= n
        @inbounds kl[i],kl[k] = k,kl[i]
    end
    k = kk

    # Iterate over next k
    k == 0 && return nothing
    return (k, @inbounds(rp[k]:Fp[k+1]-1)),k
end

function Base.iterate(kp::Iterate_kp{Val(true)}, k)
    Fp = kp.Fp
    Fi = kp.Fi
    kl = kp.kl
    rp = kp.rp
     j = kp.j

    # Iterate over next k
    k = @inbounds kl[k]
    k == 0 && return nothing
    return (k, @inbounds(rp[k]:Fp[k+1]-1)),k
end