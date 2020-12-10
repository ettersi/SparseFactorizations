#=
Low-level computational routines for (incomplete) LDLt factorizations.
=#

using SparseArrays

include("IntSets.jl")
include("iterate_jkp.jl")

"""
    symbolic_ldlt(Ap,Ai) -> Fp,Fi

Compute the sparsity structure of the LDLt factorization of `A`.

Only the lower-triangular part of `A` is accessed, and only the
lower-triangular part of the structure of `F` is returned.
"""
function symbolic_ldlt(Ap,Ai)
    n = length(Ap)-1
    @assert length(Ap) >= 1
    @assert issorted(Ap)
    @assert length(Ai) >= Ap[end]-1
    @assert all(1 .<= Ai .<= n)

    @inbounds begin
        # Return variables
        Fp = Vector{Int}(undef,n+1); Fp[1] = 1
        Fi = Vector{Int}(undef,0)

        # Workspace for a single column
        Fi_aj = IntSet(n)

        # Main algorithm
        for (j,kps) in iterate_jkp(Fp,Fi,Val(true))
            # Initialise column
            for i in @view Ai[Ap[j]:Ap[j+1]-1]
                if i < j; continue; end
                push!(Fi_aj,i)
            end

            # Pull updates into F[j:n,j]
            for (k,ps) in kps
                union!(Fi_aj, @view Fi[ps])
            end

            # Copy temporary column into F
            append!(Fi, sort!(Fi_aj.set)); empty!(Fi_aj)
            Fp[j+1] = length(Fi)+1
        end
    end
    return Fp,Fi
end

"""
    symbolic_ildlt_lof(Ap,Ai,c) -> Fp,Fi,Fl

Compute the sparsity structure of the incomplete LDLt factorization of `A`
where fill-in is restricted to entries with `level_of_fill <= c`.

`Fl` is a vector of length `nnz(F)` containing the level-of-fill of the
entries of `F`.

Only the lower-triangular part of `A` is accessed, and only the
lower-triangular part of the structure of `F` is returned.

The runtime of this function is asymptotically the same as that of
`numeric_ldlt()`.
"""
function symbolic_ildlt_lof(Ap,Ai,c)
    n = length(Ap)-1
    @assert length(Ap) >= 1
    @assert issorted(Ap)
    @assert length(Ai) >= Ap[end]-1
    @assert all(1 .<= Ai .<= n)

    @inbounds begin
        # Return variables
        Fp = Vector{Int}(undef,n+1); Fp[1] = 1
        Fi = Vector{Int}(undef,0)
        Fl = Vector{Int}(undef,0)

        # Workspace for a single column
        Fi_aj = IntSet(n)
        Fl_aj = fill(typemax(Int),n)

        # Main algorithm
        for (j,ks) in iterate_jkp(Fp,Fi)
            # Initialise column
            for i in @view Ai[Ap[j]:Ap[j+1]-1]
                if i < j; continue; end
                push!(Fi_aj,i)
                Fl_aj[i] = 0
            end

            # Pull updates into F[j:n,j]
            for (k,ps) in ks
                Fl_jk = Fl[ps[1]]
                if Fl_jk >= c; continue; end
                for p in ps
                    i = Fi[p]
                    Fl_ik = Fl[p]

                    Fl_ij = Fl_ik + Fl_jk + 1
                    if Fl_ij <= c
                        push!(Fi_aj,i)
                        Fl_aj[i] = min(Fl_aj[i],Fl_ij)
                    end
                end
            end

            # Copy workspace into F
            append!(Fi, sort!(Fi_aj.set)); empty!(Fi_aj)
            Fp[j+1] = length(Fi)+1

            sizehint!(Fl,length(Fi))
            for i = @view Fi[Fp[j]:length(Fi)]
                push!(Fl,Fl_aj[i])
                Fl_aj[i] = typemax(Int)
            end
        end
    end
    return Fp,Fi,Fl
end

"""
    numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj = conj) -> Fv

Compute the values of the (incomplete) LDLt factorization of `A` associated
with the given structure.

If `conj = conj` (default), then this function computes the Hermitian
LDLt factorization `A = L*D*L'`. If `conj = identity`, then this function
computes the complex symmetric LDLt factorization `A = L*D*transpose(L)`.

Only the lower-triangular part of `A` is accessed, and `F` is assumed to be
lower-triangular.
"""
function numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj = conj)
    Tv = eltype(Av)
    n = length(Ap)-1

    @assert length(Ap) >= 1
    @assert issorted(Ap)
    @assert length(Ai) >= Ap[end]-1
    @assert length(Av) >= Ap[end]-1
    @assert all(1 .<= Ai .<= n)
    @assert length(Fp) == length(Ap)
    @assert issorted(Fp)
    @assert length(Fi) >= Fp[end]-1
    @assert all(1 .<= Fi .<= n)

    @inbounds begin
        # Return variables
        Fv = Vector{Tv}(undef,length(Fi))

        # Workspace for a single column
        Fv_aj = Vector{Tv}(undef,n)

        # Main algorithm
        for (j,ks) in iterate_jkp(Fp,Fi)
            # Initialise column
            for i in @view Fi[Fp[j]:Fp[j+1]-1]
                Fv_aj[i] = zero(Tv)
            end
            for p in Ap[j]:Ap[j+1]-1
                Fv_aj[Ai[p]] = Av[p]
            end

            # Pull updates into F[j:n,j]
            for (k,ps) in ks
                f = Fv[Fp[k]]*conj(Fv[ps[1]])
                for p in ps
                    # We compute a few dropped fill-ins here. It turns out computing
                    # and discarding is faster than introducing a branch.
                    Fv_aj[Fi[p]] -= Fv[p]*f
                end
            end

            # Copy temporary column into F
            d = Fv_aj[j]
            Fv[Fp[j]] = d
            for p in Fp[j]+1:Fp[j+1]-1
                Fv[p] = Fv_aj[Fi[p]]/d
            end
        end
    end
    return Fv
end

"""
    ildlt_tol(Ap,Ai,tol) -> Fp,Fi,Fv

Compute the incomplete LDLt factorization of `A` where fill-in is restricted
to entries with `|F[i,j]| > tol`.

If `conj = conj` (default), then this function computes the Hermitian
LDLt factorization `A = L*D*L'`. If `conj = identity`, then this function
computes the complex symmetric LDLt factorization `A = L*D*transpose(L)`.

Only the lower-triangular part of `A` is accessed, and only the
lower-triangular part of `F` is returned.
"""
function ildlt_tol(Ap,Ai,Av,tol; conj = conj)
    n = length(Ap)-1
    Tv = float(eltype(Av))

    @assert length(Ap) >= 1
    @assert issorted(Ap)
    @assert length(Ai) >= Ap[end]-1
    @assert length(Av) >= Ap[end]-1
    @assert all(1 .<= Ai .<= n)

    @inbounds begin
        # Return variables
        Fp = Vector{Int}(undef,n+1); Fp[1] = 1
        Fi = Vector{Int}(undef,0)
        Fv = Vector{Tv}(undef,0)

        # Workspace for a single column
        Fi_aj = IntSet(n)
        Fv_aj = zeros(Tv,n)

        # Main algorithm
        for (j,kps) in iterate_jkp(Fp,Fi)
            # Initialise column
            for p in Ap[j]:Ap[j+1]-1
                i = Ai[p]
                if i < j; continue; end
                push!(Fi_aj,i)
                Fv_aj[i] = Av[p]
            end

            # Pull updates into F[j:n,j]
            for (k,ps) in kps
                Fv_kj = conj(Fv[ps[1]])
                if abs(Fv_kj) < tol; continue; end
                Fv_kk = Fv[Fp[k]]
                f = Fv_kk*Fv_kj
                for p in ps
                    i = Fi[p]
                    Fv_ik = Fv[p]
                    if abs(Fv_ik) < tol; continue; end
                    push!(Fi_aj,i)
                    Fv_aj[i] -= Fv_ik*f
                end
            end

            # Copy temporary column into F
            d = Fv_aj[j]
            push!(Fi,j)
            push!(Fv,d)
            Fv_aj[j] = 0
            for i in sort!(Fi_aj)
                if i == j; continue; end
                push!(Fi,i)
                push!(Fv,Fv_aj[i]/d)
                Fv_aj[i] = 0
            end
            Fp[j+1] = length(Fi)+1
            empty!(Fi_aj)
        end
    end
    return Fp,Fi,Fv
end
