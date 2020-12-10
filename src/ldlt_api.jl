#=
Public interface for (incomplete) LDLt factorizations.
=#

using SparseArrays
using LinearAlgebra

include("ldlt.jl")

"""
    SimplicialLDLt{T} <: Factorization{T}

Simplicial, potentially incomplete LDLt factorization of a sparse symmetric
matrix. This is the return type of [`simplicial_ldlt`](@ref),
[`ildlt_lof`](@ref), [`ildlt_tol`](@ref).

The individual components of the factorization can be accessed via `getproperty`:

| Component | Description                                 |
|:---------:|:--------------------------------------------|
| `F.L`     | `L` (unit lower triangular) part of `LDLt`  |
| `F.D`     | `D` (diagonal) part of `LDLt`               |
| `F.Lt`    | `Lt` (unit upper triangular) part of `LDLt` |
| `F.d`     | diagonal values of `D` as a `Vector`        |
"""
struct SimplicialLDLt{T,C} <: LinearAlgebra.Factorization{T}
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{T}
    conj::C
end

Base.size(F::SimplicialLDLt) = (length(F.colptr)-1).*(1,1)
Base.size(F::SimplicialLDLt, i::Integer) = size(F, i)

function Base.getproperty(F::SimplicialLDLt, d::Symbol)
    Fp,Fi,Fv,conj = getfield.(Ref(F), (:colptr,:rowval,:nzval,:conj))
    if d === :d
        return Fv[@view(Fp[1:end-1])]
    elseif d === :D
        return Diagonal(F.d)
    elseif d === :L
        return I+tril(SparseMatrixCSC(size(F)...,Fp,Fi,Fv),-1)
    elseif d === :Lt
        if conj === Base.conj
            return adjoint(F.L)
        elseif conj === identity
            return transpose(F.L)
        else
            return conj.(transpose(F.L))
        end
    else
        return getfield(F, d)
    end
end



"""
    ldlt_structure(A::SparseMatrixCSC) -> P::SparseMatrixCSC{Bool}

Compute the structurally nonzero entries of the LDLt factorization of `A`.
Only the lower-triangular part of `A` is accessed, and only the
lower-triangular part of `P` is returned.
"""
function ldlt_structure(A)
    @assert size(A,1) == size(A,2)
    @assert size(A,1) == length(A.colptr)-1
    Ap,Ai = A.colptr,A.rowval,A.nzval
    Fp,Fi = symbolic_ldlt(Ap,Ai)
    return SparseMatrixCSC(size(A)...,Fp,Fi,ones(Bool,length(Fi)))
end

"""
    level_of_fill(A::SparseMatrixCSC, c = size(A,2)) -> lof::SparseMatrixCSC{Int}

Compute the level-of-fill of each entry in the sparsity pattern of the
incomplete LDLt factorization of `A` where fill-in is restricted to entries
with `level_of_fill <= c`.

Only the lower-triangular part of `A` is accessed, and only the
lower-triangular part of `P` is returned.

Note that both structural zeros and entries with level-of-fill 0 will satisfy
`lof[i,j] = 0`, but the former will be structural zeros of `lof` while the
latter will be structural nonzeros. The sparsity pattern of `lof` can be
conveniently reduced using the `SparseArrays.fkeep!()` function.
"""
function level_of_fill(A::SparseMatrixCSC,c=size(A,2))
    @assert size(A,1) == size(A,2)
    @assert size(A,1) == length(A.colptr)-1
    Ap,Ai = A.colptr,A.rowval,A.nzval
    Fp,Fi,Fl = symbolic_ildlt_lof(Ap,Ai,c)
    return SparseMatrixCSC(size(A)...,Fp,Fi,Fl)
end



"""
    simplicial_ldlt(A::SparseMatrixCSC; conj=conj) -> F

Compute the LDLt factorization of a sparse symmetric matrix `A`. Only the
lower-triangular part of `A` is accessed.

If `conj = conj` (default), then this function computes the Hermitian
LDLt factorization `A = L*D*L'`. If `conj = identity`, then this function
computes the complex symmetric LDLt factorization `A = L*D*transpose(L)`.

The individual components of the factorization can be accessed via `getproperty`:

| Component | Description                                 |
|:---------:|:--------------------------------------------|
| `F.L`     | `L` (unit lower triangular) part of `LDLt`  |
| `F.D`     | `D` (diagonal) part of `LDLt`               |
| `F.Lt`    | `Lt` (unit upper triangular) part of `LDLt` |
| `F.d`     | diagonal values of `D` as a `Vector`        |
"""
function simplicial_ldlt(A::SparseMatrixCSC; conj=conj)
    @assert size(A,1) == size(A,2)
    @assert size(A,1) == length(A.colptr)-1
    Ap,Ai,Av = A.colptr,A.rowval,A.nzval
    Fp,Fi = symbolic_ldlt(Ap,Ai)
    Fv = numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj=conj)
    return SimplicialLDLt(Fp,Fi,Fv,conj)
end

"""
    ildlt(A::SparseMatrixCSC, P::AbstractSparseMatrixCSC; conj=conj) -> F

Compute the incomplete LDLt factorization of a sparse symmetric matrix `A`
where fill-in is restricted to nonzero entries of `P`. Only the
lower-triangular part of `A` and `P` are accessed.

If `conj = conj` (default), then this function computes the Hermitian
LDLt factorization `A = L*D*L'`. If `conj = identity`, then this function
computes the complex symmetric LDLt factorization `A = L*D*transpose(L)`.

The individual components of the factorization can be accessed via `getproperty`:

| Component | Description                                 |
|:---------:|:--------------------------------------------|
| `F.L`     | `L` (unit lower triangular) part of `LDLt`  |
| `F.D`     | `D` (diagonal) part of `LDLt`               |
| `F.Lt`    | `Lt` (unit upper triangular) part of `LDLt` |
| `F.d`     | diagonal values of `D` as a `Vector`        |
"""
function ildlt(A::SparseMatrixCSC, P::SparseArrays.AbstractSparseMatrixCSC; conj=conj)
    @assert size(A,1) == size(A,2)
    @assert size(A,1) == length(A.colptr)-1
    Ap,Ai,Av = A.colptr,A.rowval,A.nzval
    Fp,Fi = P.colptr,P.rowval
    Fv = numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj=conj)
    return SimplicialLDLt(Fp,Fi,Fv,conj)
end


"""
    ildlt_lof(A::SparseMatrixCSC, c; conj=conj) -> F

Compute the incomplete LDLt factorization of a sparse symmetric matrix `A`
where fill-in is restricted to entries with `level_of_fill <= c`. Only the
lower-triangular part of `A` is accessed.

If `conj = conj` (default), then this function computes the Hermitian
LDLt factorization `A = L*D*L'`. If `conj = identity`, then this function
computes the complex symmetric LDLt factorization `A = L*D*transpose(L)`.

The individual components of the factorization can be accessed via `getproperty`:

| Component | Description                                 |
|:---------:|:--------------------------------------------|
| `F.L`     | `L` (unit lower triangular) part of `LDLt`  |
| `F.D`     | `D` (diagonal) part of `LDLt`               |
| `F.Lt`    | `Lt` (unit upper triangular) part of `LDLt` |
| `F.d`     | diagonal values of `D` as a `Vector`        |
"""
function ildlt_lof(A::SparseMatrixCSC, c::Integer; conj=conj)
    @assert size(A,1) == size(A,2)
    @assert size(A,1) == length(A.colptr)-1
    Ap,Ai,Av = A.colptr,A.rowval,A.nzval
    Fp,Fi = symbolic_ildlt_lof(Ap,Ai,c)
    Fv = numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj=conj)
    return SimplicialLDLt(Fp,Fi,Fv,conj)
end

"""
    ildlt_tol(A::SparseMatrixCSC,tol; conj=conj) -> F

Compute the incomplete LDLt factorization of a`A` where fill-in is restricted
to entries with `|F[i,j]| > tol`. Only the lower-triangular part of `A` is
accessed.

If `conj = conj` (default), then this function computes the Hermitian
LDLt factorization `A = L*D*L'`. If `conj = identity`, then this function
computes the complex symmetric LDLt factorization `A = L*D*transpose(L)`.

The individual components of the factorization can be accessed via `getproperty`:

| Component | Description                                 |
|:---------:|:--------------------------------------------|
| `F.L`     | `L` (unit lower triangular) part of `LDLt`  |
| `F.D`     | `D` (diagonal) part of `LDLt`               |
| `F.Lt`    | `Lt` (unit upper triangular) part of `LDLt` |
| `F.d`     | diagonal values of `D` as a `Vector`        |
"""
function ildlt_tol(A::SparseMatrixCSC, tol::Real; conj=conj)
    @assert size(A,1) == size(A,2)
    @assert size(A,1) == length(A.colptr)-1
    Ap,Ai,Av = A.colptr,A.rowval,A.nzval
    Fp,Fi,Fv = ildlt_tol(Ap,Ai,Av,tol; conj=conj)
    return SimplicialLDLt(Fp,Fi,Fv,conj)
end

