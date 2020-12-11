using LinearAlgebra
using SparseArrays

include("CHOLMOD.jl")
include("../src/SparseFactorizations.jl")
import IncompleteLU

using PyPlot

function laplacian_1d(n)
    return spdiagm(
        -1 => fill(-1.0,n-1),
         0 => fill( 2.0,n),
         1 => fill(-1.0,n-1)
    )
end

function laplacian_2d(n)
    Δ = laplacian_1d(n)
    Id = sparse(I,n,n)
    return kron(Δ,Id) + kron(Id,Δ)
end

function laplacian_3d(n)
    Δ = laplacian_1d(n)
    Id = sparse(I,n,n)
    return kron(Δ,Id,Id) + kron(Id,Δ,Id) + kron(Id,Id,Δ)
end

function benchmark()
    nreps = 3
    if true
        n = round.(Int, 10.0.^LinRange(1,2.2,20))
        mat = laplacian_2d
    else
        n = round.(Int, 10.0.^LinRange(log10(5),log10(30),10))
        mat = laplacian_3d
    end
    funs = (
        ("CHOLMOD, simplicial", A->CHOLMOD.Sparse(A,-1), SC_A -> CHOLMOD.simplicial_factorize(SC_A)),
        ("CHOLMOD, supernodal", A->CHOLMOD.Sparse(A,-1), SC_A -> CHOLMOD.supernodal_factorize(SC_A)),
        # ("SparseFactorizations, analysis", identity, A -> SparseFactorizations.ldlt_structure(A)),
        ("SparseFactorizations, factorization", A->(A,SparseFactorizations.ldlt_structure(A)), ((A,P),) -> SparseFactorizations.ildlt(A,P)),
        ("SparseFactorizations", identity, A -> SparseFactorizations.ldlt(A)),
        ("SparseFactorizations, lof", identity, A -> SparseFactorizations.ildlt_lof(A,size(A,2))),
        ("SparseFactorizations, tol", identity, A -> SparseFactorizations.ildlt_tol(A,0.0)),
        ("IncompleteLU", identity, A -> IncompleteLU.ilu(A,τ=0.0)),
    )

    timings = [zeros(size(n)) for f in funs]

    # Warm up
    A = mat(4)
    for (name,setup,run) in funs
        run(setup(A))
    end

    for kn = 1:length(n)
        println("n = $(n[kn])")
        A = mat(n[kn])
        p = CHOLMOD.amd(A)
        A = A[p,p]
        for (kf,(name,setup,run)) in enumerate(funs)
            args = setup(A)
            timings[kf][kn] = minimum(@elapsed run(args) for i = 1:nreps)
        end
    end
    return (n,funs,timings)
end

function plot((n,funs,timings))
    fig = figure(figsize=(8,4))
    try
        for ((name,_),timing) in zip(funs,timings)
            loglog(n,timing,label=name)
        end
        xlabel(L"n")
        ylabel("runtime [seconds]")
        legend(
            frameon=false,
            loc="center left",
            bbox_to_anchor=(1,0.5)
        )
        tight_layout()
        savefig(@__DIR__()*"/benchmark.png")
    finally
        close(fig)
    end
end