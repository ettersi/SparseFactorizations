#=
The tests here merely check that the interface functions work.
Testing of the implementation is done in selinv.jl.
=#

using Test
using Random
using SparseArrays
using LinearAlgebra
using SparseFactorizations

@testset "selinv_api" begin
    @testset "selinv" begin
        @testset for (conj,adj) in ((conj,adjoint),(identity,transpose))
            Random.seed!(1)
            A = sprand(ComplexF64, 10,10,0.1)
            A = I + A + adj(A)
            F = simplicial_ldlt(A; conj=conj)
            B = selinv(F)
            Bref = inv(Matrix(A)) .* ldlt_structure(A)
            @test B ≈ Bref
        end
    end
end
