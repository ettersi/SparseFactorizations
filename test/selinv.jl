module Test_selinv

include("../src/ldlt.jl")
include("../src/selinv.jl")

using Test
using Random
using SparseArrays
using LinearAlgebra

@testset "selinv_from_ldlt" begin
    function test(A,conj,adj)
        Ap,Ai,Av = A.colptr,A.rowval,A.nzval
        Fp,Fi = symbolic_ldlt(Ap,Ai)
        Fv = numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj=conj)
        P = SparseMatrixCSC(size(A)...,Fp,Fi,ones(Bool,length(Fi)))
        Bv = selinv_from_ldlt(Fp,Fi,Fv; conj=conj)
        B = SparseMatrixCSC(size(A)...,Fp,Fi,Bv)
        Bref = inv(Matrix(A)) .* P
        @test B ≈ Bref
    end

    for T in (Float32,Float64,ComplexF32,ComplexF64)
        for (conj,adj) in ((conj,adjoint),(identity,transpose))
            @testset "n = 1" begin
                Random.seed!(1)
                Ap = [1,2]
                Ai = [1]
                Av = rand(T,1)
                A = SparseMatrixCSC(1,1,Ap,Ai,Av)
                A = 2I + A + adj(A)
                test(A,conj,adj)
            end

            @testset "n = 2" begin
                @testset "A = [1 0; 0 1]" begin
                    Random.seed!(1)
                    Ap = [1,2,3]
                    Ai = [1,2]
                    Av = rand(T,2)
                    A = SparseMatrixCSC(2,2,Ap,Ai,Av)
                    A = 2I + A + adj(A)
                    test(A,conj,adj)
                end
                @testset "A = [1 1; 1 0]" begin
                    Ap = [1,3,3]
                    Ai = [1,2]
                    Av = rand(T,2)
                    A = SparseMatrixCSC(2,2,Ap,Ai,Av)
                    A = 2I + A + adj(A)
                    test(A,conj,adj)
                end
                @testset "A = [1 1; 1 1]" begin
                    Ap = [1,3,4]
                    Ai = [1,2,2]
                    Av = rand(T,3)
                    A = SparseMatrixCSC(2,2,Ap,Ai,Av)
                    A = 2I + A + adj(A)
                    test(A,conj,adj)
                end
            end

            @testset "n in 3:100" begin
                @testset for seed = 1:100
                    Random.seed!(seed)
                    n = rand(3:100)
                    A = sprand(n,n,rand())
                    A = 2I + A + adj(A)
                    test(A,conj,adj)
                end
            end
        end
    end
end

end # module