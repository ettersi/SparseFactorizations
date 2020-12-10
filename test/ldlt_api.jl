#=
The tests here merely check that the interface functions work.
Testing of the implementation is done in ldlt.jl.
=#

using Test
using Random
using SparseArrays
using LinearAlgebra
using SparseFactorizations

@testset "ldlt API" begin
    @testset "ldlt_structure" begin
        Random.seed!(1)
        A = sprand(10,10,0.1)
        A = I + A + A'
        Fref = lu(Matrix(A),Val(false))
        P = ldlt_structure(A)
        @test P == (Fref.L .!= 0)
    end

    @testset "level_of_fill" begin
        Random.seed!(1)
        A = sprand(10,10,0.1)
        A = I + A + A'
        Fref = lu(Matrix(A),Val(false))
        P = level_of_fill(A)
        nonzeros(P) .= 1
        @test P == (Fref.L .!= 0)
    end

    @testset "simplicial_ldlt" begin
        @testset for (conj,adj) in ((conj,adjoint),(identity,transpose))
            Random.seed!(1)
            A = sprand(ComplexF64,10,10,0.1)
            A = I + A + adj(A)
            Fref = lu(Matrix(A),Val(false))
            F = simplicial_ldlt(A; conj=conj)

            @test F.L ≈ Fref.L
            @test F.d ≈ diag(Fref.U)
            @test F.Lt == adj(F.L)
            @test F.D == Diagonal(F.d)
        end
    end

    @testset "ildlt" begin
        @testset for (conj,adj) in ((conj,adjoint),(identity,transpose))
            Random.seed!(1)
            A = sprand(ComplexF64,10,10,0.1)
            A = I + A + adj(A)
            Fref = lu(Matrix(A),Val(false))
            P = ldlt_structure(A)
            F = ildlt(A,P; conj=conj)

            @test F.L ≈ Fref.L
            @test F.d ≈ diag(Fref.U)
            @test F.Lt == adj(F.L)
            @test F.D == Diagonal(F.d)
        end
    end

    @testset "ildlt_lof" begin
        @testset for (conj,adj) in ((conj,adjoint),(identity,transpose))
            Random.seed!(1)
            A = sprand(ComplexF64,10,10,0.1)
            A = I + A + adj(A)
            Fref = lu(Matrix(A),Val(false))
            F = ildlt_lof(A,size(A,2); conj=conj)

            @test F.L ≈ Fref.L
            @test F.d ≈ diag(Fref.U)
            @test F.Lt == adj(F.L)
            @test F.D == Diagonal(F.d)
        end
    end

    @testset "ildlt_tol" begin
        @testset for (conj,adj) in ((conj,adjoint),(identity,transpose))
            Random.seed!(1)
            A = sprand(ComplexF64,10,10,0.1)
            A = I + A + adj(A)
            Fref = lu(Matrix(A),Val(false))
            F = ildlt_tol(A,0.0; conj=conj)

            @test F.L ≈ Fref.L
            @test F.d ≈ diag(Fref.U)
            @test F.Lt == adj(F.L)
            @test F.D == Diagonal(F.d)
        end
    end
end
