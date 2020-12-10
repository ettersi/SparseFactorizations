module Test_ldlt

include("../src/ldlt.jl")

using Test
using Random
using SparseArrays
using LinearAlgebra

@testset "ldlt" begin
    @testset "symbolic_ldlt" begin
        @testset "n = 1" begin
            Ap = [1,2]
            Ai = [1]
            Fp,Fi = symbolic_ldlt(Ap,Ai)
            @test Fp == [1,2]
            @test Fi == [1]
        end

        @testset "n = 2" begin
            @testset "A = [1 0; 0 1]" begin
                Ap = [1,2,3]
                Ai = [1,2]
                Fp,Fi = symbolic_ldlt(Ap,Ai)
                @test Fp == [1,2,3]
                @test Fi == [1,2]
            end
            @testset "A = [1 1; 1 0]" begin
                Ap = [1,3,3]
                Ai = [1,2]
                Fp,Fi = symbolic_ldlt(Ap,Ai)
                @test Fp == [1,3,4]
                @test Fi == [1,2,2]
            end
            @testset "A = [1 1; 1 1]" begin
                Ap = [1,3,4]
                Ai = [1,2,2]
                Fp,Fi = symbolic_ldlt(Ap,Ai)
                @test Fp == [1,3,4]
                @test Fi == [1,2,2]
            end
        end

        @testset "n in 3:100" begin
            @testset for seed = 1:100
                Random.seed!(seed)
                n = rand(3:100)
                A = sprand(n,n,rand())
                A = 2I + A + A'
                Ap,Ai = A.colptr,A.rowval
                Fref = lu(Matrix(A),Val(false)).L .!= 0
                Fp,Fi = symbolic_ldlt(Ap,Ai)
                F = SparseMatrixCSC(n,n,Fp,Fi,ones(Bool,length(Fi)))
                @test F == Fref
            end
        end
    end

    @testset "symbolic_ildlt_lof" begin
        @testset "c = 0" begin
            @testset for seed = 1:100
                Random.seed!(seed)
                n = rand(3:100)
                A = sprand(n,n,rand())
                A = tril(2I + A + A')
                Ap,Ai = A.colptr,A.rowval

                Fp,Fi,Fl = symbolic_ildlt_lof(Ap,Ai,0)
                @test (Fp,Fi) == (Ap,Ai)
            end
        end

        @testset "c = n" begin
            @testset for seed = 1:100
                Random.seed!(seed)
                n = rand(3:100)
                A = sprand(n,n,rand())
                A = 2I + A + A'
                Ap,Ai = A.colptr,A.rowval

                rFp,rFi = symbolic_ldlt(Ap,Ai)
                Fp,Fi,Fl = symbolic_ildlt_lof(Ap,Ai,n)
                @test (Fp,Fi) == (rFp,rFi)
            end
        end

        @testset "self-consistency" begin
            @testset for seed = 1:100
                Random.seed!(seed)
                n = rand(3:100)
                A = sprand(n,n,rand())
                A = 2I + A + A'
                Ap,Ai = A.colptr,A.rowval


                Fref = lu(Matrix(A),Val(false)).L .!= 0
                Fp,Fi,Fl = symbolic_ildlt_lof(Ap,Ai,n)
                F = SparseMatrixCSC(n,n,Fp,Fi,Fl)

                cmax = maximum(Fl)
                for c = 0:cmax
                    ref = SparseArrays.fkeep!(copy(F),(i,j,l)->(l<=c))
                    rFp,rFi,rFl = ref.colptr,ref.rowval,ref.nzval
                    iFp,iFi,iFl = symbolic_ildlt_lof(Ap,Ai,c)
                    @test (iFp,iFi,iFl) == (rFp,rFi,rFl)
                end
            end
        end
    end

    @testset "numeric_ldlt" begin
        function test(A,conj,adj)
            Ap,Ai,Av = A.colptr,A.rowval,A.nzval
            Fp,Fi = symbolic_ldlt(Ap,Ai)
            Fv = numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj=conj)
            F = SparseMatrixCSC(size(A)...,Fp,Fi,Fv)

            L = tril(F,-1) + I; D = Diagonal(F);
            @test eltype(F) == eltype(A)
            @test L*D*adj(L) ≈ A
        end

        @testset for T in (Float32,Float64,ComplexF32,ComplexF64)
            @testset for (conj,adj) in ((conj,adjoint),(identity,transpose))
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

    @testset "ildlt_tol" begin
        function test(A,conj,adj)
            A = tril(A)
            Ap,Ai,Av = A.colptr,A.rowval,A.nzval

            # Test that factorization is exact if tol = 0
            Fp,Fi,Fv = ildlt_tol(Ap,Ai,Av, 0.0; conj=conj)
            F = SparseMatrixCSC(size(A)...,Fp,Fi,Fv)
            L = tril(F,-1) + I; D = Diagonal(F);
            @test eltype(F) == eltype(A)
            @test tril(L*D*adj(L)) ≈ A

            # Test that factorization produces no fill-in for tol = Inf
            Fp,Fi,Fv = ildlt_tol(Ap,Ai,Av, Inf; conj=conj)
            @test (Fp,Fi) == (Ap,Ai)
        end

        @testset for T in (Float32,Float64,ComplexF32,ComplexF64)
            @testset for (conj,adj) in ((conj,adjoint),(identity,transpose))
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
end

end