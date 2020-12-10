module Test_iterate_jkp

include("../src/iterate_jkp.jl")

using Test
using Random
using SparseArrays

function test(F,top_k)
    n = size(F,1)
    sF = sparse(F)
    Fp,Fi = sF.colptr,sF.rowval
    F = Matrix(F)

    @testset "js" begin
        @test [j for (j,kps) in iterate_jkp(Fp,Fi,Val(top_k))] == (1:n)
    end

    @testset "ks" begin
        for (j,kps) in iterate_jkp(Fp,Fi,Val(top_k))
            ks = sort!([k for (k,ps) in kps])
            ref = findall(!iszero, F[j,1:j-1])
            top_k && filter!(k->all(F[k+1:j-1,k] .== 0), ref)
            @test ks == ref
        end
    end

    @testset "ps" begin
        for (j,kps) in iterate_jkp(Fp,Fi,Val(top_k))
            for (k,ps) in kps
                @test Fi[ps] == (j-1) .+ findall(!iszero, F[j:n,k])
            end
        end
    end
end

function bit_vector(nbits,i)
    b = Vector{Bool}(undef,nbits)
    for k = 1:nbits
        b[k] = i%2
        i = i >> 1
    end
    return b
end

@testset "iterate_jkp" begin
    @testset for top_k = (false,true)
        # Test all matrices with n <= 3
        @testset for n = 1:3
            @testset for i = 1:2^(n^2)
                F = reshape(bit_vector(n^2,i-1),(n,n))
                test(F,top_k)
            end
        end

        # Test a random sample of matrices with n > 3
        @testset "n in 4:100" begin
            @testset for seed = 1:100
                Random.seed!(seed)
                n = rand(4:100)
                F = sprand(n,n,rand())
                test(F,top_k)
            end
        end
    end
end

end # module