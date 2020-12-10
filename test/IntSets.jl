module TestIntSets

include("../src/IntSets.jl")

using Test
using Random

@testset "IntSet" begin
    function test_state(s::IntSet)
        n = limit(s)
        cs = collect(s)
        @test allunique(cs)
        @test length(cs) == length(s)
        @test in.(1:n,Ref(cs)) == in.(1:n,Ref(s))
    end

    @testset "IntSet()" begin
        @testset for n = 1:5
            s = IntSet(n)
            test_state(s)
            @test length(s) == 0
        end
    end

    @testset "push!" begin
        @testset for n = 1:5
            @testset for seed = 1:10
                Random.seed!(seed)
                vs = rand(1:n,10)
                s = IntSet(n)
                for k = 1:length(vs)
                    push!(s,vs[k])
                    test_state(s)
                    @test issetequal(s,vs[1:k])
                end
            end
        end

        @testset "inbounds" begin
            s = IntSet(3); @test_throws Exception push!(s,4)
        end
    end

    @testset "union!" begin
        @testset for n = 1:5
            @testset for seed = 1:10
                Random.seed!(seed)
                vs = rand(1:n,10)
                p = min.(10,cumsum([1;rand(0:3,5)]))
                s = IntSet(n)
                for k = 1:length(p)-1
                    union!(s,vs[p[k]:p[k+1]-1])
                    test_state(s)
                    @test issetequal(s,vs[1:p[k+1]-1])
                end
            end
        end

        @testset "inbounds" begin
            s = IntSet(3); @test_throws Exception union!(s,[4])
        end
    end

    @testset "empty!" begin
        s = IntSet(1)
        empty!(s)
        test_state(s)
        @test length(s) == 0
        push!(s,1)
        empty!(s)
        test_state(s)
        @test length(s) == 0
    end

    @testset "sort!" begin
        @testset for n = 1:5
            @testset for seed = 1:10
                Random.seed!(42)
                vs = rand(1:n,rand(1:n))
                s = IntSet(n)
                union!(s,vs)
                sort!(s)
                test_state(s)
                @test collect(s) == unique!(sort!(vs))
            end
        end
    end

    @testset "IntSet(0)" begin
        s = IntSet(0)
        test_state(s)
        empty!(s)
        test_state(s)
    end
end

end # module