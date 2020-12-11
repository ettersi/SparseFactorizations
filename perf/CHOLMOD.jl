module CHOLMOD

module Implementation

using SuiteSparse
using SuiteSparse.CHOLMOD: SuiteSparse_long
const SC = SuiteSparse.CHOLMOD

using Setfield

struct C_cholmod_method_struct
    lnz::Cdouble
    fl::Cdouble
    prune_dense::Cdouble
    prune_dense2::Cdouble
    nd_oksep::Cdouble
    other_1::NTuple{4,Cdouble}
    nd_small::Csize_t
    other_2::NTuple{4,Csize_t}
    aggressive::Cint
    order_for_lu::Cint
    nd_compress::Cint
    nd_camd::Cint
    nd_components::Cint
    ordering::Cint
    other_3::NTuple{4,Csize_t}
end

mutable struct C_Common
    dbound::Cdouble
    grow0::Cdouble
    grow1::Cdouble
    grow2::Csize_t
    maxrank::Csize_t
    supernodal_switch::Cdouble
    supernodal::Cint
    final_axis::Cint
    final_super::Cint
    final_ll::Cint
    final_pack::Cint
    final_monotonic::Cint
    final_resymbol::Cint
    zrelax::NTuple{3,Cdouble}
    nrelax::NTuple{3,Csize_t}
    prefer_zomplex::Cint
    prefer_upper::Cint
    quick_return_if_not_posdef::Cint
    prefer_binary::Cint
    print::Cint
    precise::Cint
    try_catch::Cint
    error_handle::Ptr{Nothing}
    nmethods::Cint
    current::Cint
    selected::Cint
    method::NTuple{10,C_cholmod_method_struct}
    postorder::Cint
    default_nesdis::Cint
    metis_memory::Cdouble
    metis_dswitch::Cdouble
    metis_nswitch::Csize_t
    nrow::Csize_t
    mark::SuiteSparse_long
    iworksize::Csize_t
    xworksize::Csize_t
    Flag::Ptr{Nothing}
    Head::Ptr{Nothing}
    Xwork::Ptr{Nothing}
    Iwork::Ptr{Nothing}
    itype::Cint
    dtype::Cint
    no_workspace_reallocate::Cint
    status::Cint
    fl::Cdouble
    lnz::Cdouble
    anz::Cdouble
    modfl::Cdouble
    malloc_count::Csize_t
    memory_usage::Csize_t
    memory_inuse::Csize_t
    nrealloc_col::Cdouble
    nrealloc_factor::Cdouble
    nbounds_hit::Cdouble
    rowfacfl::Cdouble
    aatfl::Cdouble
    called_nd::Cint
    blas_ok::Cint
    SPQR_grain::Cdouble
    SPQR_small::Cdouble
    SPQR_shrink::Cint
    SPQR_nthreads::Cint
    SPQR_flopcount::Cdouble
    SPQR_analyze_time::Cdouble
    SPQR_factorize_time::Cdouble
    SPQR_solve_time::Cdouble
    SPQR_flopcount_bound::Cdouble
    SPQR_tol_used::Cdouble
    SPQR_norm_E_fro::Cdouble
    SPQR_istat::NTuple{10,SuiteSparse_long}
    useGPU::Cint
    maxGpuMemBytes::Csize_t
    maxGpuMemFraction::Cdouble
    gpuMemorySize::Csize_t
    gpuKernelTime::Cdouble
    gpuFlops::SuiteSparse_long
    gpuNumKernelLaunches::Cint
    gpuStream::NTuple{8,Ptr{Nothing}}
    cublasEventPotrf::NTuple{3,Ptr{Nothing}}
    updateCKernelsComplete::Ptr{Nothing}
    updateCBuffersFree::NTuple{8,Ptr{Nothing}}
    dev_mempool::Ptr{Nothing}
    dev_mempool_size::Csize_t
    host_pinned_mempool::Ptr{Nothing}
    host_pinned_mempool_size::Csize_t
    devBuffSize::Csize_t
    ibuffer::Cint
    syrkStart::Cdouble
    cholmod_cpu_gemm_time::Cdouble
    cholmod_cpu_syrk_time::Cdouble
    cholmod_cpu_trsm_time::Cdouble
    cholmod_cpu_potrf_time::Cdouble
    cholmod_gpu_gemm_time::Cdouble
    cholmod_gpu_syrk_time::Cdouble
    cholmod_gpu_trsm_time::Cdouble
    cholmod_gpu_potrf_time::Cdouble
    cholmod_assemble_time::Cdouble
    cholmod_assemble_time2::Cdouble
    cholmod_cpu_gemm_calls::Csize_t
    cholmod_cpu_syrk_calls::Csize_t
    cholmod_cpu_trsm_calls::Csize_t
    cholmod_cpu_potrf_calls::Csize_t
    cholmod_gpu_gemm_calls::Csize_t
    cholmod_gpu_syrk_calls::Csize_t
    cholmod_gpu_trsm_calls::Csize_t
    cholmod_gpu_potrf_calls::Csize_t
    padding::Csize_t

    C_Common() = new()
end

const common = C_Common()
const SC_common = unsafe_wrap(Array,reinterpret(Ptr{UInt8},pointer_from_objref(common)),SC.common_size)

const CHOLMOD_NATURAL = 0
const CHOLMOD_GIVEN = 1
const CHOLMOD_AMD = 2
const CHOLMOD_SIMPLICAL = 0
const CHOLMOD_AUTO = 1
const CHOLMOD_SUPERNODAL = 2

function __init__()
    SC.start(SC_common)
    common.print = 0
end

"""
    amd(A) -> p

Compute the approximate minimum degree (AMD) permutation of `A`.
"""
function amd end
amd(A) = amd(SC.Sparse(A,-1))
function amd(A::SC.Sparse)
    common.nmethods = 1
    common.method = set(common.method, @lens(_[1].ordering), CHOLMOD_AMD)
    common.postorder = true
    F = SC.analyze(A,SC_common)
    SC.get_perm(F)
end

"""
    simplicial_factorize(A) -> F

Compute the Cholesky or LDLt factorization of `A` using a simplicial
algorithm.
"""
function simplicial_factorize end
simplicial_factorize(A) = simplicial_factorize(SC.Sparse(A,-1))
function simplicial_factorize(A::SC.Sparse)
    common.nmethods = 1
    common.method = set(common.method, @lens(_[1].ordering), CHOLMOD_NATURAL)
    common.postorder = false
    common.supernodal = CHOLMOD_SIMPLICAL
    n = size(A,2)
    F = SC.analyze(A, SC_common)
    SC.factorize!(A,F, SC_common)
    return F
end

"""
    supernodal_factorize(A) -> F

Compute the Cholesky or LDLt factorization of `A` using a supernodal
algorithm.
"""
function supernodal_factorize end
supernodal_factorize(A) = supernodal_factorize(SC.Sparse(A,-1))
function supernodal_factorize(A::SC.Sparse)
    common.nmethods = 1
    common.method = set(common.method, @lens(_[1].ordering), CHOLMOD_NATURAL)
    common.postorder = false
    common.supernodal = CHOLMOD_SUPERNODAL
    n = size(A,2)
    F = SC.analyze_p(A, collect(1:n), SC_common)
    SC.factorize!(A,F, SC_common)
    return F
end

end  # module Implementation



module Test

using Test
using Random
using SparseArrays
using LinearAlgebra

using ..Implementation: amd, simplicial_factorize, supernodal_factorize

function test()
    @testset "CHOLMOD" begin
        Random.seed!(42)
        n = 100
        A = sprand(n,n,0.1)
        A = 100I + A + A'
        x = rand(n)
        b = A*x

        @testset "amd" begin
            p = amd(A)
            @test length(p) == n
            @test isperm(p)
        end

        @testset "simplicial_factorize" begin
            F = simplicial_factorize(A)
            @test F.p == 1:n
            @test F\b ≈ x
            @test unsafe_load(pointer(F)).is_super == 0
        end

        @testset "supernodal_factorize" begin
            F = supernodal_factorize(A)
            @test F.p == 1:n
            @test F\b ≈ x
            @test unsafe_load(pointer(F)).is_super == 1
        end
    end
end

end # module Test

using SuiteSparse
using SuiteSparse.CHOLMOD: Sparse
using .Implementation: amd, simplicial_factorize, supernodal_factorize
using .Test: test

export  amd, simplicial_factorize, supernodal_factorize

end