include("selinv.jl")


"""
    selinv(F::SimplicialLDLt) -> B::SparseMatrixCSC

Compute the entries of the inverse of `A = F.L*F.D*F.Lt` contained in the
sparsity pattern of the (incomplete) LDLt factorization `F`.
"""
function selinv(F::SimplicialLDLt)
    Fp,Fi,Fv,conj = F.colptr,F.rowval,F.nzval,F.conj
    Bv = selinv_from_ldlt(Fp,Fi,Fv; conj=conj)
    return SparseMatrixCSC(size(F)...,Fp,Fi,Bv)
end
