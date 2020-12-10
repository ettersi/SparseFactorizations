"""
    selinv_from_ldlt(Fp,Fi,Fv; conj = conj) -> Bv

Compute the entries `Bv` of the inverse of `A = F.L*F.D*F.Lt` contained in
the sparsity pattern of the (incomplete) LDLt factorization `F`.
"""
function selinv_from_ldlt(Fp,Fi,Fv; conj = Base.conj)
    Tv = eltype(Fv)
    n = length(Fp)-1

    @assert length(Fp) >= 1
    @assert issorted(Fp)
    @assert length(Fi) >= Fp[end]-1
    @assert length(Fv) >= Fp[end]-1
    @assert all(1 .<= Fi .<= n)

    @inbounds begin
        # Return variables
        Bv = Vector{Tv}(undef,length(Fi))

        # Workspace for a single column
        Fv_aj = Vector{Tv}(undef,n)
        Bv_aj = Vector{Tv}(undef,n)

        # Main algorithm
        for j in reverse(1:n)
            # Initialise column
            for p in Fp[j]+1:Fp[j+1]-1
                i = Fi[p]
                Fv_aj[i] = Fv[p]
                Bv_aj[i] = zero(Tv)
            end

            # Pull updates into B[j+1:n,j]
            for p in Fp[j]+1:Fp[j+1]-1
                k = Fi[p]
                Fv_kj = Fv_aj[k]
                Bv_kj = Bv_aj[k] - Bv[Fp[k]]*Fv_kj
                for p in Fp[k]+1:Fp[k+1]-1
                    i = Fi[p]
                    Fv_ij = Fv_aj[i]
                    Bv_ik = Bv[p]
                    Bv_aj[i] -=      Bv_ik *Fv_kj
                    Bv_kj    -= conj(Bv_ik)*Fv_ij
                end
                Bv_aj[k] = Bv_kj
            end

            # Copy temporary column into B
            for p in Fp[j]+1:Fp[j+1]-1
                i = Fi[p]
                Fv_aj[i] = zero(Tv)
                Bv[p] = Bv_aj[i]
            end

            # Deal with diagonal
            d = inv(Fv[Fp[j]])
            for p in Fp[j]+1:Fp[j+1]-1
                d -= conj(Bv[p])*Fv[p]
            end
            Bv[Fp[j]] = d
        end
    end
    return Bv
end