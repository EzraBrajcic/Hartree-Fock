struct STO

    n::Int
    l::Int
    m::Int
    ζ::Float64
    normconst::Float64
    coeff::Float64

end

struct GTO

    n::Int
    l::Int
    m::Int
    ζ::Float64
    normconst::Float64
    coeff::Float64
end

struct BasisFunction
    STOs::Vector{STO}
    MO_Coeff::Float64
end

struct BasisSet
    BasisFunctions::Vector{BasisFunction}
    BasisSize::Int
    z::Int
end

struct Atom
    BasisFunctions::Vector{BasisFunction}
    pos::Vector{Float64}
end

struct MolecularBasisSet
    Atoms::Vector{Atom}
    BasisSize::Int
    z::Int
end