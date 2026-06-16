using TimerOutputs
using PrettyTables
using LinearAlgebra
using Plots

include("lib/HFSCF.jl")
include("lib/Reader.jl")

plotlyjs()

function main()

    to = TimerOutput()
    spin = [1, -1]

    sto_1s1 = GenerateSTO(1, 0, 0, 5.40758, 1.0)
    sto_1s2 = GenerateSTO(1, 0, 0, 9.48256, 1.0)
    sto_1s3 = GenerateSTO(1, 0, 0, 1.05749, 1.0)
    sto_1s4 = GenerateSTO(1, 0, 0, 1.52427, 1.0)
    sto_1s5 = GenerateSTO(1, 0, 0, 2.68435, 1.0)
    sto_1s6 = GenerateSTO(1, 0, 0, 4.20096, 1.0)
    sto_2s1 = GenerateSTO(2, 0, 0, 5.40758, 1.0)
    sto_2s2 = GenerateSTO(2, 0, 0, 9.48256, 1.0)
    sto_2s3 = GenerateSTO(2, 0, 0, 1.05749, 1.0)  
    sto_2s4 = GenerateSTO(2, 0, 0, 1.52427, 1.0)
    sto_2s5 = GenerateSTO(2, 0, 0, 2.68435, 1.0)
    sto_2s6 = GenerateSTO(2, 0, 0, 4.20096, 1.0)
    sto_2p1 = GenerateSTO(2, 1, 0, 0.98073, 1.0)
    sto_2p2 = GenerateSTO(2, 1, 0, 1.44361, 1.0)
    sto_2p3 = GenerateSTO(2, 1, 0, 2.60051, 1.0)
    sto_2p4 = GenerateSTO(2, 1, 0, 6.51003, 1.0)
    stos = [sto_1s1, sto_1s2, sto_1s3, sto_1s4, sto_1s5, sto_1s6, sto_2s1, sto_2s2, sto_2s3, sto_2s4, sto_2s5, sto_2s6, sto_2p1, sto_2p2, sto_2p3, sto_2p4]
    BasisFuncs = BasisFunction[]
    for i in 1:16
        bf = BasisFunction([stos[i]], 1.0)
        push!(BasisFuncs, bf)
    end
    Basis = BasisSet(BasisFuncs, 16, 6)
    Nelec = 6

    sto1 = GenerateSTO(1, 0, 0, 1.41714, 0.76838)
    sto2 = GenerateSTO(1, 0, 0, 2.37682, 0.22346)
    sto3 =  GenerateSTO(1, 0, 0, 4.39628, 0.04082)
    sto4 =  GenerateSTO(1, 0, 0, 6.52699, -0.00994)
    sto5 =  GenerateSTO(2, 0, 0, 7.94252, 0.00230)

    bf1 = BasisFunction([sto1, sto2, sto3, sto4, sto5], 1.0)

    Basis = BasisSet([bf1], 1, 2)  # He: Z = 2, BasisSize = 2
    Nelec = 2

    sto_1s1 = GenerateSTO(1, 0, 0, 3.47116, 0.91896)
    sto_1s2 = GenerateSTO(1, 0, 0, 6.36861, 0.08724)
    sto_1s3 = GenerateSTO(2, 0, 0, 0.77820, 0.00108)
    sto_1s4 = GenerateSTO(2, 0, 0, 0.94067, -0.00199)
    sto_1s5 = GenerateSTO(2, 0, 0, 1.48725, 0.00176)
    sto_1s6 = GenerateSTO(2, 0, 0, 2.71830, 0.00628)

    sto_2s1 = GenerateSTO(1, 0, 0, 3.47116, -0.17092)
    sto_2s2 = GenerateSTO(1, 0, 0, 6.36861, -0.01455)
    sto_2s3 = GenerateSTO(2, 0, 0, 0.77820, 0.21186)
    sto_2s4 = GenerateSTO(2, 0, 0, 0.94067, 0.62499)
    sto_2s5 = GenerateSTO(2, 0, 0, 1.48725, 0.26662)
    sto_2s6 = GenerateSTO(2, 0, 0, 2.71830, -0.09919)

    stos = [sto_1s1, sto_1s2, sto_1s3, sto_1s4, sto_1s5, sto_1s6, sto_2s1, sto_2s2, sto_2s3, sto_2s4, sto_2s5, sto_2s6]

    bf1 = BasisFunction([sto_1s1, sto_1s2, sto_1s3, sto_1s4, sto_1s5, sto_1s6], 1.0)
    bf2 = BasisFunction([sto_2s1, sto_2s2, sto_2s3, sto_2s4, sto_2s5, sto_2s6], 1.0)

    Basis = BasisSet([bf1, bf2], 2, 4)
    Nelec = 4

    sto_1s1 = GenerateSTO(1, 0, 0, 5.41189, 0.93414)
    sto_1s2 = GenerateSTO(1, 0, 0, 9.43757, 0.07195)
    sto_1s3 = GenerateSTO(2, 0, 0, 1.18421, 0.00168)
    sto_1s4 = GenerateSTO(2, 0, 0, 1.64360, -0.00376)
    sto_1s5 = GenerateSTO(2, 0, 0, 2.73028, 0.00863)
    sto_1s6 = GenerateSTO(2, 0, 0, 4.42841, -0.00304)

    sto_2s1 = GenerateSTO(1, 0, 0, 5.41189, -0.20593)
    sto_2s2 = GenerateSTO(1, 0, 0, 9.43757, -0.01375)
    sto_2s3 = GenerateSTO(2, 0, 0, 1.18421, 0.19836)
    sto_2s4 = GenerateSTO(2, 0, 0, 1.64360, 0.68142)
    sto_2s5 = GenerateSTO(2, 0, 0, 2.73028, 0.26711)
    sto_2s6 = GenerateSTO(2, 0, 0, 4.42841, -0.13084)

    sto_2p1 = GenerateSTO(2, 1, 0, 1.10539, 0.63602)
    sto_2p2 = GenerateSTO(2, 1, 0, 0.61830, 0.07877)
    sto_2p3 = GenerateSTO(2, 1, 0, 2.26857, 0.36369)
    sto_2p4 = GenerateSTO(2, 1, 0, 5.23303, 0.02063)

    bf1 = BasisFunction([sto_1s1, sto_1s2, sto_1s3, sto_1s4, sto_1s5, sto_1s6], 1.0)
    bf2 = BasisFunction([sto_2s1, sto_2s2, sto_2s3, sto_2s4, sto_2s5, sto_2s6], 1.0)
    bf3 = BasisFunction([sto_2p1, sto_2p2, sto_2p3, sto_2p4], 1.0)

    Basis = BasisSet([bf1, bf2, bf3], 3, 6)
    Nelec = 6

    F = zeros(Float64, Basis.BasisSize, Basis.BasisSize)
    F_Ortho = zeros(Float64, Basis.BasisSize, Basis.BasisSize)
    JmK = zeros(Float64, Basis.BasisSize, Basis.BasisSize)

    C = zeros(Float64, Basis.BasisSize, Basis.BasisSize)
    P = zeros(Float64, Basis.BasisSize, Basis.BasisSize)
    P_Old = zeros(Float64, Basis.BasisSize, Basis.BasisSize)

    # lists of previous Fock matrices and error matrices for DIIS
    FockList = Vector{Matrix{Float64}}()
    ErrorList = Vector{Matrix{Float64}}()

    evecs = zeros(Float64, Basis.BasisSize, 1)
    evals = zeros(Float64, Basis.BasisSize, 1)

    S = OverlapMatrix(Basis, to)
    S_Ortho = Orthogonalize(S, to)

    H = CoreHamiltonian(Basis, to)
    H_ortho = S_Ortho' * H * S_Ortho
    
    H_evals, H_evecs = eigen(Symmetric(H_ortho))
    C = S_Ortho * H_evecs
    PGuess = DensityMatrix(C, Nelec)
  
    #=
    println("Initial Eigenvectors:")
    pretty_table(H_evecs)

    println("Initial Coefficient Matrix:")
    pretty_table(C)

    println("Initial Density Matrix Guess:")
    pretty_table(PGuess)
    =#

    FinalEnergy, Ei, Count, FinalDelta, evecs, evals, JmK, P, S, S_Ortho, F, F_Ortho, C, to = SCF(Basis, Nelec, P_Old, 2.5e-20, 5000, to)

    print_timer(to)

    println("Final Energy (Hartrees): ", FinalEnergy)

    println("Final ΔP: ", FinalDelta)

    plt = Plots.scatter(1:Count, Ei[1:Count], xlabel="SCF Iteration #", ylabel="Energy (Hartrees)", title="SCF Convergence", legend=false)
    display(plt)
    
    println("Eigenvectors:")
    pretty_table(evecs')

    println("Eigenvalues:")
    pretty_table(evals)

    println("Final Coefficient Matrix:")
    pretty_table(C')
    println(C)

    println(tr(P))
    
end

if !isdefined(Base, :test)
    main()
end
