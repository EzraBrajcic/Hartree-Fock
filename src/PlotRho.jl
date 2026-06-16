using LinearAlgebra
using GLMakie
using CUDA

include("lib/HFSCF.jl")
include("lib/GPUPlotPoint.jl")
include("lib/Reader.jl")




function PlotElectronDensityVolume(Basis, P; range=(-3.0, 3.0), resolution=50, isovalue=0.01)
    # Creates a 3D volumetric visualization of electron density

    # Arguments:
    # `Basis`: BasisSet object containing the molecular orbital basis
    # `P`: Density matrix from SCF calculation
    # `range`: Tuple specifying the spatial range (min, max) in atomic units
    # `resolution`: Number of grid points along each axis
    # `isovalue`: Density threshold for isosurface rendering
    
    # Generate 3D grid
    coords = LinRange(range[1], range[2], resolution)
    
    X = [x for x in coords, y in coords, z in coords]
    Y = [y for x in coords, y in coords, z in coords]
    Z = [z for x in coords, y in coords, z in coords]
    
    X_flat = vec(X)
    Y_flat = vec(Y)
    Z_flat = vec(Z)
    NPoints = length(X_flat)
    
    println("Computing density on $(resolution)³ = $(NPoints) grid points...")
    
    # Pack primitives for GPU
    MaxPrims, PrimN, PrimL, PrimM, Primζ, PrimNorm, PrimCoeff, 
        PrimYNorm, PrimCount, MOcoeffs = PackPrimitives(Basis)
    
    NBasis = Basis.BasisSize
    
    # Transfer to GPU
    XDev = cu(X_flat)
    YDev = cu(Y_flat)
    ZDev = cu(Z_flat)
    AODev = CUDA.zeros(Float32, NBasis * NPoints)
    
    PrimNDev = cu(PrimN)
    PrimLDev = cu(PrimL)
    PrimMDev = cu(PrimM)
    PrimζDev = cu(Primζ)
    PrimNormDev = cu(PrimNorm)
    PrimCoeffDev = cu(PrimCoeff)
    PrimYNormDev = cu(PrimYNorm)
    PrimCountDev = cu(PrimCount)
    MOcoeffsDev = cu(MOcoeffs)
    
    # Compute AO values
    threads = 256
    total = NBasis * NPoints
    blocks = cld(total, threads)
    
    println("Launching GPU kernel with $(blocks) blocks, $(threads) threads...")
    
    @cuda threads=threads blocks=blocks KernelComputeAOWithAng!(
        AODev, XDev, YDev, ZDev,
        PrimNDev, PrimLDev, PrimMDev,
        PrimζDev, PrimNormDev, PrimCoeffDev, PrimYNormDev,
        PrimCountDev, MOcoeffsDev,
        Int32(NBasis), Int32(MaxPrims), Int32(NPoints)
    )
    
    # Compute density
    println("Computing electron density...")
    AOMatrix = reshape(Array(AODev), NBasis, NPoints)
    rho_flat = ComputeRhoOnGPU!(AOMatrix, P)
    
    # Reshape to 3D grid
    rho_volume = reshape(rho_flat, resolution, resolution, resolution)
    
    # Calculate voxel volume from actual grid spacing
    dx = coords[2] - coords[1]
    voxel_volume = dx^3
    
    # Multiply density by voxel volume for proper normalization
    rho_volume = rho_volume .* voxel_volume
    
    println("Density range: [$(minimum(rho_volume)), $(maximum(rho_volume))] a.u.")
    println("Voxel volume: $(voxel_volume) a.u.")
    println("Integrated density: $(sum(rho_volume)) electrons")

    # Create 3D visualization with volumetric rendering
    fig = Figure(size=(1440, 1440))
    ax = Axis3(fig[1, 1], 
               xlabel="x (a₀)", 
               ylabel="y (a₀)", 
               zlabel="z (a₀)",
               title="Electron Density Volume",
               aspect=(1, 1, 1))
    
    # Extract endpoints for volume specification
    x_range = (coords[1], coords[end])
    y_range = (coords[1], coords[end])
    z_range = (coords[1], coords[end])
    
    # Make a colormap, with the first value being transparent
    colormap = to_colormap(:curl)
    colormap[1] = RGBAf(0,0,0,0)

    rho_log = log.(rho_volume .+ 1e-10)

    # Use maximum intensity projection to show density variations with color
    vol = volume!(ax, x_range, y_range, z_range, rho_log,
                  algorithm=:mip,
                  colormap=colormap,
                  colorrange=(minimum(rho_log), maximum(rho_log)))
    
    Colorbar(fig[1, 2], vol, label="ln(ρ(r)) (a.u.)")
    
    
    println("Visualization complete!")
    
    return fig
end

function main()

    to = TimerOutput()
    spin = [1, -1]

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

    FinalEnergy, Ei, Count, FinalDelta, evecs, evals, JmK, P, S, S_Ortho, F, F_Ortho, C, to = SCF(Basis, Nelec, P_Old, 2.5e-20, 5000, to)

    set_theme!(theme_dark())
    
    fig = PlotElectronDensityVolume(Basis, P; range=(-12, 12), resolution=500)
    display(fig)
    save("Data/Output/Carbon(1S) 1 Electron Density.png", fig, update=false)
end

# Run main when executed
if !isdefined(Base, :test) 
    main() 
end

