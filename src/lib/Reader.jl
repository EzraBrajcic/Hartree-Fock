include("DataTypes.jl")

function get_project_root()
    # Try environment variable first
    if haskey(ENV, "HARTREE_FOCK_PATH")
        return ENV["HARTREE_FOCK_PATH"]
    end
    
    # Try to find by looking for Project.toml
    current = pwd()
    while current != dirname(current)  # Stop at root
        if isfile(joinpath(current, "Project.toml"))
            return current
        end
        current = dirname(current)
    end
    
    # Fallback: assume pwd() is project root
    return pwd()
end

function GenerateSTO(n::Int, l::Int, m::Int, ζ::Float64, coeff::Float64)
    if l < 0 || l >= n
        throw(ArgumentError("Invalid quantum numbers: l must satisfy 0 ≤ l ≤ n - 1. Got n = $n, l = $l."))
    end
    normconst = (2 * ζ)^n * sqrt((2 * ζ) / (factorial(2 * n)))
    return STO(n, l, m, ζ, normconst, coeff)
end

function ReadAtomBFs(Element::String)
    project_root = get_project_root()
    basis_path = joinpath(project_root, "Data", "Input", "Basis Functions", "STOs")
    
    element_file = joinpath(basis_path, Element)
    if !isfile(element_file)
        throw(ArgumentError("Element file not found: $element_file"))
    end
    
    f = open("$element_file", "r")

    BFs = Vector{Vector{STO}}()
    NewOrbital = false
    CurrentOrbital = nothing

    for RawLine in eachline(f)
        line = chomp(RawLine)

        if isempty(strip(line))
            continue
        end

        if length(line) == 1
            if !isnothing(CurrentOrbital)
                append!(BFs, CurrentOrbital)
            end
            CurrentOrbital = Vector{Vector{STO}}()
            NewOrbital = true
            continue

        end

        # split the line into its components with tab (\t) delimiter
        data = split(line, '\t')

        # minimal validation
        len = length(data)
        if len < 5
            throw(ArgumentError("Data line has fewer than 5 tab-separated fields: $line"))
        end

        n = tryparse(Int, strip(data[1]))
        l = tryparse(Int, strip(data[2]))
        m = tryparse(Int, strip(data[3]))

        if any(x -> x === nothing, (n, l, m))
            throw(ArgumentError("Failed to parse integers for n, l, or m: $line"))
        end

        n = n::Int
        l = l::Int
        m = m::Int

        ζ = tryparse(Float64, strip(data[4]))
        
        if isnothing(ζ) == true
            throw(ArgumentError("Failed to parse float for ζ: $line"))
        end

        for i in 5:len
            coeff = tryparse(Float64, strip(data[i]))
            if isnothing(coeff) == true
                throw(ArgumentError("Failed to parse float for coefficient: $line"))
            end

            coeff = coeff::Float64
            if NewOrbital == true
                push!(CurrentOrbital, [GenerateSTO(n, l ,m, ζ, coeff)])
            else
                if i - 4 > length(CurrentOrbital)
                    throw(ArgumentError("Inconsistent number of coefficient columns in block: $line"))
                end
                push!(CurrentOrbital[i - 4], GenerateSTO(n, l, m, ζ, coeff))
            end
        end
        
        NewOrbital = false

    end

    # flush last block
    if !isnothing(CurrentOrbital)
        append!(BFs, CurrentOrbital)
    end

    close(f)
    return BFs

end
