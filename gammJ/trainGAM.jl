# example_script.jl
# Import the necessary module for handling input arguments
using ArgParse
using Pkg
Pkg.develop(path="/home/aschneuwl/workspace/MixedModels.jl")
using MixedModels
using LinearAlgebra
BLAS.set_num_threads(32)
using Parquet2, Tables, DataFrames
using DataFrames
using StatsModels
using ProgressMeter
using Random
using DataFrames
using StatsBase:sample
using DisplayAs
using SparseArrays
import SparseArrays: CHOLMOD
using SuiteSparseGraphBLAS
using JSON
using Base64
using Arrow
using JLD2
using Statistics

function read_parquet_from_base64(base64_string::String)
    # Decode the base64 string into binary data
    binary_data = Base64.base64decode(base64_string)
    
    # Create an in-memory buffer from the binary data
    io_buffer = IOBuffer(binary_data)
    
    # Read the Parquet file from the in-memory buffer
    parquet_data = Parquet2.Dataset(io_buffer)

    tbl = Tables.columntable(parquet_data);

    return tbl
end

function dataframe_to_base64(df::DataFrame)
    # Create an IOBuffer to hold the Parquet data
    buffer = IOBuffer()
    
    # Write the DataFrame to the Parquet file format and write it to the buffer
    Parquet2.writefile(buffer, df, compression_codec=:snappy)
    
    # Get the data from the buffer
    parquet_data = take!(buffer)
    
    # Encode the Parquet data as a base64 string
    base64_string = base64encode(parquet_data)
    
    return base64_string
end

function ranef_var(model::LinearMixedModel{T}) where {T}
    # Compute the total variance of the random effects
    ranef_total_var = 0
    varcorr = VarCorr(model)
    for group_name in keys(varcorr.σρ)
        cols = varcorr.σρ[group_name].σ
        for col in keys(cols)
            ranef_total_var += cols[col]^2
        end
    end
    return ranef_total_var
end

function create_random_intercept(data::Vector{String}, group_name::String)
    group_ids = sort(collect(Set(data)))
    tbl = NamedTuple([Pair(Symbol(group_name), data)])
    rhs = CategoricalTerm(Symbol(group_name), StatsModels.ContrastsMatrix(Grouping(), group_ids))
    lhs = MatrixTerm((InterceptTerm{true}(),))
    
    trm = rhs 
    refs, levels = MixedModels._ranef_refs(trm, tbl) # 
    cnames = coefnames(lhs)
    z = Matrix(transpose(modelcols(lhs, tbl)))
    T = eltype(z)
    S = size(z, 1) # number of rows of z
    inds = sizehint!(Int[], (S * (S + 1)) >> 1)
    m = reshape(1:abs2(S), (S, S))
    for j in 1:S, i in j:S
        push!(inds, m[i, j])
    end

    DefaultReMat{T,S}(
        rhs, #
        refs,
        levels,
        isa(cnames, String) ? [cnames] : collect(cnames),
        z,
        z,
        MixedModels.LowerTriangular(Matrix{T}(MixedModels.I, S, S)),
        inds,
        MixedModels.adjA(refs, z),
        Matrix{T}(undef, (S, length(levels))),
    )
end

function conditional_r_square(model::LinearMixedModel{T}) where {T}
    # variance explained by fixed effects and random effects
    var_x = var(model.X*fixef(model)) # fixed effect variance
    var_epsilon = varest(model) # residual variance
    var_random = ranef_var(model) # total random effect variance (sum of the individual RE variance components)

    return (var_x + var_random) / (var_x + var_random + var_epsilon)
    
end

function marginal_r_square(model::LinearMixedModel{T}) where {T}
    # variance explained by fixed factors
    var_x = var(model.X*fixef(model)) # fixed effect variance
    var_epsilon = varest(model) # residual variance
    var_random = ranef_var(model) # total random effect variance (sum of the individual RE variance components)

    (var_x) / (var_x + var_random + var_epsilon)
end

function create_random_intercept(data::Vector{String}, group_name::String)
    group_ids = sort(collect(Set(data)))
    tbl = NamedTuple([Pair(Symbol(group_name), data)])
    rhs = CategoricalTerm(Symbol(group_name), StatsModels.ContrastsMatrix(Grouping(), group_ids))
    lhs = MatrixTerm((InterceptTerm{true}(),))
    
    trm = rhs 
    refs, levels = MixedModels._ranef_refs(trm, tbl) # 
    cnames = coefnames(lhs)
    z = Matrix(transpose(modelcols(lhs, tbl)))
    T = eltype(z)
    S = size(z, 1) # number of rows of z
    inds = sizehint!(Int[], (S * (S + 1)) >> 1)
    m = reshape(1:abs2(S), (S, S))
    for j in 1:S, i in j:S
        push!(inds, m[i, j])
    end

    DefaultReMat{T,S}(
        rhs, #
        refs,
        levels,
        isa(cnames, String) ? [cnames] : collect(cnames),
        z,
        z,
        MixedModels.LowerTriangular(Matrix{T}(MixedModels.I, S, S)),
        inds,
        MixedModels.adjA(refs, z),
        Matrix{T}(undef, (S, length(levels))),
    )
end

function create_smooth_intercept(data::Vector{String}, group_name::String, cname::String, smooth_data::Matrix{Float64})
    group_ids = sort(collect(Set(data)))
    tbl = NamedTuple([Pair(Symbol(group_name), data)])
    rhs = CategoricalTerm(Symbol(group_name), StatsModels.ContrastsMatrix(Grouping(), group_ids))
    lhs = MatrixTerm((InterceptTerm{true}(),))
    trm = rhs
    n_rows, n_cols = size(smooth_data)
    refs, levels = MixedModels._ranef_refs(trm, tbl) # 
    z = transpose(smooth_data)
    T = eltype(z)
    
    
    sparse_z = SparseMatrixCSC(z)
    cnames = [cname]
    inds = [1]

    FlexReMat{T}(
        rhs,
        refs,
        levels,
        cnames,
        z,
        z,
        MixedModels.LowerTriangular(Matrix{T}(MixedModels.I, 1, 1)),
        inds,
        sparse_z,
        Matrix{T}(undef, (1, length(levels))),
    )
end

function createRelCoFa(reterms::Vector{<:AbstractReMat{T}}) where {T}
    factors = Vector{SparseMatrixCSC}(undef, length(reterms))
    
    for (i, term) in pairs(reterms)
        matrices = SparseMatrixCSC.([term.λ for _ in 1:length(term.levels)])
        bd = blockdiag(matrices...)
        factors[i] = bd
    end
    
    bd = blockdiag(factors...)
    return bd
end

function createZ(reterms::Vector{<:AbstractReMat{T}}) where {T}
    blocks = Vector{SparseMatrixCSC}(undef, length(reterms))
    
    for (i, term) in pairs(reterms)
        blocks[i] = SparseMatrixCSC(term.adjA')
    end
    # convert(SparseMatrixCSC{Int64, Int64}, A)
    bd = convert(SparseMatrixCSC{T, Int64}, hcat(blocks...))
    return bd
end


function buildModel(input_json_fpath::String)
    data = JSON.parsefile(input_json_fpath)

    y = Float64.(read_parquet_from_base64(data["y"]).y)

    xf = read_parquet_from_base64(data["fixed_effect_data"]["x"])
    xf_cnames = String.(data["fixed_effect_data"]["cnames"])

    fe_term = MixedModels.FeTerm(Float64.(SparseMatrixCSC(hcat(xf...))), xf_cnames)

    re_mats = AbstractReMat{Float64}[]

    # regular random effects
    for g in String.(data["random_effects_data"]["regular"]["group_names"])
        effect_data = Vector(String.(read_parquet_from_base64(data["random_effects_data"]["regular"]["groups"][g])[Symbol(g)]))
        
        re_mat = create_random_intercept(effect_data, g)
        push!(re_mats, re_mat)
    end

    # smooth term random effects
    for g in String.(data["random_effects_data"]["smooths"]["group_names"])
        effect_data = Vector(String.(read_parquet_from_base64(data["random_effects_data"]["smooths"]["groups"][g])[Symbol(g)]))
        z = Float64.(Matrix(DataFrame(read_parquet_from_base64(data["random_effects_data"]["smooths"]["z"][g]))))
        z_cname = data["random_effects_data"]["smooths"]["cnames"][g]
        #re_mat = create_random_intercept(effect_data, g)
        re_mat = create_smooth_intercept(effect_data, g, z_cname, z)
        
        ## Replace random effect Z with the one generated by smooth2random in R
        #re_mat.adjA = adjoint(SparseMatrixCSC(Float64.(Matrix(DataFrame(read_parquet_from_base64(data["random_effects_data"]["smooths"]["z"][g]))))))
        push!(re_mats, re_mat)
    end

    fm = @formula(milk ~ x)
    model = LinearMixedModel(y,fe_term, re_mats, fm)

    return model
end

function fit_model(model::LinearMixedModel, model_fpath::String, output_fpath::String)
    # Fit Model
    fit!(model, progress=true, REML=true)
    @save model_fpath model
    display(model)
    display(model.optsum)

    println("Store fitted model to $(model_fpath)")

    markdown_fpath = join(split(model_fpath, ".")[1:end-1], ".") * "_summary.markdown"
    open(markdown_fpath, "w") do io
        show(io, MIME("text/markdown"), model)
    end

    latex_fpath = join(split(model_fpath, ".")[1:end-1], ".") * "_summary.latex"
    open(latex_fpath, "w") do io
        show(io, MIME("text/latex"), model)
    end

    # Extract and Store Parameters
    betas = fixef(model);
    sigma_square = varest(model);
    thetas = MixedModels.getθ(model);
    fitted_vals = fitted(model);
    ranefs = ranef(model);
    varcorr = VarCorr(model);
    sigma = varcorr.s;
    res = residuals(model);
    obj = objective(model);

    julia_model_data = Dict();

    # Sigma
    julia_model_data["sigma"] = sigma
    julia_model_data["sigma_square"] = sigma_square

    # Betas
    df = DataFrame(betas = betas)
    julia_model_data["betas"] = dataframe_to_base64(df);

    # Fitted
    df = DataFrame(fitted = fitted_vals)
    julia_model_data["fitted"] = dataframe_to_base64(df)

    # Random Effects
    julia_model_data["random_effects"] = Dict();
    for (i, term) in pairs(model.reterms)
        group_name = string(term.trm.sym)
        levels = term.levels
        df = DataFrame()
        df[!, Symbol(group_name)] = levels
        for (j, col) in pairs(term.cnames)
            df[!, Symbol(col)] = ranefs[i]'[:,j]
        end
        julia_model_data["random_effects"][group_name] = dataframe_to_base64(df)
    end

    # VarrCorr
    varcorr_dict = Dict()
    for group_name in keys(varcorr.σρ)
        varcorr_dict[string(group_name)] = Dict()
        cols = varcorr.σρ[group_name].σ
        for col in keys(cols)
            varcorr_dict[string(group_name)][string(col)] = cols[col]
        end
    end

    julia_model_data["varcorr"] = varcorr_dict

    df = DataFrame(residuals = res)
    julia_model_data["residuals"] = dataframe_to_base64(df)

    julia_model_data["objective"] = obj

    marginal_r = marginal_r_square(model)
    conditional_r = conditional_r_square(model)
    julia_model_data["marginal_r_square"] = marginal_r
    julia_model_data["conditional_r_square"] = conditional_r

    println("Conditional R Square: $(conditional_r)")
    println("Marginal R Square: $(marginal_r)")

    println("Store relevent parameters to $(output_fpath)")
    open(output_fpath, "w") do file
        write(file, JSON.json(julia_model_data))
    end

end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--data", "-d"
            help = "Input Data Generated in R"
            required = true
            arg_type = String
        "--output", "-o"
            help = "Putput Data Generated for R"
            required = true
            arg_type = String
        "--model", "-m"
            help = "Fitted Model"
            required = true
            arg_type = String
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end

    input_fpath = parsed_args["data"]
    model = buildModel(input_fpath)

    model_fpath = parsed_args["model"]
    output_fpath = parsed_args["output"]
   
    fit_model(model, model_fpath, output_fpath)
end

main()