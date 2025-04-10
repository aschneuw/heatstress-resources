# example_script.jl
# Import the necessary module for handling input arguments
using ArgParse
using Pkg
Pkg.develop(path="/home/aschneuwl/workspace/MixedModels.jl")
using MixedModels
using LinearAlgebra
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

function print_memory_usage_gb()
    # Get the current memory usage in bytes
    current_memory = Base.gc_bytes()
    
    # Convert to gigabytes
    current_memory_gb = current_memory / (1024^3)
    
    # Print the memory usage in GB
    println("Current memory usage: ", current_memory_gb, " GB")
end

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


function postprocess(input_json_fpath::String, model_fpath::String, post_data_json_fpath::String, output_fpath::String)
    println("Load model specifications $(input_json_fpath)")
    data = JSON.parsefile(input_json_fpath);

    println("Load fitted model  $(model_fpath)")
    @load model_fpath model
    
    n_regular_effects = length(data["random_effects_data"]["regular"]["group_names"])
    sigma_square = Float32.(varest(model));
    z = GBMatrix(Float32.(createZ(model.reterms[1:n_regular_effects])))
    phi = GBMatrix(Float32.(createRelCoFa(model.reterms[1:n_regular_effects])))
    model = nothing
    print_memory_usage_gb()

    V = GBMatrix(Float32.(Diagonal(sigma_square ./ vec(Matrix{Float32}(DataFrame(read_parquet_from_base64(data["weights"])))))))
    V = V + (((phi*z')'*(phi*z'))*(sigma_square))
    #VC = SparseMatrixCSC(V)
    # display(V)
    # empty!(V)
    # V = nothing
    # VCC = CHOLMOD.Sparse(VC)
    # VC = nothing
    # GC.gc()
    data = nothing
    empty!(z)
    empty!(phi)
    z = nothing
    phi = nothing
    # GC.gc()
    # V = VCC
    print_memory_usage_gb()

    println("Cholesky Decomposition of V")
    c = nothing
    try
        c = cholesky(V, check=false);
    catch e
        if isa(e, ArgumentError)
            println("Caught an ArgumentError: ", e.msg)
            println("Do (V+V')/2")
            V = (V + V')/2
            display(V)
            println("Redo Cholesky Decomposition of V")
            c = cholesky(V, check=false);
        else
            rethrow(e)
        end
    end
    V = nothing;
    GC.gc();
    display(c)
    print_memory_usage_gb()

    println("Load post processing data  $(post_data_json_fpath)")
    post_data = JSON.parsefile(post_data_json_fpath)
    postprocess_data = Dict()
    
    Xfp = SparseArrays.CHOLMOD.Dense(Matrix{Float32}(DataFrame(read_parquet_from_base64(post_data["Xfp"]))))
    WX = CHOLMOD.solve(CHOLMOD.CHOLMOD_L,c,Xfp)
    Xfp = nothing
    postprocess_data["WX"] = dataframe_to_base64(DataFrame(WX, :auto))
    WX = nothing

    Xf = SparseArrays.CHOLMOD.Dense(Matrix{Float32}(DataFrame(read_parquet_from_base64(post_data["Xf"]))))
    XVX = CHOLMOD.solve(CHOLMOD.CHOLMOD_L,c,Xf)
    Xf = nothing
    postprocess_data["XVX"] = dataframe_to_base64(DataFrame(XVX, :auto))
    XVX = nothing

    c = nothing
    GC.gc()
    print_memory_usage_gb()

    println("Store data to $(output_fpath)")
    open(output_fpath, "w") do file
        write(file, JSON.json(postprocess_data));
    end
    print_memory_usage_gb()

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
        "--post", "-p"
            help = "Postprocess Data from R"
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
    model_fpath = parsed_args["model"]
    post_data_fpath = parsed_args["post"]
    output_fpath = parsed_args["output"]
   
    postprocess(input_fpath, model_fpath, post_data_fpath, output_fpath)
end
BLAS.set_num_threads(32)
gbset(:nthreads, 32)
main()