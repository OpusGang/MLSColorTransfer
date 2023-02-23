module MLSColorTransfer
# https://openaccess.thecvf.com/content_cvpr_2014/papers/Hwang_Color_Transfer_Using_2014_CVPR_paper.pdf
# https://joonyoung-cv.github.io/assets/paper/19_cviu_probabilistic_moving.pdf
# the second paper adds usage of spatial info but doesn't explain some things
# for extrapolation: https://www.cs.unc.edu/~welch/media/pdf/Ilie2005_Calib.pdf
using Images,
    LinearAlgebra, ProgressBars, ImageFiltering, ArgParse, NearestNeighbors, Printf
import Base.round

#=
# mapping probability doesn't usually matter much because we can just align frames ourselves
# we never actually transfer between different angles so I'm just not gonna finish implementing
# the only use case would be for grain but I think prefiltering with a blur makes more sense anyway
#
# mapping probability:
# for a single mapping with i, j being RGB sets:
#   p(src(i), ref(j)) = #matches(i, j) / #matches(i)
#
# create 20⨯20⨯20 bins filling up the color space
#
# over whole image:
#   p(M(src(i), ref(j))) = p(src(i), ref(i)) ^ 2 / Σ p(src(i), ref(k)) Σ p(src(k), ref(j))
#   where Σ over k = 1:number of dims and indexing k means a value within the k-th bin 
#   the paper is very unclear as to how ref(k)/src(k) are meant to be treated
#
#
# this changes the weighting to:
#   w[k] = 1 / (|u[k] - x| ^ 2 α + ϵ) ⨯ p(M(src[i], ref[j]))
#   # recommend ϵ = 1
#   for spatial info:
#     w[k] = exp(-0.5 ⨯ (|P_u[k] - P_x| ^ 2 / σ_s ^ 2 + |u[k] - x| ^ 2 / σ_r ^ 2)) ⨯ ...
#
#   the σs should be parameters defaulting to 20, recommend σ_s ≦ 20, σ_r ≦ 80
#
# extrapolation:
# if no point in bin (can just do nn of cube center > distance to cube edge away) add extrapolated value
# this won't be exactly the same but it should be easier to compute
# extrapolation calculation:
#  out_c = inv(B) ⨯ in_c
#  B is an m×7 matrix consisting of [input_r input_r^2 input_g input_g^2 input_b input_b^2 1] of all samples
=#

"""
    getcuboidcenters(binsize, maxval=256)

Get centers of cuboid bins inside a cube.
"""
function getcuboidcenters(binsize, maxval = 256)
    nperdim = trunc(Int, cld(maxval, binsize))
    cubes = Vector{Vector{Float64}}(undef, (nperdim)^3)
    vals = binsize/2:binsize:maxval
    l = 1
    for k in vals
        kval = k
        for j in vals
            jval = j
            for i in vals
                cubes[l] = [i, jval, kval]
                l += 1
            end
        end
    end
    cubes
end

"""
    extrapolatepoints(src, ref, maxval, binsize=20)

Extrapolate points missing in sphere around bins using second order polynomial transform.
"""
function extrapolatepoints(src, ref, maxval, binsize = 20)
    centers = getcuboidcenters(binsize, maxval)
    mat = reduce(hcat, src)
    B = [
        [src[i][1], src[i][1]^2, src[i][2], src[i][2]^2, src[i][3], src[i][3]^3, 1] for
        i = 1:length(src)
    ]
    Binv = pinv(reduce(hcat, B))
    T = [[ref[j][i] for j = 1:length(ref)] for i = 1:3]
    coeffs = [permutedims(Binv) * T[i] for i = 1:3]
    kdtree = KDTree(mat)
    changes = Vector{Vector{Float64}}[]
    Threads.@threads for c in centers
        # check if there's a value within a circle around bin center
        if nn(kdtree, c)[1] > binsize
            newB = [c[1], c[1]^2, c[2], c[2]^2, c[3], c[3]^2, 1]
            push!(changes, [c, vcat([permutedims(newB) * coeffs[i] for i = 1:3]...)])
        end
    end
    println("$(length(changes))/$(length(centers)) bins interpolated")
    for c in changes
        push!(src, c[1])
        push!(ref, c[2])
    end
    src, ref
end

"""
    getmatindex(i, dims)::Vector{Int}

Get index of a vector's corresponding matrix.
"""
function getmatindex(i, dims)::Vector{Int}
    w = floor((i - 1) / dims[1]) + 1
    [i - (w - 1) * dims[1], w]
end

# these should probably just be two methods shouldn't they

"""
    movingleastsquares(source::Vector{Vector{Float64}}, target::Vector{Vector{Float64}}, dims, σ_s::Number, σ_r::Number; m::Number=1000, output)

Moving least squares algorithm using `m` control points, with spatial weighting.
Weighting in color space domain is controlled by σ_r, while spatial domain is controlled by σ_s.
"""
function movingleastsquares(
    source::Vector{Vector{Float64}},
    target::Vector{Vector{Float64}},
    dims,
    σ_s::Number,
    σ_r::Number;
    m::Number = 1000,
    output,
)
    maxval = 255
    # what's the point of duplicates?
    samples = unique(rand(1:length(source), m))
    m = length(samples)
    positions = Vector{Vector{Int}}(undef, m)
    insamples = Vector{Vector{Float64}}(undef, m)
    outsamples = Vector{Vector{Float64}}(undef, m)
    for j = 1:length(samples)
        i = samples[j]
        positions[j] = getmatindex(i, dims)
        insamples[j] = source[i] .* maxval
        outsamples[j] = target[i] .* maxval
    end

    insamples, outsamples = extrapolatepoints(insamples, outsamples, maxval)

    Threads.@threads for i in ProgressBar(1:length(output))
        x = output[i] .* maxval
        pos = getmatindex(i, dims)
        w = Vector{Float64}(undef, m)

        u_mean_pre = zeros(3)
        v_mean_pre = zeros(3)
        sumw = 0

        for k = 1:m
            # there's clearly something wrong with position calculation
            # only using the color space distance works fine
            # tempw = exp(-(norm(insamples[k] - x) ^ 2 / (σ_r ^ 2)))
            # tempw = exp(-(norm(positions[k] - pos) ^ 2 / (σ_s ^ 2)))
            # if tempw > 0.001
            # println("Ref: $(positions[k]), Pos: $pos")
            # println("Vec: $(positions[k] - pos)")
            # println("Norm: $(norm(positions[k] - pos))")
            # println(tempw)
            # end
            tempw = exp(
                -0.5 * (
                    norm(positions[k] - pos)^2 / (σ_s^2) +
                    norm(insamples[k] - x)^2 / (σ_r^2)
                ),
            )
            w[k] = tempw
            u_mean_pre += tempw .* insamples[k]
            v_mean_pre += tempw .* outsamples[k]
            sumw += tempw
        end

        u_mean = u_mean_pre / sumw
        v_mean = v_mean_pre / sumw

        uhat = Vector{Vector{Float64}}(undef, m)
        vhat = Vector{Vector{Float64}}(undef, m)

        for k = 1:m
            uhat[k] = insamples[k] - u_mean
            vhat[k] = outsamples[k] - v_mean
        end

        # the unparallelized solution
        A_x =
            inv(sum(w .* uhat .* permutedims.(uhat))) * sum(w .* uhat .* permutedims.(vhat))

        # this seems to be what the solution for GPU processing is supposed to be
        # y1 = permutedims([permutedims(vec(uhat[k] * permutedims(uhat[k]))) for k in 1:m])
        # y2 = permutedims([permutedims(vec(uhat[k] * permutedims(vhat[k]))) for k in 1:m])

        # the paper suggests transposing w is necessary here - will need to prove
        # A_x = inv(reshape(w * y1, (3, 3))) * reshape(w * y2, (3, 3))

        output[i] = min.(max.((A_x * (x - u_mean) + v_mean) / maxval, 0), 1)
    end

    output
end

"""
    movingleastsquares(source::Vector{Vector{Float64}}, target::Vector{Vector{Float64}}, α::Number; m::Number=1000, output)

Moving least squares algorithm using `m` control points.
"""
function movingleastsquares(
    source::Vector{Vector{Float64}},
    target::Vector{Vector{Float64}},
    α::Number;
    m::Number = 1000,
    output,
)
    maxval = 255
    # what's the point of duplicates?
    samples = unique(rand(1:length(source), m))
    m = length(samples)
    insamples = Vector{Vector{Float64}}(undef, m)
    outsamples = Vector{Vector{Float64}}(undef, m)
    for j = 1:length(samples)
        i = samples[j]
        insamples[j] = source[i] * maxval
        outsamples[j] = target[i] * maxval
    end

    insamples, outsamples = extrapolatepoints(insamples, outsamples, maxval)

    palette = unique(output)

    lut = Dict{typeof(palette[1]),typeof(insamples[1])}()

    Threads.@threads for i in ProgressBar(palette)
        x = i .* 256
        w = Vector{Float64}(undef, m)

        u_mean_pre = zeros(3)
        v_mean_pre = zeros(3)
        sumw = 0

        for k = 1:m
            tempw = min(1 / (norm(x - insamples[k])^(2 * α)), 1)
            w[k] = tempw
            u_mean_pre += tempw .* insamples[k]
            v_mean_pre += tempw .* outsamples[k]
            sumw += tempw
        end

        u_mean = u_mean_pre / sumw
        v_mean = v_mean_pre / sumw

        uhat = Vector{Vector{Float64}}(undef, m)
        vhat = Vector{Vector{Float64}}(undef, m)

        for k = 1:m
            uhat[k] = insamples[k] - u_mean
            vhat[k] = outsamples[k] - v_mean
        end

        # the unparallelized solution
        A_x =
            inv(sum(w .* uhat .* permutedims.(uhat))) * sum(w .* uhat .* permutedims.(vhat))

        # this seems to be what the solution for GPU processing is supposed to be
        # y1 = permutedims([permutedims(vec(uhat[k] * permutedims(uhat[k]))) for k in 1:m])
        # y2 = permutedims([permutedims(vec(uhat[k] * permutedims(vhat[k]))) for k in 1:m])

        # the paper suggests transposing w is necessary here - will need to prove
        # A_x = inv(reshape(w * y1, (3, 3))) * reshape(w * y2, (3, 3))

        lut[i] = min.(max.((A_x * (x - u_mean) + v_mean) / maxval, 0), 1)
    end

    for i = 1:length(output)
        output[i] = lut[output[i]]
    end

    output
end

"""
    imgtovec(img::Matrix{RGB{N0f8}})

Image to vector of vector of floats.
"""
function imgtovec(img::Union{Matrix{RGB{N0f8}},Matrix{RGB{Float64}}})
    img = channelview(img)
    # convert to 2 ⨯ 2 ⨯ 3
    img = permutedims(img, (2, 3, 1))
    dims = size(img)
    # convert from matrix to to vector of [R, G, B]
    img = reshape(img, dims[1] * dims[2], 3)
    img = [img[i, :] for i = 1:size(img, 1)]
    convert(Vector{Vector{Float64}}, img)
end

"""
    vectoimg(vec::Vector{Vector{Float64}}, dims::Tuple{Int, Int})

Convert vector of vector of floats to Images-compatible matrix.
"""
function vectoimg(vec::Vector{Vector{Float64}}, dims::Tuple{Int,Int})
    vec = reshape(vec, dims)
    dims = (3, dims...)
    formatted = Array{Float64}(undef, dims)
    for j = 1:size(vec, 2)
        for i = 1:size(vec, 1)
            for k = 1:3
                formatted[k, i, j] = vec[i, j][k]
            end
        end
    end
    colorview(RGB, formatted)
end

"""
    identity3dlut(dim::Int)

Generate identity 3D LUT as vector of size `dim` containing 3D Float64 vectors..
"""
function identity3dlut(dim::Int)
    lut = Vector{Vector{Float64}}(undef, dim^3)
    dim -= 1
    step = 1.0 / dim
    l = 1
    for k = 0:dim
        kval = round(step * k, digits = 6)
        for j = 0:dim
            jval = round(step * j, digits = 6)
            for i = 0:dim
                #         ival
                lut[l] = [round(step * i, digits = 6), jval, kval]
                l += 1
            end
        end
    end
    lut
end

"""
    luttocube(lut::Vector{Vector{Float64}}, fname::String, title::String="")

Write LUT to file.
"""
function luttocube(lut::Vector{Vector{Float64}}, fname::String, title::String = "")
    lutsize = trunc(Int, cbrt(size(lut, 1)))
    open(fname, "w") do f
        write(f, "TITLE $title\nLUT_3D_SIZE $lutsize\n")
        for i = 1:size(lut, 1)
            write(f, @sprintf "%.6f %.6f %.6f\n" lut[i][1] lut[i][2] lut[i][3])
        end
    end
end

"""
    round(rgb::Union{RGB{Float32}, RGB{Float64}})

Rounding for RGB.
"""
function round(rgb::Union{RGB{Float32},RGB{Float64}})
    RGB(round(rgb.r), round(rgb.g), round(rgb.b))
end

"""
    floydsteinberg_transfer(img, outimg=nothing)

Floyd-Steinberg dithering with 8-bit output depth.
"""
function floydsteinberg_transfer(img, outimg = nothing)
    if outimg === nothing
        outimg = Matrix{RGB{N0f8}}(undef, size(img, 1), size(img, 2))
    end
    # f-s dither
    for h = 1:size(img, 1)
        for w = 1:size(img, 2)
            oldpixel = img[h, w]
            newpixel = round(oldpixel * 255) / 255
            outimg[h, w] = newpixel
            quant_error = oldpixel - newpixel
            if h + 1 < size(img, 1)
                img[h+1, w] = img[h+1, w] + quant_error * 7 / 16
            end
            if h - 1 > 0
                img[h-1, w] = img[h-1, w] + quant_error * 3 / 16
            end
            if w + 1 < size(img, 2)
                img[h, w+1] = img[h, w+1] + quant_error * 5 / 16
            end
            if (h + 1 < size(img, 1)) && (w + 1 < size(img, 2))
                img[h+1, w+1] = img[h+1, w+1] + quant_error * 1 / 16
            end
        end
    end
    outimg
end

function loadimage(imgstr, prefilterstrength)
    img = load(imgstr)

    if prefilterstrength > 0
        flt = imfilter(img, Kernel.gaussian(prefilterstrength))
    end

    imgtovec(flt)
end

function main()
    s = ArgParseSettings()
    s.prog = "MLSColorTransfer"
    s.description = "Moving least squares for color transfer."
    s.version = "0.1.0"
    @add_arg_table s begin
        "source"
        help = "Source image(s). For multiple images, this should be a folder with shared filenaming between source and target images."
        required = true
        arg_type = String
        "target"
        help = "Target image(s). For multiple images, this should be a folder.with shared filenaming between source and target images."
        required = true
        arg_type = String
        "filename"
        help = "Output image (png) or 3D LUT (cube) file name."
        required = true
        arg_type = String
        "--spatial", "-S"
        help = "Enable using spatial info. Can only be used if outputting an image. Experimental."
        action = :store_true
        "--no-prefilter", "-P"
        help = "Disable prefilter."
        action = :store_true
        "--spatial-weight", "-s"
        help = "Spatial domain weighting strength."
        arg_type = Float64
        default = 20.0
        "--color-weight", "-r"
        help = "Color space domain weighting strength."
        arg_type = Float64
        default = 20.0
        "--prefilter-strength", "-p"
        help = "Prefilter strength."
        arg_type = Float64
        default = 2.0
        "--control-points", "-c"
        help = "Number of control points to use. Lowering this will lead to worse results and potential crashes, but will decrease computation time. It's recommended to set to 1% of total pixel count for best results."
        arg_type = Int
        default = 1000
        "--lut-size", "-l"
        help = "3D LUT size."
        arg_type = Int
        default = 33
        "--image", "-i"
        help = "Image to apply transfer to. Must be set if outputting an image."
        arg_type = String
    end

    args = parse_args(s)

    println("Starting...")

    if isdir(args["source"]) && isdir(args["target"])
        args["source"] = readdir(args["source"], join = true)
        args["target"] = readdir(args["target"], join = true)
        source = vcat(loadimage.(args["source"], args["prefilter-strength"])...)
        target = vcat(loadimage.(args["target"], args["prefilter-strength"])...)
        dims = size(load(args["source"][1]))
    elseif all(isfile.([args["source"], args["target"]]))
        source = loadimage(args["source"], args["prefilter-strength"])
        target = loadimage(args["target"], args["prefilter-strength"])
        dims = size(load(args["source"]))
    else
        throw(error("Source and target must be either both folders or both single images."))
    end

    if endswith(args["filename"], ".png")
        real = load(args["image"])
        real = imgtovec(real)
    elseif endswith(args["filename"], ".cube")
        real = identity3dlut(args["lut-size"])
    else
        throw(error("Only PNG and CUBE output file formats are supported."))
    end

    if !args["spatial"]
        transfer =
            movingleastsquares(source, target, 2, m = args["control-points"], output = real)
    elseif endswith(args["filename"], ".png")
        transfer = movingleastsquares(
            source,
            target,
            dims,
            args["spatial-weight"],
            args["color-weight"],
            m = args["control-points"],
            output = real,
        )
    else
        throw(error("Cannot generate LUTs when using spatial information!"))
    end

    # I guess?
    for i = 1:length(transfer)
        transfer[i] = @. min(max(transfer[i], 0), 1)
    end

    if endswith(args["filename"], ".png")
        transfer = vectoimg(transfer, dims, endswith(args["filename"], ".png"))
        transfer8 = floydsteinberg_transfer(transfer)

        save(args["filename"], transfer8)
    else
        luttocube(transfer, args["filename"], replace(args["filename"], ".cube" => ""))
    end
end

function julia_main()::Cint
    try 
        main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1 
    end
    return 0
end

end
