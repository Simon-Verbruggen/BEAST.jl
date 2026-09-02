using SparseArrays
using LinearAlgebra
using BEAST
import Polynomials

struct MtE_OSRC_rational_op <: Operator
    κ_ϵ::ComplexF64
    tol::Float64
end

# First a rotating branch-cut rational Padé approximation of the square root function ``\sqrt{1+z^2}`` is implemented.
imag_conv = -im     # The paper uses mathematical time-harmonic convention ``e^{-i \omega t}`` -> swap to match BEAST time-harmonic convention ``e^{i \omega t}``.
function MtE_OSRC_rational_op(wavenumber::Float64, tol::Float64; curvature=nothing, ϵ=nothing)
    # get epsilon parameter
    if ϵ === nothing
        if curvature === nothing
            throw(ArgumentError("Either curvature or ϵ must be provided."))
        end
        ϵ = BEAST.OSRC_epsilon_standard(wavenumber, curvature)
    end
    κ_ϵ = wavenumber + imag_conv*ϵ
    return MtE_OSRC_rational_op(κ_ϵ, tol)
end

function scalartype(op::MtE_OSRC_rational_op)
    T = scalartype(op.κ_ϵ)
    return Complex{T}
end

##### spectrum estimation functions

import Plots
import KrylovKit
import RationalFunctionApproximation
import ComplexRegions

# estimate the spectral radius of the operator H using the Lanczos method
function lanczos_end_estimate(H, N_X, κ_ϵ) 
    x₀ = ones(ComplexF64, N_X)
    tol_lanczos = 1e-4
    λ, vecs, info = KrylovKit.eigsolve(H,x₀,1,:LM;tol=tol_lanczos)
    λmax = abs(λ[1])*(1+tol_lanczos*2)

    println("Lanczos")
    println("spectral radius = ", λmax)
    println("residual = ", info.normres[1])
    println("matrix-vector products = ", info.numops)

    end_x = - (1/κ_ϵ)^2 * λmax
    begin_x = 0
    return (begin_x, end_x)
end

# check function for rational approximation 

function relative_error(f1, f2, x_range)
    rel_error = abs.(f1.(x_range) - f2.(x_range)) ./ abs.(f1.(x_range))  
    return rel_error
end

function check_rational_approximation(f1, f2, r, begin_x, end_x, title)
    # Calculate errors
    t = 10 .^ range(-5, log10(abs(end_x)); length=300)
    x_range = begin_x .+ t * end_x/abs(end_x)

    rel_error_rational = relative_error(f1, f2, x_range) 
    rel_error_barycentric = relative_error(f1, r, x_range) 
    # plot errors
    nodes_r = minimum(abs.(x_range)) .+ abs.(RationalFunctionApproximation.nodes(r))
    plt = Plots.plot(abs.(x_range), rel_error_rational, label="error", linewidth=2, xscale=:log10)
    Plots.scatter!(plt, nodes_r, 0*nodes_r, markersize = 8, color=:black)
    Plots.savefig(plt, title)
    max_rel_error = maximum(rel_error_rational)
    return max_rel_error
end

function AAA_approximate_sqrt(begin_x, end_x, tol)
    r = RationalFunctionApproximation.approximate(x -> sqrt(Complex(1+x)), ComplexRegions.Segment(end_x, begin_x); tol=tol, float_type=BigFloat)
    b, a = RationalFunctionApproximation.residues(r)
    Np = length(a)
    R0 = sum(r.fun.w_times_f) / sum(r.fun.weights)
    Bj = - 1 ./ b
    Aj = - a .* Bj.^2
    return r, a, b, Aj, Bj, Np, R0
end

# obtain rational approximation using the AAA algorithm
function AAA_approximation_scalar(H, N_X, κ_ϵ, tol)
    (begin_x, end_x) = lanczos_end_estimate(H, N_X, κ_ϵ)
    tol_AAA_approximate = 1e-10
    r, a, b, Aj, Bj, Np, R0 = AAA_approximate_sqrt(begin_x, end_x, tol_AAA_approximate)
    f_rational = x -> R0 - sum(Aj[i] / (Bj[i]*(1 + Bj[i] * x)) for i in 1:Np)
    max_error = check_rational_approximation(x -> sqrt(Complex(1+x)), f_rational, r, begin_x, end_x, "temporary_error_check.pdf")
    if max_error > tol
        error("Max error $max_error exceeds tolerance $tol at ($begin_x, $end_x)")
    end
    return (r, a, b, Aj, Bj, Np, R0)
end

function assemble(op::MtE_OSRC_rational_op,X::Space,Y::Space; quadstrat=defaultquadstrat)

    κ_ϵ = op.κ_ϵ
    tol = op.tol

    #create auxilary basis functions
    L0_int = BEAST.lagrangec0d1(X.geo)
    grad_L0_int = BEAST.gradient(L0_int)
    # Define the relevant function spaces
    curl_X = BEAST.curl(X)
    curl_Y = BEAST.curl(Y)

    N_L0 = numfunctions(L0_int)
    N_X = numfunctions(X)
    N_Y = numfunctions(Y)

    # Assemble the submatrices of the blockmatrix of the system
    Id = BEAST.Identity();
    G = assemble(Id, X, Y)
    A = assemble(Id, curl_X, curl_Y)
    N_ϵ = (1/κ_ϵ)^2 * A
    M = assemble(Id, L0_int, L0_int)
    K_ϵ = κ_ϵ^2 * M
    L = assemble(Id, X, grad_L0_int)
    L_transpose = assemble(Id, grad_L0_int, Y)

    H = A + L * BEAST.lu(M) * L_transpose

    # perform the necessary estimation
    (r, a, b, Aj, Bj, Np, R0) = AAA_approximation_scalar(H, N_X, κ_ϵ, tol)

    # construct the sparse system matrix and invert
    function create_j_phi_matrix17(j)
        B_j = Bj[j]
        # blockmatrix of sparse matrices
        AXY = [G-B_j*N_ϵ       B_j*L
                L_transpose     K_ϵ]
        SXY = BEAST.lu(AXY)
        Sliced_SXY = BEAST.SlicedLinearMap(SXY, 1:N_X, 1:N_Y)
        P = Sliced_SXY.P
        Q = Sliced_SXY.Q
        SXY_sliced = P*SXY*Q
        return SXY_sliced
    end

    sum_Π_inv_matrix = sum(Aj[j]/Bj[j] * create_j_phi_matrix17(j) for j in 1:Np)
    G_N_ϵ_inv = BEAST.lu(G - N_ϵ)

    MtE_map = - (G_N_ϵ_inv * R0 - G_N_ϵ_inv * G * sum_Π_inv_matrix)
    return MtE_map
end