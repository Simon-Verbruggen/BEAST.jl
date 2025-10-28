# # OSRC MtE operator (Magnetic-to-Electric operator)
# The Magnetic-to-Electric map is an appoximate map of the magnetic current ``\mathbf{M}`` to the electrical current ``\mathbf{J}`` on a scattering surface.
# The approximation of the local MtE surface operator on an arbitrary surface ``\Gamma`` is given by:
# ```math
# \begin{equation}
#    \mathbf{M} + \mathbf{\Lambda}_{\epsilon_{opt}} (\mathbf{n} \times \mathbf{J}) = 0, \quad \text{on } \Gamma,
# \end{equation}
# ```
# with 
# ```math
# \begin{aligned}
#    \mathbf{\Lambda}_{\epsilon_{opt}} &= \mathbf{\Lambda}^{-1}_{1,\epsilon_{opt}} \mathbf{\Lambda}_{2,\epsilon_{opt}}, \\
#    \mathbf{\Lambda}_{1,\epsilon_{opt}} &= Z_0 \left(\mathbf{I} + \nabla_{\Gamma} \frac{1}{k_{{\epsilon}_{opt}}^2} \text{div}_{\Gamma} - \textbf{curl}_{\Gamma} \frac{1}{k_{{\epsilon}_{opt}}^2} \text{curl}_{\Gamma} \right)^{1/2}, \\
#    \mathbf{\Lambda}_{2,\epsilon_{opt}} &= \mathbf{I} - \textbf{curl}_{\Gamma} \frac{1}{k_{{\epsilon}_{opt}}^2}. \text{curl}_{\Gamma}
# \end{aligned}
# ```
# Relations like (1) correspond to the class of On-Surface Radiation Conditions (OSRC) methods.
# More details of this OSRC operator are given in the paper <a href="https://www.researchgate.net/publication/261636307_Approximate_local_magnetic-to-electric_surface_operators_for_time-harmonic_Maxwell's_equations" target='new'> Approximate local magnetic-to-electric surface operators for time-harmonic Maxwell’s equations (C. Geuzaine et al. (2014))</a>.
# 
#
# First we implement a rotating branch-cut rational Padé approximation of the square root function ``\sqrt{1+z^2}``.
using SparseArrays
using BEAST

struct Pade_approx
    Np::Int
    θ_p::Float64
end

function get_a_j(p::Pade_approx, j::Int)
    return 2/(2*p.Np+1)*(sin(j*pi/(2*p.Np+1)))^2
end

function get_b_j(p::Pade_approx, j::Int)
    return (cos(j*pi/(2*p.Np+1)))^2
end

function get_A_j(p::Pade_approx, j::Int)
    return exp(-im*p.θ_p/2) * get_a_j(p,j) / (1 + get_b_j(p,j)*(exp(-im * p.θ_p) - 1))^2
end
function get_B_j(p::Pade_approx, j::Int)
    return exp(-im*p.θ_p) * get_b_j(p,j) / (1 + get_b_j(p,j)*(exp(-im * p.θ_p) - 1))
end

function get_RNp(z, p::Pade_approx)
    R_Np = 1 + sum(get_a_j(p,j)*z/(1+get_b_j(p,j)*z) for j in 1:p.Np)
    return R_Np
end

function get_R0(p::Pade_approx)
    C0 = exp(im*p.θ_p/2) * get_RNp((exp(-im*p.θ_p)-1), p)
    R0 = C0 + sum(get_A_j(p,j)/get_B_j(p,j) for j in 1:p.Np)
    return R0
end

function rotated_pade(z, p::Pade_approx)
    return get_R0(p) - sum(get_A_j(p, j)/(get_B_j(p, j)*(1 + get_B_j(p, j)*z)) for j in 1:p.Np)
end

# The square root operator is regularized by adding a small imaginary component ``\epsilon`` to the wavenumber: ``k_{\epsilon} = k + i \epsilon``.
function MtE_damping(;wavenumber=nothing, curvature=nothing)
    return 0.39*wavenumber^(1/3)*curvature^(2/3)
end

# Next, the MtE operator is assembled as a linear map. This is implemented according to the discretization in the paper <a href="https://arxiv.org/abs/2111.10761" target='new'> An OSRC Preconditioner for the EFIE (Betcke et al. (2021))</a>.

function MtE_operator(Γ, κ, Np::Int, theta_p::Float64)
    pade_struct = Pade_approx(Np, theta_p);     # load in pade struct for pade coefficients and constants
    R_0 = get_R0(pade_struct)

    ϵ = MtE_damping(wavenumber=κ, curvature=1/1.0)
    κ_ϵ = κ + im*ϵ
    
    @hilbertspace ϕ ρ       # trial functions
    @hilbertspace w z       # testing functions

    # Define the relevant function spaces
    Nd = BEAST.nedelec(Γ);
    L0_int = BEAST.lagrangec0d1(Γ);
    grad_L0_int = BEAST.gradient(L0_int)
    curl_Nd = BEAST.curl(Nd)

    # Assemble the submatrices of the blockmatrix of the system
    Id = BEAST.Identity();
    G = assemble(Id, Nd, Nd)
    N_ϵ = sparse(assemble((1/κ_ϵ)^2 * Id, curl_Nd, curl_Nd))
    K_ϵ = sparse(assemble(κ_ϵ^2 * Id, L0_int, L0_int))
    L = assemble(Id, Nd, grad_L0_int)
    L_transpose = assemble(Id, grad_L0_int, Nd)

    # Calculate the inverse via Schur's complement
    K_ϵ_inv = BEAST.lu(K_ϵ)
    function phi_j_inv_Schur(j)
        B_j = get_B_j(pade_struct, j)
        Π = sparse(G - B_j * N_ϵ - B_j * L * K_ϵ_inv * L_transpose)
        Π_inv = BEAST.lu(Π)
        return Π_inv
    end

    # Finally, construct the MtE_map
    sum_Π_inv_matrix = sum(get_A_j(pade_struct, j)/get_B_j(pade_struct, j) * phi_j_inv_Schur(j) for j in 1:Np)
    G_N_ϵ_inv = BEAST.lu(G - N_ϵ)
    MtE_map = - (G_N_ϵ_inv * R_0 - G_N_ϵ_inv * G * sum_Π_inv_matrix)
    return MtE_map
end