module TellesQuadrature

# -------- exportet parts
# types
export TellesRule2D
export TellesRule1D
# functions
export telles_parametrized
export tellespoints
# -------- included files

# -------- imports
import ..scalartype

# -------- used packages
using FastGaussQuadrature
using CompScienceMeshes
using LinearAlgebra
using StaticArrays
using BEAST


struct TellesRule2D{A}
    qpso::A # GL quadrature for the outer integral
    qpsi::A # GL quadrature for the inner integral
end

struct TellesRule1D{A}
    qps::A # GL quadrature for the integral
end

# These are the values found to be working best for a static or Helmholtz kernel. They are
# based on on the the procedure described in "Third degree polynomial transformation for
#boundary element integrals: Further improvements" by Telles, 2002, and the values in Table
#1 of that paper.
const R_BAR_THRESHOLDS = [0.001, 0.005, 0.01, 0.03, 0.075, 0.15, 0.25, 0.4, 0.6, 0.8, 1.25,
    2.0, 3.0, 4.0, 4.5]
const R_BAR_VALUES = [0.051, 0.086, 0.116, 0.197, 0.316, 0.446, 0.564, 0.605, 0.731, 0.806,
    0.892, 0.946, 0.972, 0.98, 0.99]

"""
The parameter r_bar is a function of the distance D from the reference point to the edge
relative to the length of said edge. It is used in the quasi-singular case of the
Tellestransformation. This function returns the r_bar value for a given D, based on the
thresholds and values defined in R_BAR_THRESHOLDS and R_BAR_VALUES. It performs linear
interpolation between the thresholds to find the appropriate r_bar value for any D within
the range of the thresholds. Do note that these are somewhat kernel specific and while
always lying between 0 (the transformation equals the singular case) and 1 (Gauss case with
overhead), they are not necessarily optimal for all kernels.
"""

function getr_bar(D::Float64)
    if D <= R_BAR_THRESHOLDS[1]
        return R_BAR_VALUES[1]
    elseif D >= R_BAR_THRESHOLDS[end]
        return R_BAR_VALUES[end]
    end

    idx = searchsortedfirst(R_BAR_THRESHOLDS, D)

    # Linear interpolation between idx-1 and idx
    D0 = R_BAR_THRESHOLDS[idx-1]
    D1 = R_BAR_THRESHOLDS[idx]
    R0 = R_BAR_VALUES[idx-1]
    R1 = R_BAR_VALUES[idx]

    return R0 + (D - D0) * (R1 - R0) / (D1 - D0)
end

"""
telles_deg3(acc, igd, aux, refpoint, qpsN1)

This is the core function for the Telles transformation. It computes the integral of the
given auxiliary function aux, which is a function of the transformed variable ξ, over the
reference point refpoint, using the Telles transformation for a 1D edge. The function takes
into account the geometry of the edge and the position of the reference point to determine
whether the singularity is on the axis or not, and applies the appropriate transformation
accordingly. The function uses the quadrature points and weights provided in qpsN1 to
perform the numerical integration after the transformation. Further information on the
transformation can be found in "Third degree polynomial transformation for boundary element
integrals: Further improvements" by Telles, 2002.
"""

@inline function telles_deg3(acc, igd, aux, refpoint::SVector, qpsN1)

    # --- TYPE OF ACCUMULATOR ---
    #Telles prefers the Gauss points on the interval [-1,1]
    #rather than [0,1] so we need to adapt our points and weights and keep the jacobian
    jac = 1 / 2
    qpsN = BEAST._legendre(length(qpsN1), -1, 1)
    #[(2 * v1 - 1, 2 * w1) for (v1, w1) in qpsN1]
    # --- GEOMETRY --- We find the point T on the edge where the tangent is perpendicular to
    #the vector from the reference point to the edge. We then find the distance from the
    #reference point to T, and use this to determine the parameter η_bar, which indicates
    #how close the singularity is to the edge. We also compute the relative distance D,
    #which is the distance from the reference point to T relative to the length of the edge,
    #and use this to determine if we are in the quasi-singular case or not.
    L = volume(igd.trial_chart)
    η_tilde = normalize(igd.trial_chart.tangents[1])
    nullpoint = cartesian(neighborhood(igd.trial_chart, 0.0))
    onepoint = cartesian(neighborhood(igd.trial_chart, 1.0))

    𝒹_tilde = refpoint - nullpoint
    u_star = dot(η_tilde, 𝒹_tilde) / L
    T = nullpoint + η_tilde * L * u_star
    Dist = norm(refpoint - T)
    η_bar = u_star * 2 - 1

    if η_bar > 1
        D = 2 * norm(refpoint - onepoint) / L
    elseif η_bar < -1
        D = 2 * norm(refpoint - nullpoint) / L
    else
        D = 2 * Dist / L
    end

    # --- AXIS CHECK ---
    isonaxis = true
    threshhold = 10^-13
    if Dist > threshhold
        isonaxis = false
    end
    #In the paper cited above, a splitting of the edge is proposed for the case where the
    #singularity is close to the edge but not on the axis. However, it turns out that theh
    #gain in accuracy is not worth the additional overhead it creates. This decision is
    #based on empirical evidence testing against other methods of integrating in the quasi
    #singular case with the same amount of quadrature points. This may be revisited in the
    #future if further testing suggests that splitting could be beneficial in certain cases.
    #=
    if abs(η_bar) < 1
        # Recursively split the edge, return type T
        return tellesnocm_integratedeg3(K,
            SVector(nullpoint, nullpoint + u_star * L * η_tilde),
            refpoint, N
        ) + tellesnocm_integratedeg3(K,
            SVector(nullpoint + u_star* L * η_tilde, Edge[1]),
            refpoint, N
        )
    end=#
    # ==============================================================
    # ======================= NOT ON AXIS ===========================
    # ==============================================================
    if !isonaxis
        #We fetch r_bar with the function defined above, which is based on the distance D
        #from the reference point to the edge relative to the length of the edge. This
        #parameter is crucial for determining the coefficients of the cubic transformation
        #used in the Telles method for quasi-singular integrals.
        r_bar = getr_bar(D)

        q = (η_bar * (3 - 2 * r_bar) - (2 * η_bar^3) / (1 + 2 * r_bar)) /
            (2 * (1 + 2 * r_bar)^2) - η_bar / (2 * (1 + 2 * r_bar))
        p = (4 * r_bar * (1 - r_bar) + 3 * (1 - η_bar^2)) /
            (3 * (1 + 2 * r_bar)^2)

        γ_bar = cbrt(-q + sqrt(q^2 + p^3)) +
                cbrt(-q - sqrt(q^2 + p^3)) +
                η_bar / (1 + 2 * r_bar)

        Q = 1 + 3 * γ_bar^2

        # Polynomial coefficients
        coef = (
            a=(1 - r_bar) / Q,
            b=-3 * (1 - r_bar) * γ_bar / Q,
            c=(r_bar + 3 * γ_bar^2) / Q,
            d=3 * (1 - r_bar) * γ_bar / Q    # (=-b)
        )

        # A let-block keeps mapv / dmapv visible for loop use
        let a = coef.a, b = coef.b, c = coef.c, d = coef.d

            @inline mapv(v1) = a * v1^3 + b * v1^2 + c * v1 + d
            @inline dmapv(v1) = 3 * a * v1^2 + 2 * b * v1 + c

            @inbounds for j in eachindex(qpsN)
                v1, w1 = qpsN[j]

                t = mapv(v1)
                ξ = (t + 1) / 2
                #x = P1 + ξ * Δ

                acc += jac * dmapv(v1) * w1 *
                       aux(ξ)
            end
        end

        return acc
    end

    # ==============================================================
    # ========================== ON AXIS ===========================
    # ==============================================================

    # the axis that is refered to here is the axis of the edge, so the singularity is on the
    # axis if the reference point is directly above or below the edge. In this case, we use
    # a different transformation, which is based on a cubic polynomial with coefficients
    # determined by η_bar, which is the parameter in [-1,1]

    η_star = η_bar^2 - 1

    γ_bar = cbrt(η_bar * η_star + abs(η_star)) +
            cbrt(η_bar * η_star - abs(η_star)) +
            η_bar

    denom = 1 + 3 * γ_bar^2

    let γ = γ_bar, denom = denom

        @inline mapv(v1) = ((v1 - γ)^3 + γ * (γ^2 + 3)) / denom
        @inline dmapv(v1) = 3 * (v1 - γ)^2 / denom

        @inbounds for j in eachindex(qpsN)
            v1, w1 = qpsN[j]

            t = mapv(v1)
            ξ = (t + 1) / 2

            #x = P1 + ξ * Δ

            acc += jac * dmapv(v1) * w1 *
                   aux(ξ)
        end
    end

    return acc
end

function telles_parametrized(igd, rule::TellesRule2D, num_tshapes, num_bshapes)
    qpsi = rule.qpsi
    qpso = rule.qpso
    G = zeros(scalartype(igd.operator), num_tshapes, num_bshapes)
    for (v1, w1) in qpso
        G += w1 * telles_deg3(zeros(scalartype(igd.operator), num_tshapes, num_bshapes),
            igd, x -> igd(x, v1), cartesian(igd.test_chart, v1), qpsi)
    end
    return G
end

"""
tellespoints(igd, refspace, refpoint, qpair)

For the postprocessing of near-singular integrals, we need to evaluate the integrand at the
quadrature points of the Telles transformation. This function takes in the integration
geometry data (igd), the reference space, the reference point, and a quadrature pair (qpair)
consisting of a quadrature point and weight. It applies the Telles transformation to compute
the corresponding point in the configuration space and its associated weight, which can then
be used for evaluating the integrand at that point. The function handles both cases where
the referencepoint is on a common axis with the trial element and where it is not, ensuring
accurate evaluation of near-singular integrals.
"""
@inline function tellespoints(igd, refspace, refpoint, qpair)
    v1, w1 = qpair
    jac = 1 / 2
    # --- GEOMETRY --- We find the point T on the edge where the tangent is perpendicular to
    #the vector from the reference point to the edge. We then find the distance from the
    #reference point to T, and use this to determine the parameter η_bar, which indicates
    #how close the singularity is to the edge. We also compute the relative distance D,
    #which is the distance from the reference point to T relative to the length of the edge,
    #and use this to determine if we are in the quasi-singular case or not.
    L = volume(igd.trial_chart)
    η_tilde = normalize(igd.trial_chart.tangents[1])
    nullpoint = cartesian(neighborhood(igd.trial_chart, 0.0))
    onepoint = cartesian(neighborhood(igd.trial_chart, 1.0))

    𝒹_tilde = refpoint - nullpoint
    u_star = dot(η_tilde, 𝒹_tilde) / L
    T = nullpoint + η_tilde * L * u_star
    Dist = norm(refpoint - T)
    η_bar = u_star * 2 - 1

    if η_bar > 1
        D = 2 * norm(refpoint - onepoint) / L
    elseif η_bar < -1
        D = 2 * norm(refpoint - nullpoint) / L
    else
        D = 2 * Dist / L
    end

    # --- AXIS CHECK ---
    isonaxis = true
    threshhold = 10^-13
    if Dist > threshhold
        isonaxis = false
    end
    # ==============================================================
    # ======================= NOT ON AXIS ===========================
    # ==============================================================
    if !isonaxis
        #We fetch r_bar with the function defined above, which is based on the distance D
        #from the reference point to the edge relative to the length of the edge. This
        #parameter is crucial for determining the coefficients of the cubic transformation
        #used in the Telles method for quasi-singular integrals.
        r_bar = getr_bar(D)

        q = (η_bar * (3 - 2 * r_bar) - (2 * η_bar^3) / (1 + 2 * r_bar)) /
            (2 * (1 + 2 * r_bar)^2) - η_bar / (2 * (1 + 2 * r_bar))
        p = (4 * r_bar * (1 - r_bar) + 3 * (1 - η_bar^2)) /
            (3 * (1 + 2 * r_bar)^2)

        γ_bar = cbrt(-q + sqrt(q^2 + p^3)) +
                cbrt(-q - sqrt(q^2 + p^3)) +
                η_bar / (1 + 2 * r_bar)

        Q = 1 + 3 * γ_bar^2

        # Polynomial coefficients
        coef = (
            a=(1 - r_bar) / Q,
            b=-3 * (1 - r_bar) * γ_bar / Q,
            c=(r_bar + 3 * γ_bar^2) / Q,
            d=3 * (1 - r_bar) * γ_bar / Q    # (=-b)
        )

        # A let-block keeps mapv / dmapv visible for loop use
        acc = let a = coef.a, b = coef.b, c = coef.c, d = coef.d

            @inline mapv(v1) = a * v1^3 + b * v1^2 + c * v1 + d
            @inline dmapv(v1) = 3 * a * v1^2 + 2 * b * v1 + c

            t = mapv(v1)
            ξ = (t + 1) / 2
            mp = neighborhood(igd.trial_chart, ξ)
            (weight=jac * dmapv(v1) * w1 * jacobian(mp), point=mp, value=refspace(mp))
        end

        return acc
    end
    # ==============================================================
    # ========================== ON AXIS ===========================
    # ==============================================================

    # the axis that is refered to here is the axis of the edge, so the singularity is on the
    # axis if the reference point is directly above or below the edge. In this case, we use
    # a different transformation, which is based on a cubic polynomial with coefficients
    # determined by η_bar, which is the parameter in [-1,1]

    η_star = η_bar^2 - 1

    γ_bar = cbrt(η_bar * η_star + abs(η_star)) +
            cbrt(η_bar * η_star - abs(η_star)) +
            η_bar

    denom = 1 + 3 * γ_bar^2

    acc = let γ = γ_bar, denom = denom

        @inline mapv(v1) = ((v1 - γ)^3 + γ * (γ^2 + 3)) / denom
        @inline dmapv(v1) = 3 * (v1 - γ)^2 / denom

        t = mapv(v1)
        ξ = (t + 1) / 2

        #x = P1 + ξ * Δ

        mp = neighborhood(igd.trial_chart, ξ)
        (weight=jac * dmapv(v1) * w1 * jacobian(mp), point=mp, value=refspace(mp))
    end

    return acc
end

end
