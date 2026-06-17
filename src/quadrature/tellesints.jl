struct DoubleNumSauterTellesQstrat{R,T,S} <: AbstractQuadStrat
    outer_rule::R
    inner_rule::R
    telles_outer_rule::T
    telles_inner_rule::T
    sauter_schwab_common_edge::S
    sauter_schwab_common_vert::S
end

struct SingleNumTellesQStrat{R,T} <: AbstractQuadStrat
    gauss_rule::R
    telles_rule::T
end

"""
extending the momintegrals! function for the Telles quadrature strategy. This is used for
near-singular integrals in 2D, where the singularity is not on the edge but close to it.
"""

function momintegrals!(op::Operator,
    test_local_space, trial_local_space,
    test_chart, trial_chart,
    out, rule::BEAST.TellesQuadrature.TellesRule2D)

    num_tshapes = numfunctions(test_local_space, domain(test_chart))
    num_bshapes = numfunctions(trial_local_space, domain(trial_chart))

    igd = Integrand(op, test_local_space, trial_local_space, test_chart, trial_chart)

    G = BEAST.TellesQuadrature.telles_parametrized(igd, rule, num_tshapes, num_bshapes)

    for j in 1:num_bshapes
        for i in 1:num_tshapes
            out[i, j] += G[i, j]
        end
    end

    nothing
end
