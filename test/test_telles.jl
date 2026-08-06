using BEAST
using CompScienceMeshes
using StaticArrays
using LinearAlgebra
using Test
#import PlotlyJS       <-- For Plottong only


@testset "Telles-Integration" begin
    ℒ = 10.0
    𝒹 = 2.5 * ℒ / 2
    Stat = Helmholtz2D.singlelayer()
    P1 = SVector(0.0, 0.0)
    P2 = SVector(ℒ, 0.0)
    P3 = SVector(P1[1] + 0.25 * ℒ, P1[2] + 𝒹)
    P4 = SVector(P2[1] + 0.25 * ℒ, P2[2] + 𝒹)
    vertices = [P1, P2, P3, P4]

    el1 = CompScienceMeshes.SimplexGraph{2}(index(1,2))
    el2 = CompScienceMeshes.SimplexGraph{2}(index(3,4))
    faces = [el1, el2]
    tstmesh = Mesh(vertices, faces)
    X01 = lagrangecxd0(tstmesh)
    function referencestat(l::Float64, y::SVector)
        return -(1 / (4 * π)) * (((1 * l - y[1]) * log((1 * l - y[1])^2 + y[2]^2) - 2 * (1 * l - y[1]) + 2 * y[2] * atan((1 * l - y[1]) / y[2])) - ((0 * l - y[1]) * log((0 * l - y[1])^2 + y[2]^2) - 2 * (0 * l - y[1]) + 2 * y[2] * atan((0 * l - y[1]) / y[2])))
    end
    function referencestatE(Edge::SVector{2,<:SVector}, Edge2::SVector{2,<:SVector}, N::Int)
        qpsN = BEAST._legendre(N, -1.0, 1.0)
        J = (norm(Edge2[2] - Edge2[1]) / 2)
        Δx = (Edge2[2][1] - Edge2[1][1])
        Δy = (Edge2[2][2] - Edge2[1][2])
        L = norm(Edge[2] - Edge[1])
        return sum(w1 * J * referencestat(L, SVector(Edge2[1][1] + ((v1 + 1) / 2) * Δx, Edge2[1][2] + ((v1 + 1) / 2) * Δy)) for (v1, w1) in qpsN)
    end
    reforder = 10000
    relerT = Float64[]
    ref = referencestatE(SVector(P1, P2), SVector(P3, P4), reforder)
    quads = [10, 30]
    for i in quads
        quadstrat = BEAST.DoubleNumSauterTellesQstrat(i, i, i, i, i, i)
        tel = assemble(Stat, X01, X01, quadstrat=quadstrat)[1, 2]
        push!(relerT, abs(tel - ref) / abs(ref))
    end


    @test relerT[1] < 10^-3
    @test relerT[2] < 10^-7

    function returntestmesh(𝒹::Float64, α::Float64, ℒ::Float64)
        α = α * π / 180
        P1 = SVector(-ℒ / 2, 0.0)
        P6 = SVector(ℒ / 2, 0.0)
        P3 = SVector(P1[1], P1[2] + 𝒹)
        P4 = SVector(P6[1], P6[2] + 𝒹)
        P2 = SVector(P1[1] - (𝒹 * 0.5 / tan(α / 2)), P1[2] + 𝒹 / 2)
        P5 = SVector(P6[1] + (𝒹 * 0.5 / tan(α / 2)), P6[2] + 𝒹 / 2)
        vertices = [P1, P2, P3, P4, P5, P6]
        el1 = CompScienceMeshes.SimplexGraph{2}(index(1,2))
        el2 = CompScienceMeshes.SimplexGraph{2}(index(2,3))
        el3 = CompScienceMeshes.SimplexGraph{2}(index(3,4))
        el4 = CompScienceMeshes.SimplexGraph{2}(index(4,5))
        el5 = CompScienceMeshes.SimplexGraph{2}(index(5,6))
        el6 = CompScienceMeshes.SimplexGraph{2}(index(6,1))
        faces = [el1, el2, el3, el4, el5, el6]
        M = Mesh(vertices, faces)
        return M
    end
    mesh = returntestmesh(1 / 20, 10.0, 1.0)
    #=========================For Plotting only=========================##=
    function make_3d(M)
        v = M.vertices
        f = M.faces
        new = [SVector(v1[1], v1[2], 0.0) for v1 in v]
        M3d = Mesh(new, f)
        return M3d
    end
    mesh3d = make_3d(mesh)
    plt = PlotlyJS.plot(CompScienceMeshes.wireframe(mesh3d))
    PlotlyJS.relayout!(plt, scene=PlotlyJS.attr(
    aspectmode="data"   # keeps x, y, z scaling equal to data units
    ))
    PlotlyJS.display(plt)
    =##============================End Plots==============================#
    X1 = lagrangec0d1(mesh)
    quads = [10, 30]
    relD = Float64[]
    for i in quads
        quadstrat = BEAST.DoubleNumSauterTellesQstrat(i, i, i, i, i, i)
        default = BEAST.DoubleNumSauterQstrat(i, i, 0, 4, i, i)
        B1 = assemble(Stat, X1, X1, quadstrat=quadstrat)
        B2 = assemble(Stat, X1, X1, quadstrat=default)
        push!(relD, norm(B2 - B1) / norm(B2))
    end
    @test relD[1] < 10^-2
    @test relD[2] < 10^-4
end

@testset "Telles-Integration: Helmholtz2D Nearfield" begin

k = 0.0
r = 10.0
circle = CompScienceMeshes.meshcircle(r, 1.0)

X0 = lagrangecxd0(circle)
X1 = lagrangec0d1(circle)

S = Helmholtz2D.singlelayer(;)
D = Helmholtz2D.doublelayer(;)
Dt = Helmholtz2D.doublelayer_transposed(;)
N = Helmholtz2D.hypersingular(;)


q = 100.0
ϵ = 1.0

# Interior problem
# Formulations from Sauter and Schwab, Boundary Element Methods(2011), Chapter 3.4.1.1

pos1 = SVector(r * 1.5, 0.0)  # positioning of point charges
pos2 = SVector(-r * 1.5, 0.0)

charge1 = Helmholtz2D.monopole(position=pos1, amplitude=q / (4 * π * ϵ))
charge2 = Helmholtz2D.monopole(position=pos2, amplitude=-q / (4 * π * ϵ))

# Potential of point charges
Φ_inc(x) = charge1(x) + charge2(x)

gD0 = assemble(DirichletTrace(charge1), X0) + assemble(DirichletTrace(charge2), X0)
gD1 = assemble(DirichletTrace(charge1), X1) + assemble(DirichletTrace(charge2), X1)
gN = assemble(∂n(charge1), X1) + assemble(BEAST.n ⋅ grad(charge2), X1)

G = assemble(Identity(), X1, X1)
o = ones(numfunctions(X1))

# Interior Dirichlet problem - compare Sauter & Schwab eqs. 3.81
M_IDPSL = assemble(S, X0, X0) # Single layer (SL)
M_IDPDL = (-1 / 2 * G + assemble(D, X1, X1)) # Double layer (DL)

# Interior Neumann problem
# Neumann derivative from DL potential with deflected nullspace
M_INPDL = assemble(N, X1, X1) + G * o * o' * G
# Neumann derivative from SL potential with deflected nullspace
M_INPSL = (1 / 2 * G + assemble(Dt, X1, X1)) + G * o * o' * G

ρ_IDPSL = M_IDPSL \ (-gD0)
ρ_IDPDL = M_IDPDL \ (-gD1)

ρ_INPSL = M_INPSL \ (-gN)
ρ_INPDL = M_INPDL \ (gN)

#In order to test the Telles quadrature, we evaluate the potentials at points close to the
#boundary, where the integrals are nearly singular. We expect the Telles quadrature to
#perform better than standard Gaussian quadrature in this regime.

pts = []
currentvert = circle.vertices[1]
for i in eachindex(circle.vertices[2:end])
    tan = circle.vertices[i+1] - currentvert
    midpoint = (currentvert + (tan ./ 2))
    push!(pts, midpoint .* 0.99) # push the midpoint of each edge to the interior and add to pts
    currentvert = circle.vertices[i]
end

pot_IDPSL = potential(HH2DSingleLayerNear(S), pts, ρ_IDPSL, X0; type=ComplexF64, quadstrat=BEAST.SingleNumQStrat(3))
pot_IDPDL = potential(HH2DDoubleLayerNear(D), pts, ρ_IDPDL, X1; type=ComplexF64, quadstrat=BEAST.SingleNumQStrat(3))
pot_INPSL = potential(HH2DSingleLayerNear(S), pts, ρ_INPSL, X1; type=ComplexF64, quadstrat=BEAST.SingleNumQStrat(3))
pot_INPDL = potential(HH2DDoubleLayerNear(D), pts, ρ_INPDL, X1; type=ComplexF64, quadstrat=BEAST.SingleNumQStrat(3))

pot_IDPSL_telles = potential(HH2DSingleLayerNear(S), pts, ρ_IDPSL, X0; type=ComplexF64, quadstrat=BEAST.SingleNumTellesQStrat(3, 3))
pot_IDPDL_telles = potential(HH2DDoubleLayerNear(D), pts, ρ_IDPDL, X1; type=ComplexF64, quadstrat=BEAST.SingleNumTellesQStrat(3, 3))
pot_INPSL_telles = potential(HH2DSingleLayerNear(S), pts, ρ_INPSL, X1; type=ComplexF64, quadstrat=BEAST.SingleNumTellesQStrat(3, 3))
pot_INPDL_telles = potential(HH2DDoubleLayerNear(D), pts, ρ_INPDL, X1; type=ComplexF64, quadstrat=BEAST.SingleNumTellesQStrat(3, 3))

# Total field inside should be zero
err_IDPSL_pot = norm(pot_IDPSL + Φ_inc.(pts)) / norm(Φ_inc.(pts))
err_IDPDL_pot = norm(pot_IDPDL + Φ_inc.(pts)) / norm(Φ_inc.(pts))
err_INPSL_pot = norm(pot_INPSL + Φ_inc.(pts)) / norm(Φ_inc.(pts))
err_INPDL_pot = norm(pot_INPDL + Φ_inc.(pts)) / norm(Φ_inc.(pts))

err_IDPSL_pot_telles = norm(pot_IDPSL_telles + Φ_inc.(pts)) / norm(Φ_inc.(pts))
err_IDPDL_pot_telles = norm(pot_IDPDL_telles + Φ_inc.(pts)) / norm(Φ_inc.(pts))
err_INPSL_pot_telles = norm(pot_INPSL_telles + Φ_inc.(pts)) / norm(Φ_inc.(pts))
err_INPDL_pot_telles = norm(pot_INPDL_telles + Φ_inc.(pts)) / norm(Φ_inc.(pts))

@test err_IDPSL_pot > err_IDPSL_pot_telles
@test err_IDPDL_pot > err_IDPDL_pot_telles
@test err_INPSL_pot > err_INPSL_pot_telles
@test err_INPDL_pot > err_INPDL_pot_telles
end
