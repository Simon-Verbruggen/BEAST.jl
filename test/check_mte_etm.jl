using BEAST, CompScienceMeshes
using LinearAlgebra
using SparseArrays

Np = 3
curvature = 1.0
h = 0.5
κ = pi*1.0

Γ = CompScienceMeshes.meshsphere(radius=1.0, h=h)
Nd = BEAST.nedelec(Γ)
X = BEAST.raviartthomas(Γ)
MtE = BEAST.MtE_OSRC_op(κ, Np, pi/2, curvature)
EtM = BEAST.EtM_OSRC_op(κ, Np, pi/2, curvature)

MtE_map = assemble(MtE, Nd, Nd)
EtM_map = assemble(EtM, Nd, Nd)

G = assemble(BEAST.Identity(), X, X)
Identity_check1 = - G * EtM_map * MtE_map;
Identity_check2 = - MtE_map * G * EtM_map;


b = rand(numfunctions(X))
x = Identity_check1*b
@show norm(b - x)

x = Identity_check2*b
@show norm(b - x)


