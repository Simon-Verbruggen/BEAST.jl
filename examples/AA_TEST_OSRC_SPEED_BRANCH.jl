using CompScienceMeshes, BEAST
using LinearAlgebra

# values
h = 0.05
κ = pi*1.0
residual = 1e-6
Np = 4
curvature = 1.0

num_iters_test = 100


geo = (; Γ = CompScienceMeshes.meshsphere(radius=1.0, h=h))

_, assembly_time_EFIE, bytes, alloc, gctime = @timed begin
    Γ = geo.Γ

    X = raviartthomas(Γ)

    T = Maxwell3D.singlelayer(wavenumber=κ)

    E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ);
    e = (n × E) × n;

    bx = assemble(e, X)
    A = assemble(T,X,X); 
end
formulation_EFIE = (;bilforms=(;A), linforms=(;bx), assembly_time_EFIE=assembly_time_EFIE)

#### ASSEMBLE OSRC preconditioner #### 


_, assembly_time_EFIE, bytes, alloc, gctime = @timed MtE_map = BEAST.MtE_operator_lu(geo.Γ, κ, Np, pi/2, curvature=curvature)
OSRC_lu_preconditioner = (;MtE=MtE_map, assembly_time=0.0)


#### SOLVE EFIE

(;bilforms, linforms, assembly_time_EFIE) = formulation_EFIE
(;A) = bilforms; (;bx) = linforms;
iT = BEAST.GMRESSolver(A; restart=1_500, abstol=residual, reltol=residual, maxiter=1_500)
_, solving_time, bytes, alloc, gctime = @timed u, ch = BEAST.solve(iT, bx)
solution_EFIE = (;iters=ch.iters, assembly_time=0 + assembly_time_EFIE, solving_time=solving_time)

#### SOLVE OSCR #### 

(;bilforms, linforms, assembly_time_EFIE) = formulation_EFIE
(;A) = bilforms; (;bx) = linforms;
P_OSRC = OSRC_lu_preconditioner.MtE
iT = BEAST.GMRESSolver(A; restart=1_500, abstol=residual, reltol=residual, maxiter=1_500, left_preconditioner=P_OSRC)
_, solving_time, bytes, alloc, gctime = @timed u, ch = BEAST.solve(iT, bx)
solution_OSRC_lu_precond_EFIE = (;iters=ch.iters, assembly_time=OSRC_lu_preconditioner.assembly_time + assembly_time_EFIE, solving_time=solving_time)

time_per_iter_OSRC = solution_OSRC_lu_precond_EFIE.solving_time/ch.iters
time_per_iter_EFIE = solution_EFIE.solving_time/ch.iters

##### iter time, strict ####

println(time_per_iter_OSRC)
println(time_per_iter_EFIE)