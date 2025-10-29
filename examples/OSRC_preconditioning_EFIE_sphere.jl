# # OSRC preconditioning
# In this example, the EFIE is preconditioned with the OSRC MtE (magnetic-to-electric) operator.
# In particular, we will consider the PEC unit sphere as benchmark and compare the OSRC preconditioner with the Calderon preconditioner (in terms of number of iterations during a GMRES solve).
# More details about this benchmark can be found in the paper <a href="https://arxiv.org/abs/2111.10761" target='new'> An OSRC Preconditioner for the EFIE (Betcke et al. (2021))</a>.

using CompScienceMeshes, BEAST
using Makeitso
using LinearAlgebra

include(joinpath(@__DIR__, "..","src", "operators", "OSRC.jl"))       # TODO: properly export

@target geo (;h) -> begin
      (; Γ = CompScienceMeshes.meshsphere(radius=1.0, h=h))       # unit sphere
end

@target OSRC_preconditioner (geo,;κ, Np) -> begin
      MtE_map = MtE_operator(geo.Γ, κ, Np, pi/2)
      return (;MtE=MtE_map)
end

@target calderon_preconditioner (geo,;κ) -> begin
      Γ = geo.Γ

      X = raviartthomas(Γ)
      Y = BEAST.buffachristiansen(Γ)

      T = Maxwell3D.singlelayer(wavenumber=κ)
      N = NCross()

      Tyy = assemble(T,Y,Y);
      Nxy = Matrix(assemble(N,X,Y));
      iNxy = inv(Nxy);
      P = iNxy' * Tyy * iNxy
      return (;preconditioner=P)
end


@target formulation_EFIE (geo,; κ) -> begin
      Γ = geo.Γ

      X = raviartthomas(Γ)

      T = Maxwell3D.singlelayer(wavenumber=κ)

      E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ);
      e = (n × E) × n;    # right hand side of equation

      bx = assemble(e, X)
      A = assemble(T,X,X); 
      return (;bilforms=(;A), linforms=(;bx))
end

# Solve the EFIE in 3 different ways
# 1) Without preconditioning:
@target solution_EFIE (formulation_EFIE,; residual) -> begin
      (;bilforms, linforms) = formulation_EFIE
      (;A) = bilforms; (;bx) = linforms;
      iT = BEAST.GMRESSolver(A; restart=1_500, abstol=residual, reltol=residual, maxiter=1_500)
      u, ch = BEAST.solve(iT, bx)
      return (;iters=ch.iters)
end

# 2) With OSRC preconditioning:
@target solution_OSRC_precond_EFIE (formulation_EFIE, OSRC_preconditioner; residual) -> begin
      (;bilforms, linforms) = formulation_EFIE
      (;A) = bilforms; (;bx) = linforms;
      P_OSRC = OSRC_preconditioner.MtE
      iT = BEAST.GMRESSolver(A; restart=1_500, abstol=residual, reltol=residual, maxiter=1_500, left_preconditioner=P_OSRC)
      u, ch = BEAST.solve(iT, bx)
      return (;iters=ch.iters)
end

# 3) With Calderon preconditioning:
@target solution_calderon_precond_EFIE (formulation_EFIE, calderon_preconditioner; residual) -> begin
      (;bilforms, linforms) = formulation_EFIE
      (;A) = bilforms; (;bx) = linforms;
      P_calderon = calderon_preconditioner.preconditioner
      iT = BEAST.GMRESSolver(A; restart=1_500, abstol=residual, reltol=residual, maxiter=1_500, left_preconditioner=P_calderon)
      u, ch = BEAST.solve(iT, bx)
      return (;iters=ch.iters)
end

# Determine the number of iterations for the different solution methods, in low and high frequency regime
@target solution_comparison (solution_EFIE, solution_OSRC_precond_EFIE, solution_calderon_precond_EFIE;) -> begin
      return (solution_EFIE.iters, solution_OSRC_precond_EFIE.iters,  solution_calderon_precond_EFIE.iters)
end

@sweep OSRC_sweep_discretizations (!solution_comparison,; h=[], κ=[], residual=[], Np=[]) -> (;sol=solution_comparison,)

h_values = [0.5, 0.3, 0.15]
low_frequency_κ = pi/10
high_frequency_κ = pi*1.0
# TODO: probably just one discretization (0.15 too slow now)
sol_low_freq_1e6_Np6 = make(OSRC_sweep_discretizations; h=h_values, κ=[low_frequency_κ], residual=[1e-6], Np=6)
sol_high_freq_1e6_Np6 = make(OSRC_sweep_discretizations; h=h_values, κ=[high_frequency_κ], residual=[1e-6], Np=6)