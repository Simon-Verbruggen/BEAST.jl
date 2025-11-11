# # Check MtE operator
# This example checks the implementation of the MtE (Magnetic-to-Electric) operator.
# For a PEC sphere, the jump condition dictates: ``\mathbf{n} \times \mathbf{H} = 0 = \mathbf{M}``.
# Thus, the on-surface EtM (electric-to-magnetic) operator which maps ``\mathbf{J}`` to ``\mathbf{M}`` on a surface, should map ``\mathbf{J}`` to ``0`` on the PEC sphere.
#
# We will begin by constructing a MtE map and inverting it to obtain the EtM map. Then the convergence of the norm of ``MtE * J`` is checked, which should approach ``\mathbf{M}=0``.
# Note: with this current check the MtE map can still be implemented wrong, up to a prefactor. (But for preconditioning purposes, this prefactor is not of importance)
using CompScienceMeshes, BEAST
using Makeitso
using LinearAlgebra

@target geo (;h) -> begin
      (; Γ = CompScienceMeshes.meshsphere(radius=1.0, h=h))       # unit sphere
end

@target solution_EFIE (geo,; κ) -> begin
    Γ = geo.Γ

    X = raviartthomas(Γ);

    t = Maxwell3D.singlelayer(wavenumber=κ);
    E = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ);
    e = (n × E) × n;

    @hilbertspace j;
    @hilbertspace k;
    efie = @discretise t[k,j]==e[k]  j∈X k∈X;

    u, ch = BEAST.gmres_ch(efie; restart=1500);
    return (;current=u)
end

@target OSRC_preconditioner (geo,;κ, Np, curvature) -> begin
      MtE_map = BEAST.MtE_operator(geo.Γ, κ, Np, pi/2, curvature=curvature)
      return (;MtE=MtE_map)
end

@target check_MtE_map (solution_EFIE, OSRC_preconditioner;) -> begin
    MtE_map = OSRC_preconditioner.MtE
    u = solution_EFIE.current

    EtM_matrix = inv(Matrix(MtE_map))           # invert the map to EtM operator
    M = EtM_matrix*u                            # the magnetic current M
    err = LinearAlgebra.norm(M)/size(M)[1]
    return err
end

# convergence in function of h and Np (for high and low frequencies)
@sweep sweep_MtE_map (!check_MtE_map,; h=[], κ=[], Np=[], curvature=[]) -> (;sol=check_MtE_map,)
Np_values = [2, 3, 4, 6]
h_values = [0.5, 0.3, 0.2]

h_conv_high = make(sweep_MtE_map; h=h_values, κ=[pi*1.0], Np=[2], curvature=[1.0])
h_conv_low = make(sweep_MtE_map; h=h_values, κ=[pi/10], Np=[2], curvature=[1.0])

Np_conv_high = make(sweep_MtE_map; h=0.3, κ=[pi*1.0], Np=Np_values, curvature=[1.0])
Np_conv_low = make(sweep_MtE_map; h=0.3, κ=[pi/10], Np=Np_values, curvature=[1.0])