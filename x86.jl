using PETScDiffEq, SciMLBase
function rober!(du, u, p, t)
    du[1] = -0.04u[1] + 1.0e4 * u[2] * u[3]
    du[2] = 0.04u[1] - 1.0e4 * u[2] * u[3] - 3.0e7 * u[2]^2
    du[3] = 3.0e7 * u[2]^2
    nothing
end
function rober_jac!(J, u, p, t)
    J[1, 1] = -0.04; J[1, 2] = 1.0e4 * u[3]; J[1, 3] = 1.0e4 * u[2]
    J[2, 1] = 0.04; J[2, 2] = -1.0e4 * u[3] - 6.0e7 * u[2]; J[2, 3] = -1.0e4 * u[2]
    J[3, 1] = 0.0; J[3, 2] = 6.0e7 * u[2]; J[3, 3] = 0.0
    nothing
end
function resid!(r, du, u, p, t)
    r[1] = -0.04u[1] + 1.0e4 * u[2] * u[3] - du[1]
    r[2] = 0.04u[1] - 1.0e4 * u[2] * u[3] - 3.0e7 * u[2]^2 - du[2]
    r[3] = u[1] + u[2] + u[3] - 1.0
    nothing
end
function resid_jac!(J, du, u, p, gamma, t)
    J[1, 1] = -0.04 - gamma; J[1, 2] = 1.0e4 * u[3]; J[1, 3] = 1.0e4 * u[2]
    J[2, 1] = 0.04; J[2, 2] = -1.0e4 * u[3] - 6.0e7 * u[2] - gamma; J[2, 3] = -1.0e4 * u[2]
    J[3, 1] = 1.0; J[3, 2] = 1.0; J[3, 3] = 1.0
    nothing
end
run(name, f) = try
    s = redirect_stderr(devnull) do; f(); end
    println(rpad(name, 40), s.retcode, " t_end=", s.t[end], " naccept=", s.stats.naccept)
catch e
    println(rpad(name, 40), "THREW ", typeof(e), " ", first(split(sprint(showerror, e), '\n'))[1:min(end, 100)])
end
println("WORD_SIZE=", Sys.WORD_SIZE, " tree=", pathof(PETScDiffEq))
for (an, alg) in (("bdf", TSImplicit("bdf")), ("rosw", TSRosW()), ("arkimex4", TSARKIMEX("4")))
    for (fn, f) in (("jac", ODEFunction(rober!; jac = rober_jac!)), ("nojac", ODEFunction(rober!)))
        run("rober 1e11 $an $fn", () -> solve(ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1.0e11)), alg; abstol = 1e-10, reltol = 1e-6))
    end
end
for (fn, f) in (("jac", DAEFunction(resid!; jac = resid_jac!)), ("nojac", DAEFunction(resid!)))
    run("dae 1e5 bdf $fn", () -> solve(DAEProblem(f, [-0.04, 0.04, 0.0], [1.0, 0.0, 0.0], (0.0, 1.0e5); differential_vars = [true, true, false]), TSDAE(); abstol = 1e-10, reltol = 1e-8))
end
integ = init(ODEProblem(ODEFunction(rober!; jac = rober_jac!), [1.0, 0.0, 0.0], (0.0, 1.0e11)), TSImplicit("bdf"); abstol = 1e-10, reltol = 1e-6)
steps = []
try
    while integ.t < 1.0e11 && integ.sol.retcode == SciMLBase.ReturnCode.Default
        step!(integ); push!(steps, (integ.t, integ.dt))
    end
catch e
    println("integrator THREW ", first(split(sprint(showerror, e), '\n')))
end
for (t, dt) in steps[max(1, end - 3):end]
    println("step end t=", t, " next dt=", dt, " next end - tf=", t + dt - 1.0e11)
end
