using PETScDiffEq, SciMLBase, Logging
decay!(du, u, p, t) = (du .= -u; nothing)
function adj_f!(du, u, p, t)
    du[1] = -p[1] * u[1] + p[2] * u[1] * u[2]
    du[2] = p[3] * u[1] - p[4] * u[2]^2 + p[1] * sin(t)
    nothing
end
adj_jac!(J, u, p, t) = (J[1, 1] = -p[1] + p[2] * u[2]; J[1, 2] = p[2] * u[1]; J[2, 1] = p[3]; J[2, 2] = -2 * p[4] * u[2]; nothing)
function adj_paramjac!(pJ, u, p, t)
    fill!(pJ, 0.0); pJ[1, 1] = -u[1]; pJ[1, 2] = u[1] * u[2]; pJ[2, 1] = sin(t); pJ[2, 3] = u[1]; pJ[2, 4] = -u[2]^2
    nothing
end
aprob = ODEProblem(ODEFunction(adj_f!; jac = adj_jac!, paramjac = adj_paramjac!), [1.0, 0.5], (0.0, 1.0), [0.7, 0.3, 0.4, 0.2])
disk() = try
    r = PETScDiffEq._discrete_adjoint(aprob, TSRK("4"), PETScAdjoint(; petsc_options = ["-ts_trajectory_type", "basic"]);
        t = collect(0.0:0.1:1.0), dgdu_discrete = (out, u, p, t, i) -> (out .= u; nothing), dt = 0.01, adaptive = false)
    println("  disk adjoint OK ", r[2][1])
catch e
    println("  disk adjoint THREW ", first(split(sprint(showerror, e), '\n'))[1:min(end, 80)])
end
span(S) = (zero(real(S)), one(real(S)))
never = (dt, u, p, t) -> false
runs = (
    S -> solve(ODEProblem(decay!, S[1, 2], span(S)), TSRK("5dp"); dtmin = real(S)(1.0e-6), unstable_check = never),
    S -> solve(ODEProblem(decay!, S[1, 2], span(S)), TSMPRK([1], "p2"); dt = real(S)(0.01)),
    S -> solve(ODEProblem((du, u, p, t) -> (du[1] = 8 * u[1]; nothing), S[1], span(S)), TSImplicit("beuler"); dt = real(S)(0.125), adaptive = false),
)
scalars = (Float32, Float64, ComplexF32, ComplexF64)
q(f) = with_logger(f, NullLogger())
ph = ARGS[1]
println("WORD_SIZE=", Sys.WORD_SIZE, " phase=", ph)
if ph == "interleave"
    for _ in 1:2, run in runs, S in scalars
        q(() -> run(S))
    end
    disk()
elseif ph == "pivots"
    for S in scalars
        println("  pivot ", S, " ", q(() -> runs[3](S)).retcode)
    end
    disk()
elseif ph == "pivot64"
    println("  pivot Float64 ", q(() -> runs[3](Float64)).retcode)
    disk()
elseif ph == "nopivot"
    for run in runs[1:2], S in scalars
        q(() -> run(S))
    end
    disk()
elseif ph == "stops"
    for span in ((0.0f0, 1.0f6), (0.0f0, 1.0f5)), alg in (TSImplicit("bdf"), TSRosW(), TSARKIMEX("3"))
        stops = [span[2] / 7 * k for k in 1:6]
        sol = q(() -> solve(ODEProblem(decay!, Float32[1], span), alg; tstops = stops, abstol = 1.0f-5, reltol = 1.0f-4))
        println("  ", rpad("$(span[2]) $(nameof(typeof(alg)))", 24), sol.retcode, " t_end=", sol.t[end], " n=", length(sol.t))
    end
    sol = solve(ODEProblem(decay!, [1.0], (0.0, 1.0e6)), TSImplicit("bdf"); tstops = [1.0e6 / 7 * k for k in 1:6], abstol = 1e-5, reltol = 1e-4)
    println("  control Float64 bdf ", sol.retcode, " t_end=", sol.t[end])
end
