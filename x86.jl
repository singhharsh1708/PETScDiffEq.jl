using PETScDiffEq, SciMLBase
function adj_f!(du, u, p, t)
    du[1] = -p[1] * u[1] + p[2] * u[1] * u[2]
    du[2] = p[3] * u[1] - p[4] * u[2]^2 + p[1] * sin(t)
    nothing
end
function adj_jac!(J, u, p, t)
    J[1, 1] = -p[1] + p[2] * u[2]; J[1, 2] = p[2] * u[1]; J[2, 1] = p[3]; J[2, 2] = -2 * p[4] * u[2]
    nothing
end
function adj_paramjac!(pJ, u, p, t)
    fill!(pJ, 0.0); pJ[1, 1] = -u[1]; pJ[1, 2] = u[1] * u[2]; pJ[2, 1] = sin(t); pJ[2, 3] = u[1]; pJ[2, 4] = -u[2]^2
    nothing
end
prob = ODEProblem(ODEFunction(adj_f!; jac = adj_jac!, paramjac = adj_paramjac!), [1.0, 0.5], (0.0, 1.0), [0.7, 0.3, 0.4, 0.2])
grad(opts) = PETScDiffEq._discrete_adjoint(prob, TSRK("4"), PETScAdjoint(; petsc_options = opts);
    t = collect(0.0:0.1:1.0), dgdu_discrete = (out, u, p, t, i) -> (out .= u; nothing), dt = 0.01, adaptive = false)
run(name, f) = try
    r = f(); println(rpad(name, 46), "OK ", r[2])
catch e
    println(rpad(name, 46), "THREW ", first(split(sprint(showerror, e), '\n'))[1:min(end, 90)])
end
println("WORD_SIZE=", Sys.WORD_SIZE, " ", ENV["REF"], " phase=", ARGS[1])
if ARGS[1] == "fresh"
    run("memory trajectory", () -> grad(String[]))
    run("disk trajectory", () -> grad(["-ts_trajectory_type", "basic"]))
elseif ARGS[1] == "after-f32"
    s = solve(ODEProblem((du, u, p, t) -> (du .= -u; nothing), Float32[1.0], (0.0f0, 1.0f0)), TSRK("4"); dt = 0.1f0, adaptive = false)
    println("float32 solve: ", eltype(s.u[end]), " ", s.retcode)
    run("disk trajectory after Float32", () -> grad(["-ts_trajectory_type", "basic"]))
    run("disk trajectory again", () -> grad(["-ts_trajectory_type", "basic"]))
elseif ARGS[1] == "after-complex"
    try
        s = solve(ODEProblem((du, u, p, t) -> (du .= -im .* u; nothing), ComplexF64[1.0], (0.0, 1.0)), TSRK("4"); dt = 0.1, adaptive = false)
        println("complex solve: ", eltype(s.u[end]), " ", s.retcode)
    catch e
        println("complex solve THREW ", first(split(sprint(showerror, e), '\n'))[1:min(end, 80)])
    end
    run("disk trajectory after complex", () -> grad(["-ts_trajectory_type", "basic"]))
end
