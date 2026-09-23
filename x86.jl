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
prob = ODEProblem(ODEFunction(rober!; jac = rober_jac!), [1.0, 0.0, 0.0], (0.0, 1.0e11))
run(name, alg; kw...) = try
    s = solve(prob, alg; abstol = 1e-10, reltol = 1e-6, kw...)
    println(rpad(name, 44), s.retcode, " t_end=", s.t[end], " naccept=", s.stats.naccept)
catch e
    println(rpad(name, 44), "THREW ", first(split(sprint(showerror, e), '\n'))[1:min(end, 90)])
end
println("WORD_SIZE=", Sys.WORD_SIZE)
run("bdf default", TSImplicit("bdf"))
run("bdf -pc_type lu", TSImplicit("bdf", ["-pc_type", "lu"]))
run("bdf -ksp_type preonly -pc_type lu", TSImplicit("bdf", ["-ksp_type", "preonly", "-pc_type", "lu"]))
run("bdf -pc_type none", TSImplicit("bdf", ["-pc_type", "none"]))
run("bdf save_everystep=false", TSImplicit("bdf"); save_everystep = false)

println("---- adapt monitor, last lines")
out = mktemp() do path, io
    redirect_stdout(io) do
        try
            solve(prob, TSImplicit("bdf", ["-ts_adapt_monitor", "-ts_monitor"]); abstol = 1e-10, reltol = 1e-6)
        catch
        end
        Libc.flush_cstdio()
    end
    flush(io)
    read(path, String)
end
for l in split(out, '\n')[max(1, end - 14):end]
    println(l)
end
