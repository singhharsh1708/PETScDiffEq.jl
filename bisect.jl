# Runs the header of test/runtests.jl, every non-testset statement of its outer testset, and the
# first `cut` inner testsets (each isolated so a failure does not abort), then the disk adjoint.
using Test
src = read(joinpath(pwd(), "test", "runtests.jl"), String)
top = Meta.parseall(src)
outer_i = findfirst(a -> a isa Expr && a.head === :macrocall && a.args[1] === Symbol("@testset"), top.args)
header = top.args[1:outer_i-1]
outer = top.args[outer_i]
body = outer.args[end]
items = [a for a in body.args if !(a isa LineNumberNode)]
is_ts(a) = a isa Expr && a.head === :macrocall && a.args[1] === Symbol("@testset")
tsidx = findall(is_ts, items)
if length(ARGS) > 0 && ARGS[1] == "count"
    println("testsets: ", length(tsidx))
    for (k, i) in enumerate(tsidx)
        nm = items[i].args[3]
        println(k, " ", nm isa String ? nm : string(nm)[1:min(end, 60)])
    end
    exit()
end
cut = parse(Int, ARGS[1])
cd(joinpath(pwd(), "test"))
for h in header
    Core.eval(Main, h)
end
ts_seen = 0
for a in items
    if is_ts(a)
        ts_seen += 1
        ts_seen > cut && continue
    end
    try
        redirect_stdout(devnull) do
            Core.eval(Main, a)
        end
    catch e
        println("item error: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 80)])
    end
end
println("ran ", min(cut, length(tsidx)), " of ", length(tsidx), " testsets")
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
aprob = SciMLBase.ODEProblem(SciMLBase.ODEFunction(adj_f!; jac = adj_jac!, paramjac = adj_paramjac!), [1.0, 0.5], (0.0, 1.0), [0.7, 0.3, 0.4, 0.2])
try
    r = PETScDiffEq._discrete_adjoint(aprob, PETScDiffEq.TSRK("4"), PETScDiffEq.PETScAdjoint(; petsc_options = ["-ts_trajectory_type", "basic"]);
        t = collect(0.0:0.1:1.0), dgdu_discrete = (out, u, p, t, i) -> (out .= u; nothing), dt = 0.01, adaptive = false)
    println("CUT=", cut, " disk adjoint OK ", r[2][1])
catch e
    println("CUT=", cut, " disk adjoint THREW ", first(split(sprint(showerror, e), '\n'))[1:min(end, 80)])
end
