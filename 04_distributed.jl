using Distributed
using Printf

const N = 400_000_000
const NPROCS = parse(Int, get(ARGS, 1, string(Sys.CPU_THREADS)))

t0 = time_ns()
addprocs(NPROCS)
const START_TIME = (time_ns() - t0) / 1e9

@everywhere function partial_sum(lo::Int, hi::Int, h::Float64)
    s = 0.0
    for i in lo:hi
        x = (i - 0.5) * h
        s += 4.0 / (1.0 + x * x)
    end
    return s
end

@everywhere chunk(n, p, k) = (div((k - 1) * n, p) + 1):div(k * n, p)

pi_local(n) = partial_sum(1, n, 1.0 / n) / n

function pi_remotecall(n, ws)
    h = 1.0 / n
    p = length(ws)
    futures = [remotecall(partial_sum, ws[k], first(chunk(n, p, k)), last(chunk(n, p, k)), h)
               for k in 1:p]
    return sum(fetch, futures) * h
end

function pi_distributed(n)
    h = 1.0 / n
    s = @distributed (+) for i in 1:n
        x = (i - 0.5) * h
        4.0 / (1.0 + x * x)
    end
    return s * h
end

function pi_pmap(n, nparts)
    h = 1.0 / n
    sums = pmap(k -> partial_sum(first(chunk(n, nparts, k)), last(chunk(n, nparts, k)), h), 1:nparts)
    return sum(sums) * h
end

function measure(f, args...; runs = 3)
    result = f(args...)
    best = Inf
    for _ in 1:runs
        t0 = time_ns()
        result = f(args...)
        best = min(best, (time_ns() - t0) / 1e9)
    end
    return result, best
end

function print_row(name, p, t, t1, r)
    s = t1 / t
    @printf("%-26s %9d %10.1f %7.2f %7.2f %11.2e\n", name, p, 1000t, s, s / p, abs(r - pi))
end

function main()
    ws = workers()
    np = length(ws)
    println("Example 4: Distributed, pi by the midpoint rule, n = ", N)
    @printf("Julia %s, worker processes = %d, start of the processes: %.2f s\n", VERSION, np, START_TIME)
    println()
    @printf("%-26s %9s %10s %7s %7s %11s\n", "Method", "processes", "time, ms", "S", "E", "error")

    r, t1 = measure(pi_local, N)
    print_row("Sequential (main process)", 1, t1, t1, r)

    p = 1
    while p < np
        r, t = measure(pi_remotecall, N, ws[1:p])
        print_row("remotecall + fetch", p, t, t1, r)
        p *= 2
    end
    r, t = measure(pi_remotecall, N, ws)
    print_row("remotecall + fetch", np, t, t1, r)

    r, t = measure(pi_distributed, N)
    print_row("@distributed (+)", np, t, t1, r)

    r, t = measure(pi_pmap, N, 8np)
    print_row("pmap, $(8np) parts", np, t, t1, r)
end

main()
