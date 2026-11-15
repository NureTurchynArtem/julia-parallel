using Printf

const N = 400_000_000

function partial_sum(lo::Int, hi::Int, h::Float64)
    s = 0.0
    for i in lo:hi
        x = (i - 0.5) * h
        s += 4.0 / (1.0 + x * x)
    end
    return s
end

chunk(n, p, k) = (div((k - 1) * n, p) + 1):div(k * n, p)

pi_seq(n) = partial_sum(1, n, 1.0 / n) / n

function pi_threads(n, p)
    h = 1.0 / n
    sums = zeros(p)
    Threads.@threads :static for k in 1:p
        r = chunk(n, p, k)
        sums[k] = partial_sum(first(r), last(r), h)
    end
    return sum(sums) * h
end

function pi_tasks(n, ntasks)
    h = 1.0 / n
    tasks = [Threads.@spawn partial_sum(first(chunk(n, ntasks, k)), last(chunk(n, ntasks, k)), h)
             for k in 1:ntasks]
    return sum(fetch, tasks) * h
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
    @printf("%-24s %7d %10.1f %7.2f %7.2f %11.2e\n", name, p, 1000t, s, s / p, abs(r - pi))
end

function thread_counts(nt)
    ps = Int[]
    p = 1
    while p < nt
        push!(ps, p)
        p *= 2
    end
    push!(ps, nt)
    return ps
end

function main()
    nt = Threads.nthreads()
    println("Example 1: pi as integral of 4/(1+x^2) on [0, 1], midpoint rule, n = ", N)
    println("Julia ", VERSION, ", threads = ", nt, ", CPU threads = ", Sys.CPU_THREADS,
            ", CPU: ", Sys.cpu_info()[1].model)
    println()
    @printf("%-24s %7s %10s %7s %7s %11s\n", "Method", "threads", "time, ms", "S", "E", "error")

    r, t1 = measure(pi_seq, N)
    print_row("Sequential", 1, t1, t1, r)

    for p in thread_counts(nt)
        r, t = measure(pi_threads, N, p)
        print_row("@threads :static", p, t, t1, r)
    end

    for ntasks in (nt, 8nt)
        r, t = measure(pi_tasks, N, ntasks)
        print_row("@spawn, $ntasks tasks", nt, t, t1, r)
    end
end

main()
