using Printf

const N = 2_000_000

mutable struct AtomicCounter
    @atomic n::Int
end

@noinline function increment!(c::Vector{Int})
    c[1] += 1
    return nothing
end

function count_nosync(n, p)
    c = [0]
    Threads.@threads :static for k in 1:p
        for _ in 1:div(n, p)
            increment!(c)
        end
    end
    return c[1]
end

function count_atomic(n, p)
    c = Threads.Atomic{Int}(0)
    Threads.@threads :static for k in 1:p
        for _ in 1:div(n, p)
            Threads.atomic_add!(c, 1)
        end
    end
    return c[]
end

function count_atomic_field(n, p)
    c = AtomicCounter(0)
    Threads.@threads :static for k in 1:p
        for _ in 1:div(n, p)
            @atomic c.n += 1
        end
    end
    return @atomic c.n
end

function count_lock(n, p)
    c = [0]
    l = ReentrantLock()
    Threads.@threads :static for k in 1:p
        for _ in 1:div(n, p)
            lock(l)
            increment!(c)
            unlock(l)
        end
    end
    return c[1]
end

function count_spinlock(n, p)
    c = [0]
    l = Threads.SpinLock()
    Threads.@threads :static for k in 1:p
        for _ in 1:div(n, p)
            lock(l)
            increment!(c)
            unlock(l)
        end
    end
    return c[1]
end

function count_local(n, p)
    tasks = [Threads.@spawn begin
                 c = [0]
                 for _ in 1:div(n, p)
                     increment!(c)
                 end
                 c[1]
             end for k in 1:p]
    return sum(fetch, tasks)
end

function measure(f, n, p; runs = 3)
    f(1000p, p)
    result = 0
    best = Inf
    for _ in 1:runs
        t0 = time_ns()
        result = f(n, p)
        best = min(best, (time_ns() - t0) / 1e9)
    end
    return result, best
end

function main()
    p = Threads.nthreads()
    expected = div(N, p) * p
    println("Example 3: ", expected, " increments of a shared counter, threads = ", p)
    println("Julia ", VERSION, ", CPU: ", Sys.cpu_info()[1].model)
    println()
    @printf("%-34s %10s %10s %8s\n", "Method", "time, ms", "counter", "correct")
    methods = [
        ("No synchronization", count_nosync),
        ("Threads.Atomic + atomic_add!", count_atomic),
        ("@atomic field", count_atomic_field),
        ("ReentrantLock", count_lock),
        ("Threads.SpinLock", count_spinlock),
        ("Local counters + fetch", count_local),
    ]
    for (name, f) in methods
        r, t = measure(f, N, p)
        @printf("%-34s %10.1f %10d %8s\n", name, 1000t, r, r == expected ? "yes" : "no")
    end
end

main()
