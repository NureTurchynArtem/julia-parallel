using Printf

const W = 2400
const H = 1800
const MAXITER = 1000
const XMIN, XMAX = -2.2, 1.0
const YMIN, YMAX = -1.2, 1.2

@inline function escape_time(cr::Float64, ci::Float64, maxiter::Int)
    zr = 0.0
    zi = 0.0
    for k in 1:maxiter
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > 4.0
            return k - 1
        end
        zi = 2.0 * zr * zi + ci
        zr = zr2 - zi2 + cr
    end
    return maxiter
end

function render_row!(iters, j)
    ci = YMAX - (j - 1) * (YMAX - YMIN) / (H - 1)
    for i in 1:W
        cr = XMIN + (i - 1) * (XMAX - XMIN) / (W - 1)
        iters[i, j] = escape_time(cr, ci, MAXITER)
    end
end

function mandel_seq!(iters)
    for j in 1:H
        render_row!(iters, j)
    end
end

function mandel_blocks!(iters, p, times)
    Threads.@threads :static for k in 1:p
        t0 = time_ns()
        for j in (div((k - 1) * H, p) + 1):div(k * H, p)
            render_row!(iters, j)
        end
        times[k] = (time_ns() - t0) / 1e6
    end
end

function mandel_cyclic!(iters, p)
    Threads.@threads :static for k in 1:p
        for j in k:p:H
            render_row!(iters, j)
        end
    end
end

function mandel_greedy!(iters)
    Threads.@threads :greedy for j in 1:H
        render_row!(iters, j)
    end
end

function mandel_spawn!(iters, rows)
    @sync for lo in 1:rows:H
        hi = min(lo + rows - 1, H)
        Threads.@spawn for j in lo:hi
            render_row!(iters, j)
        end
    end
end

function measure(f, args...; runs = 3)
    f(args...)
    best = Inf
    for _ in 1:runs
        t0 = time_ns()
        f(args...)
        best = min(best, (time_ns() - t0) / 1e9)
    end
    return best
end

function color(it, maxiter)
    it >= maxiter && return (0x00, 0x00, 0x00)
    t = log(it + 1) / log(maxiter + 1)
    r = 9.0 * (1 - t) * t^3
    g = 15.0 * (1 - t)^2 * t^2
    b = 8.5 * (1 - t)^3 * t
    return (round(UInt8, 255 * min(r, 1.0)), round(UInt8, 255 * min(g, 1.0)), round(UInt8, 255 * min(b, 1.0)))
end

function save_bmp(path, iters)
    rowsize = (3W + 3) & ~3
    open(path, "w") do io
        write(io, UInt8('B'), UInt8('M'), UInt32(54 + rowsize * H), UInt32(0), UInt32(54))
        write(io, UInt32(40), Int32(W), Int32(H), UInt16(1), UInt16(24), UInt32(0),
              UInt32(rowsize * H), Int32(2835), Int32(2835), UInt32(0), UInt32(0))
        line = zeros(UInt8, rowsize)
        for j in H:-1:1
            for i in 1:W
                r, g, b = color(iters[i, j], MAXITER)
                line[3i - 2] = b
                line[3i - 1] = g
                line[3i] = r
            end
            write(io, line)
        end
    end
end

function print_row(name, t, t1, p, equal)
    s = t1 / t
    @printf("%-28s %10.1f %7.2f %7.2f %7s\n", name, 1000t, s, s / p, equal ? "yes" : "no")
end

function main()
    nt = Threads.nthreads()
    println("Example 2: Mandelbrot set ", W, "x", H, ", max iterations = ", MAXITER)
    println("Julia ", VERSION, ", threads = ", nt, ", CPU: ", Sys.cpu_info()[1].model)
    println()
    @printf("%-28s %10s %7s %7s %7s\n", "Method", "time, ms", "S", "E", "equal")

    ref = zeros(Int32, W, H)
    t1 = measure(mandel_seq!, ref)
    print_row("Sequential", t1, t1, 1, true)

    a = zeros(Int32, W, H)
    times = zeros(nt)
    t = measure(mandel_blocks!, a, nt, times)
    print_row("@threads :static, blocks", t, t1, nt, a == ref)
    block_times = copy(times)

    a = zeros(Int32, W, H)
    t = measure(mandel_cyclic!, a, nt)
    print_row("@threads :static, cyclic", t, t1, nt, a == ref)

    a = zeros(Int32, W, H)
    t = measure(mandel_greedy!, a)
    print_row("@threads :greedy", t, t1, nt, a == ref)

    a = zeros(Int32, W, H)
    t = measure(mandel_spawn!, a, 8)
    print_row("@spawn, 8 rows per task", t, t1, nt, a == ref)

    println()
    println("Time of each block of rows (static blocks), ms:")
    println("  ", join([@sprintf("%.1f", x) for x in block_times], "  "))
    @printf("  max / min = %.1f\n", maximum(block_times) / minimum(block_times))

    save_bmp("mandelbrot.bmp", ref)
    println()
    println("Image saved: mandelbrot.bmp")
end

main()
