package conv

import "base:runtime"
import "core:fmt"
import "core:math"
import "core:os"
import gd "../grid"

/*
  The potential is the steady state of the masked normalised convolution in goal.odin,

      u = keep * (K correlated with u) / max(K correlated with m, eps) + source

  and a flat sweep reaches it one stencil radius per pass. The pass budget flat needs is set
  by decay, not by the map: the residual of a truncated geometric chain is keep^T of the whole
  series, so a relative tolerance of 1e-4 costs about ln(1e-4)/-decay sweeps - 9200 passes at
  decay 1e-3, 4.6e6 at 2e-6, every one of them over 160000 cells. The screening length
  lambda = sqrt(rate / (4 * decay)) only says how far the field survives; it does not bound the
  chain, and a decay small enough to keep the field alive at the spawn band leaves the chain
  unreachably long (measured: decay 0.01, spawn band 414-641 cells out, field 1e-45, gradient
  direction meaningless).

  So the chain is summed at several scales at once instead of only at the native one: a Gaussian
  pyramid, the scale-space ladder pyrDown/pyrUp build in OpenCV. The map halved until it is too
  small to be worth subdividing, and the point of the ladder is the scale-space property - a
  3x3 stencil is local relative to a coarse cell, so the coarse field is already within one
  coarse cell of the fine one, and the long part of the chain (the constant mode, 1/decay terms)
  only has to be paid where the grid is tiny: 625 cells instead of 160000. Coarse solve, then
  prolong up, then repair at the finer rung, is the coarse-to-fine V-cycle of the multigrid
  literature, restriction and prolongation being its two named half-steps.

  Two things break under a naive ladder and both are enforced here rather than papered over:

  *  Coarsening a mask as a fraction opens corridors that are closed at the native scale, and
     then the coarse operator walks through walls that are not walls. A coarse cell is walkable
     only when every fine cell under it is, so the coarse field is a starting point and never the
     answer, and a coarse pool is allowed to close a corridor only in that direction.

  *  The operator is normalised, so restriction is a pair, not a ratio. Downsample the fraction
     conv(u)/conv(m) alone and you drift off the lattice; coarsen numerator and coverage
     separately and you keep the lattice and drift off the fraction. What is right by
     construction is the pair - value times mask and mask alone averaged the same way - and every
     rung's coverage comes from that rung's own mask, because coverage borrowed from the finer
     rung leaves that rung's walls only partly insulating: the row sums slip off one by the same
     slip and the field stops being monotone along the corridors.
*/

MIN_SIDE :: #config(PYRAMID_MIN_SIZE, 25)
MAX_LEVELS :: #config(PYRAMID_LEVELS, 8)

// A rung is swept until its own residual is under this fraction of its own peak, then handed
// up. The budget is a tolerance rather than a pass count for exactly that reason: the pass
// count a rung needs follows its cell count and its decay, and only the tolerance is meaningful
// across rungs. PYRAMID_SWEEP_CAP bounds the search so an unconverged rung costs time instead
// of hanging.
PYRAMID_TOL      :: #config(PYRAMID_TOL, 0.00005)
PYRAMID_SWEEP_CAP :: #config(PYRAMID_SWEEP_CAP, 2000000)

// Where the pass budget actually went. A ladder that is slow or imprecise is only diagnosable
// rung by rung, and the runner swallows stdout on a passing test, so the report is a file.
PYRAMID_REPORT :: #config(PYRAMID_REPORT, false)

Level :: struct {
    w, h:        int,
    cw, ch:      int,      // fine cells per coarse cell: this rung restricts the one below it
    mask, coverage, src: gd.Grid2D(f32),
    field, next: gd.Grid2D(f32), // the iterate, and the scratch the pass is writing into
}

// Restriction of a mask. A coarse cell is walkable only when every fine cell it covers is: a
// pool that averages the fraction opens the 2-cell corridors the native scale closes, and the
// coarse operator then runs through walls that exist only at the finer rung. Closing is allowed
// in that direction only, and the coarse field is then a starting point and never the answer.
@(private)
restrict_mask :: proc(fine: gd.Grid2D(f32), w, h: int, allocator := context.allocator) -> (out: gd.Grid2D(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    out = gd.make(f32, w, h, allocator) or_return

    cw := (fine.width + w - 1) / w
    ch := (fine.height + h - 1) / h

    for y in 0..<h {
        for x in 0..<w {
            x1 := min((x + 1) * cw, fine.width)
            y1 := min((y + 1) * ch, fine.height)
            open := true
            for j in (y * ch)..<y1 {
                for i in (x * cw)..<x1 {
                    if fine.data[i + j * fine.width] <= 0 {
                        open = false
                    }
                }
            }
            if open {
                out.data[x + y * w] = 1
            } else {
                out.data[x + y * w] = 0
            }
        }
    }

    return
}

// Restriction of the source: the numerator rule, source times mask summed over the block over
// the mask alone. A plain box average would move the splat onto cells that are not walkable,
// and the rung reads that as heat standing on a wall.
@(private)
restrict_src :: proc(fine, walkable: gd.Grid2D(f32), w, h: int, allocator := context.allocator) -> (out: gd.Grid2D(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    out = gd.make(f32, w, h, allocator) or_return

    cw := (fine.width + w - 1) / w
    ch := (fine.height + h - 1) / h

    for y in 0..<h {
        for x in 0..<w {
            x1 := min((x + 1) * cw, fine.width)
            y1 := min((y + 1) * ch, fine.height)
            total := f32(0)
            weight := f32(0)
            for j in (y * ch)..<y1 {
                for i in (x * cw)..<x1 {
                    at := i + j * fine.width
                    total += fine.data[at] * walkable.data[at]
                    weight += walkable.data[at]
                }
            }
            if weight > 0 {
                out.data[x + y * w] = total / weight
            } else {
                out.data[x + y * w] = 0
            }
        }
    }

    return
}

// Coverage of a rung, from that rung's own mask: the weight the stencil actually has over the
// walkable cells of that rung. Dividing by it is what makes the rung's walls inert taps rather
// than cold ones. Coverage copied up from a finer rung leaves this rung's walls only partly
// insulating, and the row sums slip off one by the same slip.
@(private)
coverage_of :: proc(kernel: Kernel, walkable: gd.Grid2D(f32), allocator := context.allocator) -> (out: gd.Grid2D(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    out = gd.make(f32, walkable.width, walkable.height, allocator) or_return

    for y in 0..<walkable.height {
        for x in 0..<walkable.width {
            out.data[x + y * walkable.width] = convolution_step(kernel, walkable, x, y, Border.CONSTANT)
        }
    }

    return
}

// Landing the source. The min-rule mask can close a coarse cell whose fine cells were mostly
// walkable, and then the restricted splat sits entirely on closed ground: the coarse solve is the
// zero problem, prolongation of a zero field is a zero field, and the fine rung starts over from
// scratch, which is what the ladder used to do (measured: rungs 2-4 converged in one pass at
// residual zero, so the coarse rungs bought nothing). Closed ground hands its source to the open
// cells it touches, which moves the splat at most one coarse cell, and the scale-space property
// already spends one coarse cell of displacement - same error budget, not a new one. Mass at
// closed ground that touches no open cell is counted and let go: at that scale the goal is not
// reachable, so there is no coarse answer to keep it in.
@(private)
land_source :: proc(ground: gd.Grid2D(f32), src: gd.Grid2D(f32), let_go: ^f32, allocator := context.allocator) -> (out: gd.Grid2D(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    let_go^ = 0
    w, h := ground.width, ground.height
    out = gd.make(f32, w, h, allocator) or_return

    // own ground first, so the additions below start from a defined value either way
    for y in 0..<h {
        for x in 0..<w {
            i := x + y * w
            if ground.data[i] > 0 {
                out.data[i] = src.data[i]
            } else {
                out.data[i] = 0
            }
        }
    }

    for y in 0..<h {
        for x in 0..<w {
            i := x + y * w
            if ground.data[i] > 0 || src.data[i] <= 0 do continue

            n := 0
            if x > 0 && ground.data[i - 1] > 0 { n += 1 }
            if x + 1 < w && ground.data[i + 1] > 0 { n += 1 }
            if y > 0 && ground.data[i - w] > 0 { n += 1 }
            if y + 1 < h && ground.data[i + w] > 0 { n += 1 }

            if n == 0 {
                let_go^ += src.data[i]
                continue
            }

            share := src.data[i] / f32(n)
            if x > 0 && ground.data[i - 1] > 0 { out.data[i - 1] += share }
            if x + 1 < w && ground.data[i + 1] > 0 { out.data[i + 1] += share }
            if y > 0 && ground.data[i - w] > 0 { out.data[i - w] += share }
            if y + 1 < h && ground.data[i + w] > 0 { out.data[i + w] += share }
        }
    }

    return
}

// Prolongation: the field of the rung below, pushed up into this rung's cells. Bilinear, so a
// coarse cell bleeds into the fine cells it touches instead of leaping out of its own square,
// then multiplied by this rung's mask: this rung may have closed a corridor the coarser field
// walked through, and a value that survives into a wall cell makes the next pass's residual
// meaningless rather than merely large.
@(private)
prolongate_into :: proc(from: gd.Grid2D(f32), walkable: gd.Grid2D(f32), out: gd.Grid2D(f32)) {
    assert(out.width == walkable.width && out.height == walkable.height, "prolongate: the output must be this rung")

    sx := f32(from.width) / f32(out.width)
    sy := f32(from.height) / f32(out.height)

    for y in 0..<out.height {
        for x in 0..<out.width {
            gx := (f32(x) + 0.5) * sx - 0.5
            gy := (f32(y) + 0.5) * sy - 0.5
            x0 := clamp(int(math.floor(gx)), 0, from.width - 1)
            y0 := clamp(int(math.floor(gy)), 0, from.height - 1)
            x1 := min(x0 + 1, from.width - 1)
            y1 := min(y0 + 1, from.height - 1)
            fx := clamp(gx, 0, f32(from.width - 1)) - f32(x0)
            fy := clamp(gy, 0, f32(from.height - 1)) - f32(y0)

            d := (1 - fx) * (1 - fy) * from.data[x0 + y0 * from.width]
            d += fx * (1 - fy) * from.data[x1 + y0 * from.width]
            d += (1 - fx) * fy * from.data[x0 + y1 * from.width]
            d += fx * fy * from.data[x1 + y1 * from.width]

            out.data[x + y * out.width] = walkable.data[x + y * out.width] * d
        }
    }
}

// One pass of the operator over one rung, and the residual it left behind, relative to the peak
// of the image just written. The residual is goal.odin's fixed-point equation read at this
// rung's own arrays - the same acceptance test pyramid_test.odin applies to the native grid, at
// every rung instead of only at the top of the ladder.
@(private)
sweep_once :: proc(kernel: Kernel, keep: f32, rung: ^Level) -> f32 {
    peak := f32(0)
    worst := f32(0)

    for y in 0..<rung.h {
        for x in 0..<rung.w {
            at := x + y * rung.w
            if rung.mask.data[at] <= 0 {
                rung.next.data[at] = 0
                continue
            }
            v := convolution_step(kernel, rung.field, x, y, Border.CONSTANT)
            c := max(rung.coverage.data[at], f32(1e-6))
            d := keep * v / c + rung.src.data[at]
            rung.next.data[at] = d
            if d > peak {
                peak = d
            }
            e := abs(d - rung.field.data[at])
            if e > worst {
                worst = e
            }
        }
    }

    return worst / max(peak, f32(1e-30))
}

// The V-cycle: restrict down to a rung small enough that the chain can be paid for outright,
// sum it there, prolong up, repair at each finer rung, and leave the converged potential in
// s.field. The native rung is the solver's own field and tmp, so the only extra memory is the
// rungs below it, whose cells total a third of the native grid.
solve :: proc(s: ^HeatSolver, source: gd.Grid2D(f32), allocator := context.allocator) -> runtime.Allocator_Error {
    assert(source.width == s.mask.width && source.height == s.mask.height, "solve: the source and the mask must be the same image")

    ladder: [MAX_LEVELS]Level
    dropped_of: [MAX_LEVELS]f32
    rungs := 0

    ladder[0] = Level{
        w        = s.mask.width,
        h        = s.mask.height,
        mask     = s.mask,
        coverage = s.coverage,
        src      = source,
        field    = s.field,
        next     = s.tmp,
    }
    rungs += 1

    for ladder[rungs - 1].w > MIN_SIDE || ladder[rungs - 1].h > MIN_SIDE {
        if rungs >= MAX_LEVELS do break

        fine := ladder[rungs - 1]
        w := max(fine.w / 2, 1)
        h := max(fine.h / 2, 1)

        mask, err_mask := restrict_mask(fine.mask, w, h, allocator)
        if err_mask != nil do return err_mask

        coverage, err_cov := coverage_of(s.kernel, mask, allocator)
        if err_cov != nil {
            delete(mask.data)
            return err_cov
        }

        src, err_src := restrict_src(fine.src, fine.mask, w, h, allocator)
        if err_src != nil {
            delete(mask.data)
            delete(coverage.data)
            return err_src
        }

        lost := f32(0)
        landed, err_landed := land_source(mask, src, &lost, allocator)
        if err_landed != nil {
            delete(mask.data)
            delete(coverage.data)
            delete(src.data)
            return err_landed
        }
        delete(src.data)
        src = landed
        dropped_of[rungs] = lost

        field, err_field := gd.make(f32, w, h, allocator)
        if err_field != nil {
            delete(mask.data)
            delete(coverage.data)
            delete(src.data)
            return err_field
        }

        next, err_next := gd.make(f32, w, h, allocator)
        if err_next != nil {
            delete(mask.data)
            delete(coverage.data)
            delete(src.data)
            delete(field.data)
            return err_next
        }

        ladder[rungs] = Level{
            w = w, h = h,
            cw = (fine.w + w - 1) / w,
            ch = (fine.h + h - 1) / h,
            mask = mask, coverage = coverage, src = src,
            field = field, next = next,
        }
        rungs += 1
    }

    tol := f32(PYRAMID_TOL)
    passes_of: [MAX_LEVELS]int
    residual_of: [MAX_LEVELS]f32

    // The coarsest rung seeds itself from its own restricted source: the leading term of the
    // chain is the source, and everything past it is what the sweeps then pay for, at 1/640th
    // the cell cost of the native grid apiece.
    coarsest := &ladder[rungs - 1]
    for i in 0..<len(coarsest.src.data) {
        coarsest.field.data[i] = coarsest.mask.data[i] * coarsest.src.data[i]
    }

    for sweep_index in 0..<rungs {
        index := rungs - 1 - sweep_index
        rung := &ladder[index]

        if index < rungs - 1 {
            prolongate_into(ladder[index + 1].field, rung.mask, rung.field)
        }

        passes := 0
        residual := f32(1)
        for passes < PYRAMID_SWEEP_CAP && residual > tol {
            residual = sweep_once(s.kernel, s.keep, rung)
            rung.field, rung.next = rung.next, rung.field
            passes += 1
        }

        passes_of[index] = passes
        residual_of[index] = residual
    }

    for index in 1..<rungs {
        rung := ladder[index]
        delete(rung.mask.data)
        delete(rung.coverage.data)
        delete(rung.src.data)
        delete(rung.field.data)
        delete(rung.next.data)
    }

    when PYRAMID_REPORT {
        handle, err_open := os.open(
            "/tmp/opencode/pyramid_rungs.txt",
            os.O_WRONLY + os.O_TRUNC + os.O_CREATE,
        )
        assert(err_open == nil, "solve: cannot open the rung report")
        defer os.close(handle)

        for index in 0..<rungs {
            open := 0
            for cell in ladder[index].mask.data {
                if cell > 0 { open += 1 }
            }
            os.write_string(handle, fmt.aprintf(
                "rung {}: {} x {} cells, {} open, {:.6e} source let go, {} sweeps, residual {:.6e}\n",
                index,
                ladder[index].w,
                ladder[index].h,
                int(open),
                dropped_of[index],
                passes_of[index],
                residual_of[index],
            ))
        }
    }

    return .None
}
