package conv

import "core:testing"
import "../grid"
import "../maps"

// A coarse-to-fine solve is only worth its complexity if the result satisfies the fixed-point
// equation to rounding. On an open field every level is the same operator (mask 1, coverage
// keep), so this is also the test that catches a broken residual, restriction or prolongation
// without any mask bookkeeping to hide behind.
@(private)
open_field :: proc(w, h: int, allocator := context.temp_allocator) -> grid.Grid2D(u8) {
    g, err := grid.make(u8, w, h, allocator)
    assert(err == .None, "open_field: cannot allocate")
    for i in 0..<len(g.data) {
        g.data[i] = maps.WALKABLE
    }
    return g
}

// Worst residual of the fixed-point equation, relative to the image's own peak.
@(private)
worst_residual :: proc(s: ^HeatSolver, source: grid.Grid2D(f32)) -> f32 {
    content, err := grid.make(f32, s.field.width, s.field.height, context.temp_allocator)
    assert(err == .None, "worst_residual: cannot allocate")
    for i in 0..<len(s.field.data) {
        content.data[i] = s.field.data[i] * s.mask.data[i]
    }

    peak: f32
    worst: f32
    for y in 0..<s.field.height {
        for x in 0..<s.field.width {
            i := x + y * s.field.width
            peak = max(peak, abs(s.field.data[i]))
            if s.mask.data[i] <= 0 {
                continue
            }
            v := convolution_step(s.kernel, content, x, y, Border.CONSTANT)
            c := max(s.coverage.data[i], f32(1e-6))
            worst = max(worst, abs(source.data[i] - s.field.data[i] + s.keep * v / c))
        }
    }
    return worst / max(peak, f32(1e-30))
}

@(test)
vcycle_converges_on_an_open_field :: proc(t: ^testing.T) {
    walls := open_field(32, 32)
    s, err := heat(walls, 0.5, 0.001, 0, 0, context.temp_allocator)
    testing.expect(t, err == .None, "heat: cannot allocate the solver")

    source, err_src := grid.make(f32, 32, 32, context.temp_allocator)
    testing.expect(t, err_src == .None, "source: cannot allocate")
    source.data[16 + 16 * 32] = 1

    solved := solve(&s, source)
    testing.expect(t, solved == .None, "solve: cannot run")

    residual := worst_residual(&s, source)
    testing.expectf(t, residual < 1e-4, "solve: residual {:.3e} is not a converged fixed point", residual)

    negative := 0
    for value in s.field.data {
        if value < 0 {
            negative += 1
        }
    }
    testing.expectf(t, negative == 0, "solve: the potential went negative on {} cells", negative)
}
