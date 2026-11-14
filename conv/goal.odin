package conv

import "base:runtime"
import "core:math"
import gd "../grid"
import "../maps"

Goal :: struct {
    pos:    [2]f32,
    weight: f32,
}

HeatSolver :: struct {
    grid:            gd.Grid2D(u8),
    field:           gd.Grid2D(f32),
    tmp:             gd.Grid2D(f32),
    kernel:          Kernel,
    mask:            gd.Grid2D(f32),
    coverage:        gd.Grid2D(f32),
    keep:            f32,
    rate:            f32,
    decay:           f32,
    source_radius:   f32,
    source_strength: f32,
}

heat :: proc(walls: gd.Grid2D(u8), rate, decay, source_radius, source_strength: f32, allocator := context.allocator) -> (s: HeatSolver, err: runtime.Allocator_Error) #optional_allocator_error {
    s.grid           = walls
    s.rate            = rate
    s.decay           = decay
    s.source_radius   = source_radius
    s.source_strength = source_strength

    s.field, err = gd.make(f32, walls.width, walls.height, allocator)
    if err != nil {
        return
    }

    s.tmp, err = gd.make(f32, walls.width, walls.height, allocator)
    if err != nil {
        delete(s.field.data)
        s.field = {}
        return
    }

    // The whole per-cell operator lives in this 3x3 array (conv.diffusion): arms are
    // fixed at rate / 4 rather than rate / deg(x), and decay and the identity mix are
    // baked in too, so one kernel array is all that both the OpenCV call site and the
    // DFT symbol need. conv.diffusion clamps rate and decay into [0, 1].
    s.kernel, err = diffusion(rate, decay, allocator)
    if err != nil {
        delete(s.field.data)
        delete(s.tmp.data)
        s.field = {}
        s.tmp = {}
        return
    }

    // The walkable region as an f32 image: 1 where walkable, 0 at walls. The step is
    // u' = m * (K correlated with u), so this mask multiply is the only obstruction in
    // the operator; drop it and heat leaks through walls.
    s.mask, err = gd.make(f32, walls.width, walls.height, allocator)
    if err != nil {
        delete(s.field.data)
        delete(s.tmp.data)
        delete(s.kernel.data)
        s.field = {}
        s.tmp = {}
        s.kernel = {}
        s.mask = {}
        return
    }

    // Same dims and the same accessor on both grids, so a flat correspondence is enough.
    for i in 0..<len(s.mask.data) {
        s.mask.data[i] = f32(0)
        if walls.data[i] == maps.WALKABLE {
            s.mask.data[i] = f32(1)
        }
    }

    // Normalized convolution: the same kernel is also correlated with the mask once,
    // giving the weight the stencil actually has at each cell. Dividing by it is what
    // makes a wall an inert tap rather than a cold one (see diffuse).
    s.coverage, err = gd.make(f32, walls.width, walls.height, allocator)
    if err != nil {
        delete(s.field.data)
        delete(s.tmp.data)
        delete(s.kernel.data)
        delete(s.mask.data)
        s = {}
        return
    }

    s.keep = 1.0 - clamp(decay, 0.0, 1.0)
    for y in 0..<walls.height {
        for x in 0..<walls.width {
            gd.at(s.coverage, x, y)^ = convolution_step(s.kernel, s.mask, x, y, Border.CONSTANT)
        }
    }

    return
}

step :: proc(s: ^HeatSolver, goals: []Goal, substeps: int) {
    if substeps <= 0 {
        return
    }

    for _ in 0..<substeps {
        diffuse(s)

        for goal in goals {
            add_source(&s.tmp, s.grid, goal, s.source_radius, s.source_strength)
        }

        s.field, s.tmp = s.tmp, s.field
    }
}

make_source :: proc(walls: gd.Grid2D(u8), goal: Goal, sigma, strength: f32, allocator := context.allocator) -> (src: gd.Grid2D(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    src = gd.make(f32, walls.width, walls.height, allocator) or_return

    weight := max(goal.weight, f32(0))
    if weight == 0 || strength <= 0 {
        return
    }

    if sigma <= 0 {
        x := int(goal.pos[0])
        y := int(goal.pos[1])
        if x >= 0 && x < src.width && y >= 0 && y < src.height &&
           gd.at(walls, x, y)^ == maps.WALKABLE {
            gd.at(src, x, y)^ = weight * strength
        }
        return
    }

    radius := sigma * 3
    x0 := max(int(goal.pos[0] - radius), 0)
    x1 := min(int(goal.pos[0] + radius) + 1, src.width - 1)
    y0 := max(int(goal.pos[1] - radius), 0)
    y1 := min(int(goal.pos[1] + radius) + 1, src.height - 1)
    if x0 > x1 || y0 > y1 {
        return
    }

    two_sigma_sq := 2 * sigma * sigma

    kernel_sum: f32
    for y in y0..=y1 {
        for x in x0..=x1 {
            if gd.at(walls, x, y)^ != maps.WALKABLE {
                continue
            }
            dx := (f32(x) + 0.5) - goal.pos[0]
            dy := (f32(y) + 0.5) - goal.pos[1]
            kernel_sum += math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
        }
    }
    if kernel_sum <= 0 {
        return
    }

    scale := weight * strength / kernel_sum
    for y in y0..=y1 {
        for x in x0..=x1 {
            if gd.at(walls, x, y)^ != maps.WALKABLE {
                continue
            }
            dx := (f32(x) + 0.5) - goal.pos[0]
            dy := (f32(y) + 0.5) - goal.pos[1]
            gd.at(src, x, y)^ = scale * math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
        }
    }

    return
}

// The goal as an image rather than a coordinate: 1 on the walkable cells whose centre
// lies within radius of the goal point, 0 elsewhere. The heat splat and the goal splat
// the units sink into read the same array, so nothing downstream carries a goal position
// and absorbing is just a second mask multiply. radius is measured to cell centres: 0.4 the goal cell alone, 1.0
// the four-neighbour cross, 1.5 the whole 3x3 block.
goal_image :: proc(walls: gd.Grid2D(u8), goal: Goal, radius: f32, allocator := context.allocator) -> (src: gd.Grid2D(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    src = gd.make(f32, walls.width, walls.height, allocator) or_return

    if goal.weight <= 0 || radius <= 0 {
        return
    }

    x0 := max(int(goal.pos[0] - radius), 0)
    x1 := min(int(goal.pos[0] + radius) + 1, src.width - 1)
    y0 := max(int(goal.pos[1] - radius), 0)
    y1 := min(int(goal.pos[1] + radius) + 1, src.height - 1)
    if x0 > x1 || y0 > y1 {
        return
    }

    for y in y0..=y1 {
        for x in x0..=x1 {
            if gd.at(walls, x, y)^ != maps.WALKABLE {
                continue
            }
            dx := (f32(x) + 0.5) - goal.pos[0]
            dy := (f32(y) + 0.5) - goal.pos[1]
            if dx * dx + dy * dy > radius * radius {
                continue
            }
            gd.at(src, x, y)^ = 1.0
        }
    }

    return
}

seed_source :: proc(s: ^HeatSolver, source: gd.Grid2D(f32)) {
    assert(source.width == s.field.width && source.height == s.field.height)

    if s.decay > 0 {
        inv_decay := 1 / s.decay
        for i in 0..<len(s.field.data) {
            s.field.data[i] = source.data[i] * inv_decay
        }
    } else {
        copy(s.field.data, source.data)
    }
}

// The same seeding, with the splat conv does for itself rather than one the caller hands
// over: the goal written through the solver's own radius (source_radius / source_strength),
// into the solver's own scratch buffer. The scratch is cleared first, because the splat only
// writes walkable cells inside its box, and it has to be clean before it is read as the seed.
// Nothing is allocated here: the next diffuse writes every cell of the scratch, so it is
// still a valid swap partner afterwards.
seed_goal :: proc(s: ^HeatSolver, goal: Goal) {
    for i in 0..<len(s.tmp.data) {
        s.tmp.data[i] = f32(0)
    }
    add_source(&s.tmp, s.grid, goal, s.source_radius, s.source_strength)
    seed_source(s, s.tmp)
}

step_with_source :: proc(s: ^HeatSolver, source: gd.Grid2D(f32), substeps: int) {
    if substeps <= 0 {
        return
    }
    assert(source.width == s.field.width && source.height == s.field.height)

    for _ in 0..<substeps {
        diffuse(s)

        for i in 0..<len(s.tmp.data) {
            s.tmp.data[i] += source.data[i]
        }

        s.field, s.tmp = s.tmp, s.field
    }
}

// The step as an image-processing operator: normalized convolution,
//
//     u' = keep * (K correlated with u) / max(K correlated with m, eps)
//
// with K the 3x3 array from conv.diffusion and m the walkable mask. Dividing by the
// stencil's own weight over the mask is what makes a wall an *inert* tap: the cell
// averages only its walkable neighbours and a constant field stays constant, which is a
// no-flux (insulating) boundary. Multiplying by m instead - the older u' = m * K(u) -
// makes a wall a *cold* tap that votes zero; in a one-cell corridor that operator has
// u = 0.25 (u_left + u_right), so the potential decays by 0.268 per corridor cell and
// underflows f32 after a few hundred cells (measured: the spawn band sits 414-641 cells
// from the goal and the field there read 1e-45/1e-10, i.e. denormals or interpolation
// noise, and its gradient carried no direction). Coverage is static, so it is convolved
// once in heat() instead of every step.
//
// In free space K(m) = keep, so the step reduces to the plain masked operator and the
// symbol of the iteration is unchanged: keep * ((1 - rate) + rate * (cos_x + cos_y) / 2).
// Out-of-frame stencil cells read 0 (Border.CONSTANT), the same boundary cv::filter2D's
// Border::CONSTANT gives and the same one the DFT diagonalizes once the image is padded.
//
// Same operator in OpenCV, once odin-opencv exports filter2D (no FFI edits here):
//   cv::filter2D(cv::Mat1f(u), cv::Mat1f(v), cv::Mat3f(K), cv::Border::CONSTANT);
//   cv::filter2D(cv::Mat1f(m), cv::Mat1f(c), cv::Mat3f(K), cv::Border::CONSTANT);
//   cv::multiply(v, cv::Mat1f(keep));  cv::divide(v, cv::Mat1f(c), v);
// Grid2D(f32).data is row-major and contiguous, so cv::Mat1f(u) is a zero-copy view.
@(private)
diffuse :: proc(s: ^HeatSolver) {
    assert(s.mask.width == s.field.width && s.mask.height == s.field.height)
    assert(s.coverage.width == s.field.width && s.coverage.height == s.field.height)

    for y in 0..<s.field.height {
        for x in 0..<s.field.width {
            i := x + y * s.field.width
            if s.mask.data[i] <= 0 {
                s.tmp.data[i] = 0
                continue
            }
            v := convolution_step(s.kernel, s.field, x, y, Border.CONSTANT)
            c := max(s.coverage.data[i], f32(1e-6))
            s.tmp.data[i] = s.keep * v / c
        }
    }
}

@(private)
add_source :: proc(dst: ^gd.Grid2D(f32), walls: gd.Grid2D(u8), goal: Goal, sigma, strength: f32) {
    weight := max(goal.weight, f32(0))
    if weight == 0 || strength <= 0 {
        return
    }

    if sigma <= 0 {
        x := int(goal.pos[0])
        y := int(goal.pos[1])
        if x >= 0 && x < dst.width && y >= 0 && y < dst.height &&
           gd.at(walls, x, y)^ == maps.WALKABLE {
            gd.at(dst^, x, y)^ += weight * strength
        }
        return
    }

    radius := sigma * 3
    x0 := max(int(goal.pos[0] - radius), 0)
    x1 := min(int(goal.pos[0] + radius) + 1, dst.width - 1)
    y0 := max(int(goal.pos[1] - radius), 0)
    y1 := min(int(goal.pos[1] + radius) + 1, dst.height - 1)
    if x0 > x1 || y0 > y1 {
        return
    }

    two_sigma_sq := 2 * sigma * sigma

    kernel_sum: f32
    for y in y0..=y1 {
        for x in x0..=x1 {
            if gd.at(walls, x, y)^ != maps.WALKABLE {
                continue
            }
            dx := (f32(x) + 0.5) - goal.pos[0]
            dy := (f32(y) + 0.5) - goal.pos[1]
            kernel_sum += math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
        }
    }
    if kernel_sum <= 0 {
        return
    }

    scale := weight * strength / kernel_sum
    for y in y0..=y1 {
        for x in x0..=x1 {
            if gd.at(walls, x, y)^ != maps.WALKABLE {
                continue
            }
            dx := (f32(x) + 0.5) - goal.pos[0]
            dy := (f32(y) + 0.5) - goal.pos[1]
            gd.at(dst^, x, y)^ += scale * math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
        }
    }
}
