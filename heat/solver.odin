package heat

import "base:runtime"
import "core:math"
import cv "../odin-opencv"

HeatSolver :: struct {
    navmap:   cv.Mat(f32),
    field:    cv.Mat(f32),
    tmp:      cv.Mat(f32),
    coverage: cv.Mat(f32),
    kernel:   cv.Mat(f32),
    keep:     f32,
    rate:     f32,
    decay:    f32,
}

solver :: proc(navmap: cv.Mat(f32), rate, decay: f32, allocator := context.allocator) -> (s: HeatSolver, err: runtime.Allocator_Error) #optional_allocator_error {
    s.rate   = rate
    s.decay  = decay
    s.keep   = 1.0 - clamp(decay, 0.0, 1.0)

    w := int(navmap.cols)
    h := int(navmap.rows)

    s.field    = cv.mat_new(f32, h, w, allocator) or_return
    s.tmp      = cv.mat_new(f32, h, w, allocator) or_return
    s.navmap   = cv.mat_new(f32, h, w, allocator) or_return
    s.coverage = cv.mat_new(f32, h, w, allocator) or_return
    s.kernel   = kernel(rate, decay, allocator)   or_return
    
    copy(s.navmap.data[:w * h], navmap.data[:w * h])

    filter := cv.Mat(f32)(s.kernel)

    cv.filter_2d(&s.navmap, &s.coverage, &filter)
    cv.convert_to(&s.coverage, &s.coverage, cv.mat_type_of(f32), f64(1 / max(s.keep, f32(1e-6))))

    cv.threshold(&s.navmap, &s.tmp, 0, 1e-6, .BINARY_INV)
    cv.add(&s.coverage, &s.tmp, &s.coverage)

    return
}

step :: proc(s: ^HeatSolver, source: ^cv.Mat(f32), substeps: int) {
    assert(source == nil || (source.rows == s.field.rows && source.cols == s.field.cols))

    if substeps <= 0 {
        return
    }

    for _ in 0..<substeps {
        diffuse(s)

        if source != nil {
            cv.add(&s.tmp, source, &s.tmp)
        }

        s.field, s.tmp = s.tmp, s.field
    }
}

splat_heat_source :: proc(navmap: cv.Mat(f32), pos: [2]int, sigma, strength: f32, allocator := context.allocator) -> (src: cv.Mat(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    src = cv.mat_new(f32, int(navmap.rows), int(navmap.cols), allocator) or_return
    if strength <= 0 do return

    splat_clipped(&src, navmap, pos, sigma)
    total := f32(cv.sum(&src))
    if total <= 0 do return

    cv.convert_to(&src, &src, cv.mat_type_of(f32), f64(strength / total))
    return
}

splat_goal_disc :: proc(walls: cv.Mat(f32), pos: [2]int, radius: f32, allocator := context.allocator) -> (src: cv.Mat(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    src = cv.mat_new(f32, int(walls.rows), int(walls.cols), allocator) or_return
    if radius <= 0 do return

    peak := splat_clipped(&src, walls, pos, radius)
    cutoff := f64(peak) * 0.6065306597126334 * (1 - 1e-3)
    cv.threshold(&src, &src, cutoff, 1, .BINARY)
    return
}

seed_source :: proc(s: ^HeatSolver, source: cv.Mat(f32)) {
    assert(source.cols == s.field.cols && source.rows == s.field.rows)

    src := source

    if s.decay > 0 {
        cv.convert_to(&src, &s.field, cv.mat_type_of(f32), f64(1 / s.decay))
    } 
    else {
        cv.convert_to(&src, &s.field, cv.mat_type_of(f32))
    }
}

@(private)
diffuse :: proc(s: ^HeatSolver) {
    cv.filter_2d(&s.field, &s.tmp, &s.kernel)
    cv.div(&s.tmp, &s.coverage, &s.tmp)
    cv.mul(&s.tmp, &s.navmap, &s.tmp)
}

@(private)
splat_clipped :: proc(img: ^cv.Mat(f32), navmap: cv.Mat(f32), pos: [2]int, sigma: f32) -> (peak: f32) {
    x := clamp(pos[0], 0, int(img.cols) - 1)
    y := clamp(pos[1], 0, int(img.rows) - 1)
    splat(img, x, y, sigma)
    peak = cv.at(img, x, y)^
    walkable := navmap
    cv.mul(img, &walkable, img)
    return
}

@(private)
splat :: proc(img: ^cv.Mat(f32), x, y: int, sigma: f32) {
    cv.at(img, x, y)^ = 1

    size := i32(math.ceil(f64(sigma) * 6))
    if size % 2 == 0 {
        size += 1
    }
    cv.gaussian_blur(img, img, {size, size}, {f64(sigma), f64(sigma)}, cv.BorderType.CONSTANT)
}
