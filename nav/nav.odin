package nav

import "base:runtime"
import "../heat"
import "../maps"
import cv "../odin-opencv"

FlowField :: struct {
    navmap:         cv.Mat(f32),
    field:          cv.Mat([2]f32),
    gx:             cv.Mat(f32),
    gy:             cv.Mat(f32),
    mag:            cv.Mat(f32),
    avg:            cv.Mat(f32),
    cross:          cv.Mat(f32),
    wall_coverage:  cv.Mat(f32),
    wall_mask:      cv.Mat(u8),
}

make :: proc(walls: cv.Mat(f32), allocator := context.allocator) -> (f: FlowField, err: runtime.Allocator_Error) #optional_allocator_error {
    f.navmap = walls

    w := int(walls.cols)
    h := int(walls.rows)

    f.field     = cv.mat_new([2]f32, h, w, allocator)   or_return
    f.gx        = cv.mat_new(f32, h, w, allocator)      or_return
    f.gy        = cv.mat_new(f32, h, w, allocator)      or_return
    f.mag       = cv.mat_new(f32, h, w, allocator)      or_return
    f.avg       = cv.mat_new(f32, h, w, allocator)      or_return
    f.wall_coverage = cv.mat_new(f32, h, w, allocator)  or_return
    f.wall_mask     = cv.mat_new(u8,  h, w, allocator)  or_return

    values := [9]f32{
        0.00, 0.25, 0.00,
        0.25, 0.00, 0.25,
        0.00, 0.25, 0.00,
    }

    f.cross = cv.mat_new(f32, 3, 3, allocator) or_return
    copy(f.cross.data[:9], values[:])

    prepare(&f)
    return
}

update :: proc(f: ^FlowField, s: ^heat.HeatSolver) {
    assert(s.field.cols == f.navmap.cols && s.field.rows == f.navmap.rows)
    cv.filter_2d(&s.field, &f.avg, &f.cross, -1, .REPLICATE)
    cv.div(&f.avg, &f.wall_coverage, &f.avg)
    cv.copy_to(&f.avg, &s.field, &f.wall_mask)
    cv.gradient_x(&s.field, &f.gx, .REPLICATE)
    cv.gradient_y(&s.field, &f.gy, .REPLICATE)
    cv.set_to(&s.field, 0, &f.wall_mask)
    cv.set_to(&f.gx, 0, &f.wall_mask)
    cv.set_to(&f.gy, 0, &f.wall_mask)
    cv.magnitude(&f.gx, &f.gy, &f.mag)
    cv.max_scalar(&f.mag, 1e-30, &f.mag)
    cv.div(&f.gx, &f.mag, &f.gx)
    cv.div(&f.gy, &f.mag, &f.gy)
    cv.merge(&f.gx, &f.gy, &f.field)
}

@(private)
prepare :: proc(f: ^FlowField) {
    w := int(f.navmap.cols)
    h := int(f.navmap.rows)

    for y in 0..<h {
        for x in 0..<w {
            i := x + y * w
            f.wall_mask.data[i] = 1 if f.navmap.data[i] != maps.WALKABLE else 0
        }
    }

    for y in 0..<h {
        for x in 0..<w {
            i := x + y * w
            weight := f32(0)
            if x == 0 || f.wall_mask.data[i - 1] == 0 {
                weight += 0.25
            }
            if x + 1 == w || f.wall_mask.data[i + 1] == 0 {
                weight += 0.25
            }
            if y == 0 || f.wall_mask.data[i - w] == 0 {
                weight += 0.25
            }
            if y + 1 == h || f.wall_mask.data[i + w] == 0 {
                weight += 0.25
            }
            f.wall_coverage.data[i] = max(weight, f32(1e-6))
        }
    }
}
