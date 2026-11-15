package nav

import "base:runtime"
import "../heat"
import "../maps"
import cv "../odin-opencv"

FlowField :: struct {
    navmap:    cv.Mat(f32),
    field:     cv.Mat([2]f32),
    padded:    cv.Mat(f32),
    avg:       cv.Mat(f32),
    gx:        cv.Mat(f32),
    gy:        cv.Mat(f32),
    wall_cov:  cv.Mat(f32),
    wall_mask: cv.Mat(u8),
    mag:       cv.Mat(f32),
    zero:      cv.Mat(f32),
    unit:      cv.Mat([2]f32),
    map_x:     cv.Mat(f32),
    map_y:     cv.Mat(f32),
    cross:     cv.Mat(f32),
}

make :: proc(walls: cv.Mat(f32), allocator := context.allocator) -> (f: FlowField, err: runtime.Allocator_Error) #optional_allocator_error {
    f.navmap = walls

    w := int(walls.cols)
    h := int(walls.rows)

    f.field  = cv.mat_new([2]f32, h, w, allocator) or_return

    // OpenCV's Sobel does not compute its own outermost
    // pixel row and column, so a unit standing on the
    // edge of the map would have no direction to follow. 
    f.padded    = cv.mat_new(f32, h + 2, w + 2, allocator)      or_return
    f.avg       = cv.mat_new(f32, h + 2, w + 2, allocator)      or_return
    f.gx        = cv.mat_new(f32, h + 2, w + 2, allocator)      or_return
    f.gy        = cv.mat_new(f32, h + 2, w + 2, allocator)      or_return
    f.wall_cov  = cv.mat_new(f32, h + 2, w + 2, allocator)      or_return
    f.wall_mask = cv.mat_new(u8,  h + 2, w + 2, allocator)      or_return
    f.mag       = cv.mat_new(f32, h + 2, w + 2, allocator)      or_return
    f.zero      = cv.mat_new(f32, h + 2, w + 2, allocator)      or_return
    f.unit      = cv.mat_new([2]f32, h + 2, w + 2, allocator)   or_return
    f.map_x     = cv.mat_new(f32, h, w, allocator)              or_return
    f.map_y     = cv.mat_new(f32, h, w, allocator)              or_return

    values := [9]f32 {
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
    cv.copy_make_border(&s.field, &f.padded, 1, 1, 1, 1, cv.BorderType.REPLICATE)
    cv.filter_2d(&f.padded, &f.avg, &f.cross)
    cv.div(&f.avg, &f.wall_cov, &f.avg)
    cv.copy_to(&f.avg, &f.padded, &f.wall_mask)
    cv.gradient_x(&f.padded, &f.gx)
    cv.gradient_y(&f.padded, &f.gy)
    cv.set_to(&f.gx, 0, &f.wall_mask)
    cv.set_to(&f.gy, 0, &f.wall_mask)
    cv.magnitude(&f.gx, &f.gy, &f.mag)
    cv.threshold(&f.mag, &f.zero, 0, 1, .BINARY_INV)
    cv.max(&f.mag, &f.zero, &f.mag)
    cv.div(&f.gx, &f.mag, &f.gx)
    cv.div(&f.gy, &f.mag, &f.gy)
    cv.merge(&f.gx, &f.gy, &f.unit)
    cv.resample(&f.unit, &f.field, &f.map_x, &f.map_y, .NEAREST)
}

@(private)
prepare :: proc(f: ^FlowField) {
    w := int(f.navmap.cols)
    h := int(f.navmap.rows)
    pw := w + 2

    for y in 0..<h + 2 {
        for x in 0..<w + 2 {
            i := x + y * pw
            if x >= 1 && x <= w && y >= 1 && y <= h {
                f.wall_mask.data[i] = 1 if f.navmap.data[x - 1 + (y - 1) * w] != maps.WALKABLE else 0
            } else {
                f.wall_mask.data[i] = 0
            }
        }
    }

    for y in 0..<h + 2 {
        for x in 0..<w + 2 {
            i := x + y * pw
            weight := f32(0)
            if x > 0 && f.wall_mask.data[i - 1] == 0 {
                weight += 0.25
            }
            if x + 1 < w + 2 && f.wall_mask.data[i + 1] == 0 {
                weight += 0.25
            }
            if y > 0 && f.wall_mask.data[i - pw] == 0 {
                weight += 0.25
            }
            if y + 1 < h + 2 && f.wall_mask.data[i + pw] == 0 {
                weight += 0.25
            }
            f.wall_cov.data[i] = max(weight, f32(1e-6))
        }
    }

    for y in 0..<h {
        for x in 0..<w {
            i := x + y * w
            f.map_x.data[i] = f32(x + 1)
            f.map_y.data[i] = f32(y + 1)
        }
    }
}

