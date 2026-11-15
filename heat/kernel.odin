package heat

import "base:runtime"
import cv "../odin-opencv"

kernel :: proc(rate, decay: f32, allocator := context.allocator) -> (k: cv.Mat(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    k = cv.mat_new(f32, 3, 3, allocator) or_return

    keep := 1.0 - clamp(decay, 0.0, 1.0)
    mix := clamp(rate, 0.0, 1.0)
    arm := keep * mix / 4.0
    center := keep * (1.0 - mix)

    values := [9]f32{
        0.0, arm, 0.0,
        arm, center, arm,
        0.0, arm, 0.0,
    }
    m := cv.Mat(f32)(k)
    copy(m.data[:9], values[:])

    return
}
