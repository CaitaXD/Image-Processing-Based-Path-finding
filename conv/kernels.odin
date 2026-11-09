package conv

import "base:runtime"
import "core:math"
import "../grid"

Kernel :: distinct grid.Grid2D(f32)

Sobel_Direction :: enum {
    X,
    Y,
}

gauss :: proc(size: int = 3, sigma: f32 = 0) -> (k: Kernel, err: runtime.Allocator_Error) {
    assert(size > 0 && size % 2 == 1, "gauss: size must be a positive odd number")

    sig := sigma
    if sig <= 0 {
        sig = f32(size) / 6.0
    }

    g := grid.make(f32, size, size) or_return
    k = Kernel(g)

    center := f32(size / 2)
    two_sigma_sq := 2.0 * sig * sig
    sum: f32

    for y in 0..<size {
        for x in 0..<size {
            dx := f32(x) - center
            dy := f32(y) - center
            value := math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
            grid.rowmaj(k, x, y)^ = value
            sum += value
        }
    }

    if sum != 0 {
        for i in 0..<len(k.data) {
            k.data[i] /= sum
        }
    }

    return
}

sobel :: proc(direction: Sobel_Direction = .X) -> (k: Kernel, err: runtime.Allocator_Error) {
    g := grid.make(f32, 3, 3) or_return
    k = Kernel(g)

    x_kernel := [9]f32{
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1,
    }
    y_kernel := [9]f32{
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1,
    }

    values := x_kernel if direction == .X else y_kernel
    copy(k.data, values[:])

    return
}

laplace :: proc() -> (k: Kernel, err: runtime.Allocator_Error) {
    g := grid.make(f32, 3, 3) or_return
    k = Kernel(g)

    values := [9]f32{
        0,  1, 0,
        1, -4, 1,
        0,  1, 0,
    }
    copy(k.data, values[:])

    return
}
