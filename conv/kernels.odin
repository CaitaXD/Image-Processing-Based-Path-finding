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

// The heat operator as an explicit kernel array. The arm weight is fixed at rate / 4
// rather than rate / deg(x), so the whole per-cell operator is a single 3x3 stencil:
// walls and out-of-frame cells are simply read as 0 instead of being skipped, which
// makes the boundary of the walkable region an open boundary that loses mass.
//
// OpenCV equivalent, once odin-opencv exports it (no FFI edits in this checkout):
//   cv::filter2D(cv::Mat1f(u), cv::Mat1f(v), cv::Mat3f(K), cv::Border::CONSTANT)
// grid.Grid2D(f32).data is row-major and contiguous, so u is a zero-copy Mat1f view.
// The DFT symbol of this kernel is keep * ((1 - rate) + rate * (cos_x + cos_y) / 2)
// where cos_x = cos(2 pi p / M) and cos_y = cos(2 pi q / N). It is a formula in the
// frequency index, not an array: a t-step chain is one pointwise power of it and the
// infinite chain is one divide, so no array of any radius is ever needed.
diffusion :: proc(rate, decay: f32, allocator := context.allocator) -> (k: Kernel, err: runtime.Allocator_Error) #optional_allocator_error {
    g := grid.make(f32, 3, 3, allocator) or_return
    k = Kernel(g)

    keep := 1.0 - clamp(decay, 0.0, 1.0)
    mix := clamp(rate, 0.0, 1.0)
    arm := keep * mix / 4.0
    center := keep * (1.0 - mix)

    values := [9]f32{
        0.0, arm, 0.0,
        arm, center, arm,
        0.0, arm, 0.0,
    }
    copy(k.data, values[:])

    return
}
