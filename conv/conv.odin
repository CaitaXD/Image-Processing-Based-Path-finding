package conv

import "base:runtime"
import "base:intrinsics"
import "core:math"
import "../grid"

NUMERIC    :: intrinsics.type_is_numeric
INTEGER    :: intrinsics.type_is_integer
FLOAT      :: intrinsics.type_is_float
QUATERNION :: intrinsics.type_is_quaternion
COMPLEX    :: intrinsics.type_is_complex

// Source-stencil policy, named after OpenCV's ktype so one Kernel array means the
// same operator on both code paths. CONSTANT is zero extension (out-of-frame cells
// contribute 0), REPLICATE clamps, WRAP is toroidal. WRAP is what an FFT-based
// convolution computes implicitly, since the DFT diagonalizes cyclic convolution.
Border :: enum {
    CONSTANT  = 0,
    REPLICATE = 1,
    REFLECT   = 2,
    WRAP      = 3,
}

// The stencil site is x + k - kc, so the kernel is applied as a correlation. That
// is identical to convolution for a symmetric kernel, which every heat kernel here
// is.
@(private)
stencil_at :: proc(g: grid.Grid2D($T), x, y: int, border: Border) -> T {
    if x >= 0 && x < g.width && y >= 0 && y < g.height {
        return grid.at(g, x, y)^
    }
    if border == Border.CONSTANT {
        return T(0)
    }
    assert(border == Border.REPLICATE, "stencil_at: REFLECT and WRAP extension are not implemented")
    return grid.at(g, clamp(x, 0, g.width - 1), clamp(y, 0, g.height - 1))^
}

// One stencil application at one site. Exported so the heat solver in this package's
// other module applies the same operator the convolution test asserts.
convolution_step :: proc(kernel: Kernel, g: grid.Grid2D($T), x, y: int, border: Border) -> T {
    kw := kernel.width
    kh := kernel.height
    kcx := kw / 2
    kcy := kh / 2

    when COMPLEX(T) || QUATERNION(T) {
        acc: T
        for ky in 0..<kh {
            for kx in 0..<kw {
                acc += T(grid.at(kernel, kx, ky)^) * stencil_at(g, x + kx - kcx, y + ky - kcy, border)
            }
        }
        return acc
    } 
    else {
        acc: f64
        for ky in 0..<kh {
            for kx in 0..<kw {
                acc += f64(grid.at(kernel, kx, ky)^) * f64(stencil_at(g, x + kx - kcx, y + ky - kcy, border))
            }
        }

        when INTEGER(T) {
            return T(clamp(math.round(acc), f64(min(T)), f64(max(T))))
        } 
        else when FLOAT(T) {
            return T(acc)
        }
        else {
            #panic("Not implemented")
        }
    }
}

convolution :: proc(kernel: Kernel, g: grid.Grid2D($T), repeat := 1, border := Border.REPLICATE) -> (out: grid.Grid2D(T), err: runtime.Allocator_Error) where NUMERIC(T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    
    out_grid := grid.make(T, g.width, g.height) or_return

    out = out_grid
    copy(out.data, g.data)

    if repeat <= 0 do return

    tmp := grid.make(T, g.width, g.height, context.temp_allocator) or_return
    
    src := &out
    dst := &tmp

    for _ in 0..<repeat {
        for y in 0..<g.height {
            for x in 0..<g.width {
                grid.at(dst^, x, y)^ = convolution_step(kernel, src^, x, y, border)
            }
        }
        src, dst = dst, src
    }

    if repeat % 2 != 0 {
        copy(out.data, tmp.data)
    }

    return
}

sum :: proc(g: grid.Grid2D($T), x, y: int, radious := 1) -> T where NUMERIC(T) {
    sum :T= 0
    for i in -radious..=radious {
        for j in -radious..=radious {
            if i == 0 && j == 0 do continue
            nx := x + i
            ny := y + j
            if nx >= 0 && nx < g.width && ny >= 0 && ny < g.height {
                sum += grid.at(g, nx, ny)^
            }
        }
    }
    return sum
}
