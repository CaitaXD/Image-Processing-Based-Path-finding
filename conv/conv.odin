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

@(private)
convolution_step :: proc(kernel: Kernel, g: grid.Grid2D($T), x, y: int) -> T {
    kw := kernel.width
    kh := kernel.height
    kcx := kw / 2
    kcy := kh / 2

    when COMPLEX(T) || QUATERNION(T) {
        acc: T
        for ky in 0..<kh {
            for kx in 0..<kw {
                sx := clamp(x + kx - kcx, 0, g.width - 1)
                sy := clamp(y + ky - kcy, 0, g.height - 1)
                acc += T(grid.at(kernel, kx, ky)^) * grid.at(g, sx, sy)^
            }
        }
        return acc
    } 
    else {
        acc: f64
        for ky in 0..<kh {
            for kx in 0..<kw {
                sx := clamp(x + kx - kcx, 0, g.width - 1)
                sy := clamp(y + ky - kcy, 0, g.height - 1)
                acc += f64(grid.at(kernel, kx, ky)^) * f64(grid.at(g, sx, sy)^)
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

convolution :: proc(kernel: Kernel, g: grid.Grid2D($T), repeat := 1) -> (out: grid.Grid2D(T), err: runtime.Allocator_Error) where NUMERIC(T) {
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
                grid.at(dst^, x, y)^ = convolution_step(kernel, src^, x, y)
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
