package grid

import "base:runtime"
import "base:builtin"

Grid2D :: struct($T: typeid) {
    data: []T,
    width, height: int
}

make :: proc($T: typeid, w, h: int, allocator := context.allocator) -> (grid: Grid2D(T), err: runtime.Allocator_Error) {
    grid.data   = builtin.make([]T, w*h, allocator) or_return
    grid.width  = w
    grid.height = h
    return
}

rowmaj :: proc(g: $G/Grid2D($T), x, y: int) -> ^T {
    return &g.data[x + y*g.width]
}

colmaj :: proc(g: $G/Grid2D($T), x, y: int) -> ^T {
    return &g.data[y + x*g.height]
}

ROWMAJ :: true

at :: proc(g: $G/Grid2D($T), x, y: int) -> ^T {
    when ROWMAJ do return rowmaj(g, x, y)
    else        do return colmaj(g, x, y)
}
