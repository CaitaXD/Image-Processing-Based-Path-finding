#+feature using-stmt
package raylayout
import rl "vendor:raylib"
import la "core:math/linalg"

@(private)
scalar_absolute :: proc(v: f32) -> Scalar {
    return {.Absolute, v}
}

@(private)
scalar_relative :: proc(v: f32) -> Scalar {
    return {.Relative, v}
}

@(private)
scalar_viewport :: proc(v: f32) -> Scalar {
    return {.Viewport, v}
}

@(private)
vector2_absolute :: proc(x,y:f32) -> [2]Scalar {
    return {scalar_absolute(x), scalar_absolute(y)}
}

@(private)
vector2_relative :: proc(x,y:f32) -> [2]Scalar {
    return {scalar_relative(x), scalar_relative(y)}
}

@(private)
vector2_viewport :: proc(x,y:f32) -> [2]Scalar {
    return {scalar_viewport(x), scalar_viewport(y)}
}

@(private)
scalar_remainder :: proc(v:f32=1.0) -> Scalar {
    return {.Remainder, v}
}

@(private)
vector2_remainder :: proc(x,y:f32) -> [2]Scalar {
    return {scalar_remainder(x), scalar_remainder(y)}
}

@(private)
scalar_reference :: proc(v: f32) -> Scalar {
    return {.Reference, v}
}

@(private)
vector2_reference :: proc(x,y:f32) -> [2]Scalar {
    return {scalar_reference(x), scalar_reference(y)}
}

asp_f32 :: proc(v: f32) -> Scalar {
    return {.Aspect, v}
}

asp_xy :: proc(x,y:f32) -> Scalar {
    return {.Aspect, x/y}
}

abs  :: proc{scalar_absolute, vector2_absolute}
rel  :: proc{scalar_relative, vector2_relative}
view :: proc{scalar_viewport, vector2_viewport}
rem  :: proc{scalar_remainder, vector2_remainder}
ref  :: proc{scalar_reference, vector2_reference}
asp  :: proc{asp_f32, asp_xy}

push_vertical :: proc(size: Scalar, spacing: f32 = 4, padding: f32 = 8) -> bool {
    b := bounds_from_size(size)
    if top == 0 {
        b = padded(b, padding)
    }
    return push_layout(Layout {
        bounds     = b,
        spacing    = spacing,
        vert       = true,
    })
}

@(deferred_none=pop)
vertical_guard :: proc(size: Scalar, spacing: f32 = 4, padding: f32 = 8) -> bool {
    return push_vertical(size, spacing, padding)
}

push_horizontal :: proc(size: Scalar, spacing: f32 = 4, padding: f32 = 8) -> bool {
    b := bounds_from_size(size)
    if top == 0 {
        b = padded(b, padding)
    }
    return push_layout(Layout {
        bounds     = b,
        spacing    = spacing,
        vert       = false,
    })
}

@(deferred_none=pop)
horizontal_guard :: proc(size: Scalar, spacing: f32 = 4, padding: f32 = 8) -> bool {
    return push_horizontal(size, spacing, padding)
}

peek_rect :: proc{peek_rect_t, peek_rect_args}

peek_rect_t :: proc(t: Transform) -> rl.Rectangle {
    l, ok := top_layout()
    if !ok {
        return {}
    }
    sw := cast(f32)rl.GetScreenWidth()
    sh := cast(f32)rl.GetScreenHeight()
    slot: rl.Rectangle
    if l.vert {
        sz := flow_size(t.size[0], l.bounds.width, l.cursor, sw, ref_width)
        if t.size[0].type == .Aspect {
            sz = l.bounds.height * t.size[0].value
        }
        slot = {l.bounds.x + l.cursor, l.bounds.y, sz, l.bounds.height}
    } else {
        sz := flow_size(t.size[1], l.bounds.height, l.cursor, sh, ref_height)
        if t.size[1].type == .Aspect {
            sz = l.bounds.width / t.size[1].value
        }
        slot = {l.bounds.x, l.bounds.y + l.cursor, l.bounds.width, sz}
    }
    off := point_value(t.position, slot.width, slot.height, sw, sh)
    scv := size_value(t.scale, 1, 1, 1, 1)
    return {slot.x + off[0], slot.y + off[1], slot.width * scv[0], slot.height * scv[1]}
}

peek_rect_args :: proc(
    size: Vector,
    point: Point = {{.Relative, 0}, {.Relative, 0}},
    scale: Vector = {{.Relative, 1}, {.Relative, 1}},
) -> rl.Rectangle {
    return peek_rect_t(transform(size, point, scale))
}

get_rect :: proc{get_rect_t, get_rect_args}

get_rect_t :: proc(t: Transform) -> rl.Rectangle {
    rect := peek_rect_t(t)
    if top >= 1 {
        l := &stack[top - 1]
        sz: f32
        if l.vert {
            sz = flow_size(t.size[0], l.bounds.width, l.cursor, cast(f32)rl.GetScreenWidth(), ref_width)
            if t.size[0].type == .Aspect {
                sz = l.bounds.height * t.size[0].value
            }
        } else {
            sz = flow_size(t.size[1], l.bounds.height, l.cursor, cast(f32)rl.GetScreenHeight(), ref_height)
            if t.size[1].type == .Aspect {
                sz = l.bounds.width / t.size[1].value
            }
        }
        l.cursor += sz + l.spacing
    }
    return rect
}

get_rect_args :: proc(
    size: Vector,
    point: Point = {{.Relative, 0}, {.Relative, 0}},
    scale: Vector = {{.Relative, 1}, {.Relative, 1}},
) -> rl.Rectangle {
    return get_rect_t(transform(size, point, scale))
}

fill_layout :: proc() -> rl.Rectangle {
    return get_rect(transform(remaining_vec()))
}

resolve :: proc{resolve_t, resolve_args}

resolve_t :: proc(t: Transform) -> rl.Rectangle {
    l, ok := top_layout()
    if !ok {
        return {}
    }
    sw := cast(f32)rl.GetScreenWidth()
    sh := cast(f32)rl.GetScreenHeight()
    off := point_value(t.position, l.bounds.width, l.bounds.height, sw, sh)
    sz  := size_value(t.size, l.bounds.width, l.bounds.height, sw, sh)
    sc  := size_value(t.scale, 1, 1, 1, 1)
    return {
        l.bounds.x + off[0],
        l.bounds.y + off[1],
        sz[0] * sc[0],
        sz[1] * sc[1],
    }
}

resolve_args :: proc(
    size: Vector,
    point: Point = {{.Relative, 0}, {.Relative, 0}},
    scale: Vector = {{.Relative, 1}, {.Relative, 1}},
) -> rl.Rectangle {
    return resolve_t(transform(size, point, scale))
}

@(private)
point_value :: proc(pos: Point, boundary_x: f32, boundary_y: f32, viewport_x: f32, viewport_y: f32) -> [2]f32 {
    return {
        scalar_value(pos.x, boundary_x, viewport_x, ref_width),
        scalar_value(pos.y, boundary_y, viewport_y, ref_height),
    }
}

@(private)
size_value :: proc(size: Vector, boundary_w: f32, boundary_h: f32, viewport_w: f32, viewport_h: f32) -> [2]f32 {
    xr := scalar_value(size[0], boundary_w, viewport_w, ref_width)
    yr := scalar_value(size[1], boundary_h, viewport_h, ref_height)
    if size[0].type == .Aspect {
        xr = yr * size[0].value
    } else if size[1].type == .Aspect {
        yr = xr / size[1].value
    }
    return {xr, yr}
}

@(private)
scalar_value :: proc(using scalar: Scalar, boundary: f32, viewport_size: f32, reference: f32) -> f32 {
    switch type {
    case .Absolute:
        return value
    case .Relative:
        return boundary * value
    case .Viewport:
        return viewport_size * value
    case .Remainder:
        return boundary * value
    case .Reference:
        return value * (viewport_size / max(reference, 1))
    case .Aspect:
        return value
    }
    return 0
}

@(private)
flow_size :: proc(using s: Scalar, length: f32, cursor: f32, viewport: f32, reference: f32) -> f32 {
    switch type {
    case .Absolute:
        return value
    case .Relative:
        return length * value
    case .Viewport:
        return viewport * value
    case .Remainder:
        return max(length - cursor, 0) * value
    case .Reference:
        return value * (viewport / max(reference, 1))
    case .Aspect:
        return value
    }
    return 0
}

@(private)
Layout :: struct {
    bounds:   rl.Rectangle,
    spacing:  f32,
    vert: bool,
    cursor:   f32,
}

@(private)
Layout_Stack_Capacity :: 64

@(private)
stack: [Layout_Stack_Capacity]Layout
@(private)
top:   int

@(private)
ref_width:  f32 = 1920
@(private)
ref_height: f32 = 1080

set_reference_resolution :: proc(w: f32, h: f32) {
    ref_width = w
    ref_height = h
}

@(private)
top_layout :: proc() -> (^Layout, bool) {
    if top == 0 {
        return nil, false
    }
    return &stack[top - 1], true
}

@(private)
push_layout :: proc(l: Layout) -> bool {
    if top >= Layout_Stack_Capacity {
        return false
    }
    stack[top] = l
    top += 1
    return true
}

pop :: proc() {
    if top > 0 {
        top -= 1
    }
}

bounds_from_size :: proc(size: Scalar) -> rl.Rectangle {
    if top >= 1 {
        l := &stack[top - 1]
        if l.vert {
            return get_rect(transform({size, {.Relative, 1}}))
        }
        return get_rect(transform({{.Relative, 1}, size}))
    }
    return rl.Rectangle {0, 0, cast(f32)rl.GetScreenWidth(), cast(f32)rl.GetScreenHeight()}
}

@(private)
remaining_vec :: proc() -> Vector {
    l, ok := top_layout()
    if !ok {
        return {}
    }
    if l.vert {
        return {{.Remainder, 1}, {.Relative, 1}}
    }
    return {{.Relative, 1}, {.Remainder, 1}}
}

@(private)
padded :: proc(b: rl.Rectangle, p: f32) -> rl.Rectangle {
    return rl.Rectangle {b.x + p, b.y + p, b.width - 2 * p, b.height - 2 * p}
}


@(private)
MeasurentType :: enum {
    Relative,
    Absolute,
    Viewport,
    Remainder,
    Reference,
    Aspect,
}

Scalar :: struct {
    type:  MeasurentType,
    value: f32,
}

Point :: [2]Scalar

Vector :: [2]Scalar

Transform :: struct {
    position: Point,
    size: Vector,
    scale: Vector,
}

transform :: proc(size: Vector, point: Point = {{.Relative, 0}, {.Relative, 0}}, scale: Vector = {{.Relative, 1}, {.Relative, 1}}) -> Transform {
    return {
        position = point,
        size     = size,
        scale    = scale,
    }
}
