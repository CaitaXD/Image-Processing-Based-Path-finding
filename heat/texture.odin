package heat

import rl "vendor:raylib"
import "../maps"
import cv "../odin-opencv"

update_texture :: proc(tex: rl.Texture2D, img: ^rl.Image, s: ^HeatSolver) {
    max_u: f32
    for value in cv.pixels(s.field) {
        if value > max_u {
            max_u = value
        }
    }
    inv_max := 1 / max(max_u, f32(1e-6))

    pixels := cast([^]rl.Color)img.data
    width := int(s.field.cols)
    height := int(s.field.rows)
    for y in 0..<height {
        for x in 0..<width {
            index := x + y * width
            if cv.at(&s.navmap, x, y)^ == maps.OBSTACLE {
                pixels[index] = {0, 0, 0, 0}
            } else {
                pixels[index] = heat_color(cv.at(&s.field, x, y)^ * inv_max)
            }
        }
    }

    rl.UpdateTexture(tex, img.data)
}

heat_color :: proc(t: f32) -> rl.Color {
    t := clamp(t, 0, 1)
    r, g, b: f32

    if t < 0.25 {
        k := t / 0.25
        r, g, b = 0, k, 1
    } else if t < 0.5 {
        k := (t - 0.25) / 0.25
        r, g, b = 0, 1, 1 - k
    } else if t < 0.75 {
        k := (t - 0.5) / 0.25
        r, g, b = k, 1, 0
    } else {
        k := (t - 0.75) / 0.25
        r, g, b = 1, 1 - k, 0
    }

    shade := 0.15 + 0.85 * t
    return {
        u8(r * 255 * shade),
        u8(g * 255 * shade),
        u8(b * 255 * shade),
        255,
    }
}
