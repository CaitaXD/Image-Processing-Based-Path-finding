package simulation

import "core:fmt"
import "core:os"
import "core:testing"
import cv "../odin-opencv"

// Writes one 8-bit PNG per pipeline stage for a single frame of the seeded
// fixture, in the order the frame runs them, so each OpenCV call can be shown
// next to the image it produced. Output goes to slides/imagens/etapas/ (run
// from the repository root).
@(test)
stage_dump_writes_pngs :: proc(t: ^testing.T) {
	defer free_all(context.temp_allocator)
	dir :: "slides/imagens/etapas"
	_ = os.make_directory("slides/imagens")
	_ = os.make_directory(dir)

	s := pipeline_headless(SEED)
	p := s.pipe
	w := int(p.board.cols)
	h := int(p.board.rows)

	dump_scaled(t, dir, "00_heat_before", &s.heat.field, 0, max_value(cv.pixels(&s.heat.field)))

	pipeline_masks(&s)
	dump_u8(t, dir, "01_board", &p.board, 85)
	dump_u8(t, dir, "02_walkable", &p.walkable, 255)
	dump_u8(t, dir, "03_beyond", &p.beyond, 255)
	dump_u8(t, dir, "04_goal_mask", &p.goal_mask, 255)
	dump_u8(t, dir, "05_goal_inv", &p.goal_inv, 255)
	dump_u8(t, dir, "06_wall_mask", &p.wall_mask, 255)
	dump_u8(t, dir, "07_units_mask", &p.units_mask, 255)

	pipeline_heat(&s)
	dump_scaled(t, dir, "08_heat_after", &s.heat.field, 0, max_value(cv.pixels(&s.heat.field)))

	pipeline_flow(&s)
	dump_scaled(t, dir, "09_flow_gx", &s.flow.gx, -1, 1)
	dump_scaled(t, dir, "10_flow_gy", &s.flow.gy, -1, 1)
	dump_scaled(t, dir, "11_flow_mag", &s.flow.mag, 0, max_value(cv.pixels(&s.flow.mag)))

	before, err := cv.mat_new(f32, h, w, context.temp_allocator)
	assert(err == .None)
	copy(cv.pixels(&before), cv.pixels(&s.units))
	dump_scaled(t, dir, "12_units_before", &before, 0, 1)

	resample_units(&s.units, &s.units_next, s.navmap, &s.flow, &p.map_x, &p.map_y)
	dump_displacement(t, dir, "13_map_dx", &p.map_x, 0)
	dump_displacement(t, dir, "14_map_dy", &p.map_y, 1)
	dump_scaled(t, dir, "15_units_after", &s.units, 0, 1)

	mass_before := f32(0)
	for v in cv.pixels(&before) {
		mass_before += v
	}
	mass_after := f32(0)
	for v in cv.pixels(&s.units) {
		mass_after += v
	}
	testing.expect(t, abs(mass_after - mass_before) < 0.01, "the remap must conserve unit mass")
}

@(private)
dump_png :: proc(t: ^testing.T, dir, name: string, img: ^cv.Mat(u8)) {
	path := fmt.ctprintf("%s/%s.png", dir, name)
	testing.expect(t, cv.imwrite(path, img) == 1, "stage image must be written")
}

@(private)
dump_u8 :: proc(t: ^testing.T, dir, name: string, src: ^cv.Mat(u8), scale: u8) {
	out, err := cv.mat_new(u8, int(src.rows), int(src.cols), context.temp_allocator)
	assert(err == .None)
	from := cv.pixels(src)
	to := cv.pixels(&out)
	for i in 0 ..< len(from) {
		to[i] = u8(min(int(from[i]) * int(scale), 255))
	}
	dump_png(t, dir, name, &out)
}

@(private)
dump_scaled :: proc(t: ^testing.T, dir, name: string, src: ^cv.Mat(f32), lo, hi: f32) {
	out, err := cv.mat_new(u8, int(src.rows), int(src.cols), context.temp_allocator)
	assert(err == .None)
	from := cv.pixels(src)
	to := cv.pixels(&out)
	top := hi
	if top <= lo {
		top = lo + 1
	}
	for i in 0 ..< len(from) {
		v := (from[i] - lo) / (top - lo)
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		to[i] = u8(v * 255)
	}
	dump_png(t, dir, name, &out)
}

// Shows where each cell's source pixel sits relative to itself: 128 means no
// move, lighter/darker means the unit is pulled from further right/below or
// further left/above.
@(private)
dump_displacement :: proc(t: ^testing.T, dir, name: string, map_src: ^cv.Mat(f32), axis: int) {
	w := int(map_src.cols)
	out, err := cv.mat_new(u8, int(map_src.rows), w, context.temp_allocator)
	assert(err == .None)
	from := cv.pixels(map_src)
	to := cv.pixels(&out)
	for i in 0 ..< len(from) {
		d: f32
		if axis == 0 {
			d = from[i] - f32(i % w)
		} else {
			d = from[i] - f32(i / w)
		}
		v := 128 + 32 * d
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		to[i] = u8(v)
	}
	dump_png(t, dir, name, &out)
}

@(private)
max_value :: proc(px: []f32) -> f32 {
	m: f32 = 0
	for v in px {
		if v > m {
			m = v
		}
	}
	return m
}
