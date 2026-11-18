package simulation

import cv "../odin-opencv"
import "core:math"
import "../heat"
import "../maps"
import "../nav"

Pipeline :: struct {
	board:      cv.Mat(u8),
	walkable:   cv.Mat(u8),
	beyond:     cv.Mat(u8),
	goal_inv:   cv.Mat(u8),
	units_mask: cv.Mat(u8),
	goal_mask:  cv.Mat(u8),
	wall_mask:  cv.Mat(u8),

	units:   cv.Mat(f32),
	scratch: cv.Mat(f32),

	map_x:    cv.Mat(f32),
	map_y:    cv.Mat(f32),
}

pipeline_setup :: proc(s: ^Simulation) {
	if s.pipe == nil {
		s.pipe = new(Pipeline)
	}
	p := s.pipe
	w := s.dims[0]
	h := s.dims[1]

	p.board = cv.mat_new(u8, h, w)
	p.walkable = cv.mat_new(u8, h, w)
	p.beyond = cv.mat_new(u8, h, w)
	p.goal_inv = cv.mat_new(u8, h, w)
	p.units_mask = cv.mat_new(u8, h, w)
	p.goal_mask = cv.mat_new(u8, h, w)
	p.wall_mask = cv.mat_new(u8, h, w)

	p.map_x = cv.mat_new(f32, h, w)
	p.map_y = cv.mat_new(f32, h, w)

	p.units = s.units
	p.scratch = s.units_next
}

pipeline_masks :: proc(s: ^Simulation) {
	p := s.pipe
	for i in 0..<int(p.board.rows * p.board.cols) {
		if s.navmap.data[i] != maps.WALKABLE {
			p.board.data[i] = 0
			continue
		}
		if s.goal.data[i] > 0 {
			p.board.data[i] = 3
			continue
		}
		if s.units.data[i] > 0 {
			p.board.data[i] = 2
			continue
		}
		p.board.data[i] = 1
	}

	cv.threshold(&p.board, &p.walkable, 0.0, 1, cv.THRESH_BINARY)
	cv.threshold(&p.board, &p.beyond, 1.0, 1, cv.THRESH_BINARY)
	cv.threshold(&p.board, &p.goal_mask, 2.0, 1, cv.THRESH_BINARY)
	cv.threshold(&p.board, &p.goal_inv, 2.0, 1, cv.THRESH_BINARY_INV)
	cv.threshold(&p.board, &p.wall_mask, 0.0, 1, cv.THRESH_BINARY_INV)
	cv.and(&p.beyond, &p.goal_inv, &p.units_mask)
}

pipeline_heat :: proc(s: ^Simulation) {
	heat.step(&s.heat, &s.goal, HEAT_SUBSTEPS)
}

pipeline_prebake :: proc(s: ^Simulation) {
	pipeline_masks(s)
	heat.step(&s.heat, &s.goal, HEAT_PREBAKE)
}

pipeline_flow :: proc(s: ^Simulation) {
	nav.update(&s.flow, &s.heat)
}

resample_units :: proc(
	units: ^cv.Mat(f32),
	scratch: ^cv.Mat(f32),
	navmap: cv.Mat(f32),
	flow: ^nav.FlowField,
	map_x, map_y: ^cv.Mat(f32),
) {
	w := int(navmap.cols)
	h := int(navmap.rows)

	trash := 0
	for i in 0..<w * h {
		if navmap.data[i] != maps.WALKABLE {
			trash = i
			break
		}
	}
	for i in 0..<int(map_x.rows * map_x.cols) {
		map_x.data[i] = -1
		map_y.data[i] = 0
	}

	for i in 0..<w * h {
		if units.data[i] > 0 && navmap.data[i] == maps.WALKABLE {
			map_x.data[i] = f32(i % w)
			map_y.data[i] = f32(i / w)
		}
	}

	for _ in 0..<8 {
		moved := 0
		for y in 0..<h {
			for x in 0..<w {
				i := x + y * w
				if units.data[i] <= 0 || navmap.data[i] != maps.WALKABLE {
					continue
				}
				if map_x.data[i] != f32(x) || map_y.data[i] != f32(y) {
					continue
				}
				dir := flow.field.data[i]
				if dir[0] == 0 && dir[1] == 0 {
					continue
				}
				dx := int(math.round(dir[0]))
				dy := int(math.round(dir[1]))
				if dx == 0 && dy == 0 {
					ax := dir[0]
					if ax < 0 {
						ax = -ax
					}
					ay := dir[1]
					if ay < 0 {
						ay = -ay
					}
					if ax >= ay {
						if dir[0] > 0 {
							dx = 1
						} else {
							dx = -1
						}
					} else {
						if dir[1] > 0 {
							dy = 1
						} else {
							dy = -1
						}
					}
				}
				dest := -1
				candidates := [3][2]int{{dx, dy}, {dx, 0}, {0, dy}}
				for c in candidates {
					if c[0] == 0 && c[1] == 0 {
						continue
					}
					nx := x + c[0]
					ny := y + c[1]
					if nx < 0 || ny < 0 || nx >= w || ny >= h {
						continue
					}
					j := nx + ny * w
					if navmap.data[j] != maps.WALKABLE || map_x.data[j] >= 0 {
						continue
					}
					dest = j
					break
				}
				if dest < 0 {
					best := f32(-2)
					sides := [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
					for c in sides {
						nx := x + c[0]
						ny := y + c[1]
						if nx < 0 || ny < 0 || nx >= w || ny >= h {
							continue
						}
						j := nx + ny * w
						if navmap.data[j] != maps.WALKABLE || map_x.data[j] >= 0 {
							continue
						}
						along := f32(c[0]) * dir[0] + f32(c[1]) * dir[1]
						if along > best {
							best = along
							dest = j
						}
					}
				}
				if dest < 0 {
					continue
				}
				map_x.data[i] = -1
				map_x.data[dest] = f32(x)
				map_y.data[dest] = f32(y)
				moved += 1
			}
		}
		if moved == 0 {
			break
		}
	}

	for i in 0..<int(map_x.rows * map_x.cols) {
		if map_x.data[i] < 0 {
			map_x.data[i] = f32(trash % w)
			map_y.data[i] = f32(trash / w)
		}
	}

	cv.resample(units, scratch, map_x, map_y, cv.INTER_NEAREST)
	copy(units.data[:w * h], scratch.data[:w * h])
}

pipeline_step :: proc(s: ^Simulation, units: ^cv.Mat(f32) = nil) {
	target := units
	if target == nil {
		target = &s.units
	}

	pipeline_masks(s)
	pipeline_heat(s)
	pipeline_flow(s)
	resample_units(target, &s.units_next, s.navmap, &s.flow, &s.pipe.map_x, &s.pipe.map_y)
}

