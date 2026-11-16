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

	origin_x: cv.Mat(f32),
	origin_y: cv.Mat(f32),

	dir_x: cv.Mat(f32),
	dir_y: cv.Mat(f32),

	half: cv.Mat(f32),
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
	p.origin_x = cv.mat_new(f32, h, w)
	p.origin_y = cv.mat_new(f32, h, w)
	cv.identity_maps(&p.origin_x, &p.origin_y)
	p.dir_x = cv.mat_new(f32, h, w)
	p.dir_y = cv.mat_new(f32, h, w)
	p.half = cv.mat_new(f32, h, w)
	cv.fill(&p.half, 0.5)

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
	flow: ^nav.FlowField,
	map_x, map_y: ^cv.Mat(f32),
	half: ^cv.Mat(f32),
	dir_x, dir_y: ^cv.Mat(f32),
	origin_x, origin_y: ^cv.Mat(f32),
) {
	cv.sub(origin_x, &flow.gx, map_x)
	cv.sub(origin_y, &flow.gy, map_y)
	// The trace runs to the midpoint and then takes one full step from there,
	// which lands on the cell that actually steps into this destination instead
	// of drifting off a curving flow.
	cv.mul(&flow.gx, half, dir_x)
	cv.mul(&flow.gy, half, dir_y)
	cv.sub(origin_x, dir_x, map_x)
	cv.sub(origin_y, dir_y, map_y)
	cv.resample(&flow.gx, dir_x, map_x, map_y, cv.INTER_NEAREST)
	cv.resample(&flow.gy, dir_y, map_x, map_y, cv.INTER_NEAREST)
	cv.sub(origin_x, dir_x, map_x)
	cv.sub(origin_y, dir_y, map_y)
	cv.resample(units, scratch, map_x, map_y, cv.INTER_NEAREST)
	cv.set_to(scratch, 0, &flow.wall_mask)
	copy(units.data[0:int(units.rows * units.cols)], scratch.data[0:int(scratch.rows * scratch.cols)])
}

pipeline_step :: proc(s: ^Simulation, units: ^cv.Mat(f32) = nil) {
	target := units
	if target == nil {
		target = &s.units
	}

	pipeline_masks(s)
	pipeline_heat(s)
	pipeline_flow(s)
	resample_units(target, &s.units_next, &s.flow, &s.pipe.map_x, &s.pipe.map_y, &s.pipe.half, &s.pipe.dir_x, &s.pipe.dir_y, &s.pipe.origin_x, &s.pipe.origin_y)
}

