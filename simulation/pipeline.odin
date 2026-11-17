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

	// wall_mask is 1 in a wall and walk is its inverse, so a shifted read of
	// walk answers "is the destination of this move inside the grid and open".
	wall_mask: cv.Mat(f32),
	walk:      cv.Mat(f32),

	weighted: cv.Mat(f32),
	tmp:      cv.Mat(f32),

	// The nine moves a cell can make, in the order of move_dirs: the eight
	// neighbours and staying put. Each image holds 1 where the cell takes that
	// whole step.
	weights: [9]cv.Mat(f32),

	// open[i] is 1 where the destination of move_dirs[i] is inside the grid and
	// not a wall.
	open: [8]cv.Mat(f32),

	// The four signs of the flow, whether each axis moves at all, and the
	// scratches the step works in.
	ax_pos, ax_neg, ay_pos, ay_neg: cv.Mat(f32),
	ax, ay, not_ax, not_ay:         cv.Mat(f32),
	moved, free, xs, ys, free2:     cv.Mat(f32),
	vert, horiz:                    cv.Mat(f32),
}

// move_dirs is the (dx, dy) of each image in Pipeline.weights: east, west,
// south, north, the four diagonals, then no move at all.
move_dirs :: [9][2]i32{
	{1, 0}, {-1, 0}, {0, 1}, {0, -1},
	{1, 1}, {-1, 1}, {1, -1}, {-1, -1},
	{0, 0},
}

// pipeline_alloc allocates the images a w by h pipeline works on. units and
// scratch are left out because they belong to the simulation, and
// pipeline_setup points this pipeline at them.
pipeline_alloc :: proc(p: ^Pipeline, w, h: int) {
	p.board = cv.mat_new(u8, h, w)
	p.walkable = cv.mat_new(u8, h, w)
	p.beyond = cv.mat_new(u8, h, w)
	p.goal_inv = cv.mat_new(u8, h, w)
	p.units_mask = cv.mat_new(u8, h, w)
	p.goal_mask = cv.mat_new(u8, h, w)
	p.wall_mask = cv.mat_new(f32, h, w)
	p.walk = cv.mat_new(f32, h, w)

	for i in 0..<9 {
		p.weights[i] = cv.mat_new(f32, h, w)
	}
	for i in 0..<8 {
		p.open[i] = cv.mat_new(f32, h, w)
	}

	p.ax_pos = cv.mat_new(f32, h, w)
	p.ax_neg = cv.mat_new(f32, h, w)
	p.ay_pos = cv.mat_new(f32, h, w)
	p.ay_neg = cv.mat_new(f32, h, w)
	p.ax = cv.mat_new(f32, h, w)
	p.ay = cv.mat_new(f32, h, w)
	p.not_ax = cv.mat_new(f32, h, w)
	p.not_ay = cv.mat_new(f32, h, w)
	p.moved = cv.mat_new(f32, h, w)
	p.free = cv.mat_new(f32, h, w)
	p.xs = cv.mat_new(f32, h, w)
	p.ys = cv.mat_new(f32, h, w)
	p.free2 = cv.mat_new(f32, h, w)
	p.vert = cv.mat_new(f32, h, w)
	p.horiz = cv.mat_new(f32, h, w)
	p.weighted = cv.mat_new(f32, h, w)
	p.tmp = cv.mat_new(f32, h, w)
}

pipeline_setup :: proc(s: ^Simulation) {
	if s.pipe == nil {
		s.pipe = new(Pipeline)
	}
	p := s.pipe
	pipeline_alloc(p, s.dims[0], s.dims[1])
}

// pipeline_walk derives the wall image and its inverse from a navmap. The unit
// step reads the inverse shifted, so it knows which destinations are open.
pipeline_walk :: proc(p: ^Pipeline, navmap: ^cv.Mat(f32)) {
	cv.threshold(navmap, &p.wall_mask, 0.5, 1, cv.THRESH_BINARY_INV)
	cv.threshold(&p.wall_mask, &p.walk, 0.5, 1, cv.THRESH_BINARY_INV)
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
	pipeline_walk(p, &s.navmap)
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

// resample_units moves the unit image one whole cell along the flow, which is
// the advection: the flow is a unit vector, so thresholding each axis at a half
// turns it into the direction of the step, and the nine moves are the eight
// neighbours plus staying put.
//
// A move whose destination is a wall is never taken. The unit slides instead:
// the x axis of the wanted move is tried, then its y axis, then the open side
// of a dead end, and only a cell with no open neighbour at all stays put.
// Without the slide a unit that faces a wall freezes there for good, and the
// flow of a heat field that is zero inside the walls points into a wall in a
// good many cells.
resample_units :: proc(p: ^Pipeline, units, scratch: ^cv.Mat(f32), flow: ^nav.FlowField) {
	// The sign of each axis, and whether that axis moves at all: a diagonal cell
	// has both, a straight one has one, and a cell with no flow has neither.
	cv.threshold(&flow.gx, &p.ax_pos, 0.5, 1, cv.THRESH_BINARY)
	cv.threshold(&flow.gx, &p.ax_neg, -0.5, 1, cv.THRESH_BINARY_INV)
	cv.threshold(&flow.gy, &p.ay_pos, 0.5, 1, cv.THRESH_BINARY)
	cv.threshold(&flow.gy, &p.ay_neg, -0.5, 1, cv.THRESH_BINARY_INV)
	cv.add(&p.ax_pos, &p.ax_neg, &p.ax)
	cv.add(&p.ay_pos, &p.ay_neg, &p.ay)
	cv.threshold(&p.ax, &p.not_ax, 0.5, 1, cv.THRESH_BINARY_INV)
	cv.threshold(&p.ay, &p.not_ay, 0.5, 1, cv.THRESH_BINARY_INV)

	// Which destinations are open. A shift leaves the far edge untouched, so the
	// image is filled with "closed" first and a step off the grid is not taken
	// either.
	dirs := move_dirs
	for i in 0..<8 {
		cv.fill(&p.open[i], 0)
		cv.shift_copy(&p.walk, -int(dirs[i][0]), -int(dirs[i][1]), &p.open[i])
	}

	// The eight moves, each taken where its own destination is open. A product
	// of these 0/1 images is their overlap, so the diagonals are the two axes
	// together and the straight moves are one axis on its own: a diagonal cell
	// moves once instead of once per axis.
	cv.mul(&p.ax_pos, &p.not_ay, &p.weights[0]) // east
	cv.mul(&p.ax_neg, &p.not_ay, &p.weights[1]) // west
	cv.mul(&p.ay_pos, &p.not_ax, &p.weights[2]) // south
	cv.mul(&p.ay_neg, &p.not_ax, &p.weights[3]) // north
	cv.mul(&p.ax_pos, &p.ay_pos, &p.weights[4]) // south east
	cv.mul(&p.ax_neg, &p.ay_pos, &p.weights[5]) // south west
	cv.mul(&p.ax_pos, &p.ay_neg, &p.weights[6]) // north east
	cv.mul(&p.ax_neg, &p.ay_neg, &p.weights[7]) // north west
	for i in 0..<8 {
		cv.mul(&p.weights[i], &p.open[i], &p.weights[i])
	}

	// moved is 1 where one of the eight moves is taken, and free is its
	// complement: the cells that have to settle for a slide.
	cv.max(&p.weights[0], &p.weights[1], &p.moved)
	cv.max(&p.moved, &p.weights[2], &p.moved)
	cv.max(&p.moved, &p.weights[3], &p.moved)
	cv.max(&p.moved, &p.weights[4], &p.moved)
	cv.max(&p.moved, &p.weights[5], &p.moved)
	cv.max(&p.moved, &p.weights[6], &p.moved)
	cv.max(&p.moved, &p.weights[7], &p.moved)
	cv.threshold(&p.moved, &p.free, 0.5, 1, cv.THRESH_BINARY_INV)

	// The two parts of the wanted move, each where its own axis neighbour is
	// open. xs is tried before ys, the order the flow asks for them in.
	cv.mul(&p.ax_pos, &p.open[0], &p.tmp)
	cv.mul(&p.ax_neg, &p.open[1], &p.xs)
	cv.max(&p.tmp, &p.xs, &p.xs)
	cv.mul(&p.ay_pos, &p.open[2], &p.tmp)
	cv.mul(&p.ay_neg, &p.open[3], &p.ys)
	cv.max(&p.tmp, &p.ys, &p.ys)

	// A straight move is taken where its destination is open, and also where the
	// wanted move was blocked but this axis of it is open: that is the slide
	// along the wall. The y axis only slides where the x axis did not.
	cv.max(&p.not_ay, &p.free, &p.tmp)
	cv.mul(&p.ax_pos, &p.open[0], &p.weights[0])
	cv.mul(&p.weights[0], &p.tmp, &p.weights[0])
	cv.mul(&p.ax_neg, &p.open[1], &p.weights[1])
	cv.mul(&p.weights[1], &p.tmp, &p.weights[1])

	cv.threshold(&p.xs, &p.tmp, 0.5, 1, cv.THRESH_BINARY_INV)
	cv.mul(&p.free, &p.tmp, &p.free2)
	cv.max(&p.not_ax, &p.free2, &p.tmp)
	cv.mul(&p.ay_pos, &p.open[2], &p.weights[2])
	cv.mul(&p.weights[2], &p.tmp, &p.weights[2])
	cv.mul(&p.ay_neg, &p.open[3], &p.weights[3])
	cv.mul(&p.weights[3], &p.tmp, &p.weights[3])

	// A cell whose wanted move runs into a wall dead on, with both of its axes
	// blocked, slides along the wall: the open side of the flow's other axis.
	cv.mul(&p.ay, &p.not_ax, &p.vert)
	cv.mul(&p.ax, &p.not_ay, &p.horiz)
	cv.threshold(&p.ys, &p.tmp, 0.5, 1, cv.THRESH_BINARY_INV)
	cv.mul(&p.free2, &p.tmp, &p.free2)

	cv.threshold(&flow.gx, &p.tmp, -1e-6, 1, cv.THRESH_BINARY)
	cv.mul(&p.tmp, &p.open[0], &p.tmp)
	cv.mul(&p.tmp, &p.vert, &p.tmp)
	cv.mul(&p.tmp, &p.free2, &p.tmp)
	cv.add(&p.weights[0], &p.tmp, &p.weights[0])
	cv.threshold(&flow.gx, &p.tmp, -1e-6, 1, cv.THRESH_BINARY_INV)
	cv.mul(&p.tmp, &p.open[1], &p.tmp)
	cv.mul(&p.tmp, &p.vert, &p.tmp)
	cv.mul(&p.tmp, &p.free2, &p.tmp)
	cv.add(&p.weights[1], &p.tmp, &p.weights[1])
	cv.threshold(&flow.gy, &p.tmp, -1e-6, 1, cv.THRESH_BINARY)
	cv.mul(&p.tmp, &p.open[2], &p.tmp)
	cv.mul(&p.tmp, &p.horiz, &p.tmp)
	cv.mul(&p.tmp, &p.free2, &p.tmp)
	cv.add(&p.weights[2], &p.tmp, &p.weights[2])
	cv.threshold(&flow.gy, &p.tmp, -1e-6, 1, cv.THRESH_BINARY_INV)
	cv.mul(&p.tmp, &p.open[3], &p.tmp)
	cv.mul(&p.tmp, &p.horiz, &p.tmp)
	cv.mul(&p.tmp, &p.free2, &p.tmp)
	cv.add(&p.weights[3], &p.tmp, &p.weights[3])

	// Whatever is left stays: a cell with no flow, and a cell whose every
	// neighbour is a wall.
	cv.add(&p.weights[0], &p.weights[1], &p.tmp)
	cv.add(&p.tmp, &p.weights[2], &p.tmp)
	cv.add(&p.tmp, &p.weights[3], &p.tmp)
	cv.add(&p.tmp, &p.weights[4], &p.tmp)
	cv.add(&p.tmp, &p.weights[5], &p.tmp)
	cv.add(&p.tmp, &p.weights[6], &p.tmp)
	cv.add(&p.tmp, &p.weights[7], &p.tmp)
	cv.threshold(&p.tmp, &p.weights[8], 0.5, 1, cv.THRESH_BINARY_INV)

	// Scatter: every move writes the mass of the cells that chose it into the
	// destination image, offset by that move. The nine moves partition the grid,
	// so every unit is written exactly once and the step conserves mass.
	cv.fill(scratch, 0)
	for i in 0..<9 {
		cv.mul(units, &p.weights[i], &p.weighted)
		cv.shift_add(&p.weighted, int(dirs[i][0]), int(dirs[i][1]), scratch)
	}

	cv.copy_to(scratch, units)
}

pipeline_step :: proc(s: ^Simulation, units: ^cv.Mat(f32) = nil) {
	target := units
	if target == nil {
		target = &s.units
	}

	pipeline_masks(s)
	pipeline_heat(s)
	pipeline_flow(s)
	resample_units(s.pipe, target, &s.units_next, &s.flow)
}
