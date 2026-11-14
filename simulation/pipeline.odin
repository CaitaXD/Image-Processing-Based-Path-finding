package simulation

import cv "../odin-opencv"
import "core:math"
import "../conv"
import "../grid"
import "../maps"
import "../nav"

// Experiment 1 of the assignment, with OpenCV doing the image operations:
//
//  1. Mask (input)     cv2.threshold on ONE label image that carries walls,
//                      walkable cells, units and the goal, so the same image
//                      feeds the units and the objective. The labels are
//                      walls 0, walkable 1, units 2, goal 3; thresholds at 0,
//                      1 and 2 split it into wall / unit / goal masks.
//  2. Heat diffusion   cv2.filter2D with the diffusion kernel. Walls are
//                      cleared through the mask: cv2.bitwise_and on CV_32F has
//                      freeze semantics (masked-out cells keep their value), so
//                      cv.Mat.set_to(0, wall_mask) is the operation that
//                      actually zeroes them; on the CV_8U masks themselves
//                      cv2.bitwise_and runs as-is.
//  3. Vector field     cv2.Sobel(X) and cv2.Sobel(Y) over the heat image.
//  4. Wall masking     cv2.bitwise_and again, clearing the flow on walls.
//  5. Resampling       cv2.remap moves the units image downstream along the
//                      normalized vector field.
//
// There is no double buffering: every step writes back into the image it
// reads. The one exception is cv2.remap, which cannot sample the buffer it is
// writing; its output goes into scratch and is copied back into the canonical
// units image instead of being swapped in, so the units stay a single image.

// OpenCV image state of the pipeline. Every cv.Mat is a zero-copy view over
// the grid buffers below (or over the solver's heat field and the goal splat),
// so the probes and the renderer keep reading the same grids.
Pipeline :: struct {
	board:      cv.Mat(u8),
	walkable:   cv.Mat(u8),
	beyond:     cv.Mat(u8),
	goal_inv:   cv.Mat(u8),
	units_mask: cv.Mat(u8),
	goal_mask:  cv.Mat(u8),
	wall_mask:  cv.Mat(u8),

	// The four images the pipeline does not own: views of the simulation's heat
	// field, goal splat, unit image and remap scratch image.
	heat:     cv.Mat(f32),
	goal_src: cv.Mat(f32),
	units:    cv.Mat(f32),
	scratch:  cv.Mat(f32),

	gx:       cv.Mat(f32),
	gy:       cv.Mat(f32),
	sealed:   cv.Mat(f32),
	avg:      cv.Mat(f32),
	wall_cov: cv.Mat(f32),
	map_x:    cv.Mat(f32),
	map_y:    cv.Mat(f32),
	kernel:   cv.Mat(f32),
	box:      cv.Mat(f32),
	coverage: cv.Mat(f32),
}

// pipeline_setup views the simulation's images as OpenCV Mats and allocates
// the pipeline's own mask/gradient/map images. Called from reset, after the
// heat field, goal splat, units and flow field exist.
pipeline_setup :: proc(s: ^Simulation) {
	if s.pipe == nil {
		s.pipe = new(Pipeline)
	}
	p := s.pipe
	w := s.dims[0]
	h := s.dims[1]

	p.board = cv.new_mat(u8, h, w)
	p.walkable = cv.new_mat(u8, h, w)
	p.beyond = cv.new_mat(u8, h, w)
	p.goal_inv = cv.new_mat(u8, h, w)
	p.units_mask = cv.new_mat(u8, h, w)
	p.goal_mask = cv.new_mat(u8, h, w)
	p.wall_mask = cv.new_mat(u8, h, w)

	p.gx = cv.new_mat(f32, h, w)
	p.gy = cv.new_mat(f32, h, w)
	p.sealed = cv.new_mat(f32, h, w)
	p.avg = cv.new_mat(f32, h, w)
	p.wall_cov = cv.new_mat(f32, h, w)
	p.map_x = cv.new_mat(f32, h, w)
	p.map_y = cv.new_mat(f32, h, w)

	kernel, err_kernel := conv.diffusion(HEAT_RATE, HEAT_DECAY)
	assert(err_kernel == .None, "pipeline: cannot build the diffusion kernel")
	p.kernel = cv.new_mat(f32, 3, 3)
	for i in 0..<9 {
		p.kernel.data[i] = kernel.data[i]
	}

	// The normalized convolution divides by the weight the stencil actually has
	// at each cell. Building that weight with the very same filter2D call that
	// will run on the heat image keeps the two in lock step: the divisor is then
	// exactly the sum of the weights filter2D applied, including its zero border,
	// so the update is a true weighted average and cannot amplify. (Rebuilding
	// conv.heat's coverage array here instead left the edge ring and a few
	// wall-adjacent cells under-normalized, and a long relaxation blew them up.)
	p.coverage = cv.new_mat(f32, h, w)
	for i in 0..<int(p.coverage.rows * p.coverage.cols) {
		p.coverage.data[i] = 1 if s.navmap.data[i] == maps.WALKABLE else 0
	}

	// The 3x3 cross used to extend the heat across the walls before the Sobel.
	// wall_cov is the walkable weight of that cross at every cell, so dividing
	// the cross sum of the heat by it turns a wall into the average of its
	// walkable orthogonal neighbours. That is the zero-flux ghost value: the
	// wall above a corridor cell takes the corridor value, not an average that
	// also pulls in the colder diagonal cells.
	p.box = cv.new_mat(f32, 3, 3)
	p.box.data[0 + 1*3] = 0.25
	p.box.data[2 + 1*3] = 0.25
	p.box.data[1 + 0*3] = 0.25
	p.box.data[1 + 2*3] = 0.25
	for y in 0..<h {
		for x in 0..<w {
			weight := f32(0)
			for k in 0..<4 {
				nx := x
				ny := y
				switch k {
				case 0: nx = x - 1
				case 1: nx = x + 1
				case 2: ny = y - 1
				case 3: ny = y + 1
				}
				if nx < 0 || ny < 0 || nx >= w || ny >= h {
					continue
				}
				if s.navmap.data[nx + ny * w] == maps.WALKABLE {
					weight += 0.25
				}
			}
			p.wall_cov.data[x + y * w] = max(weight, f32(1e-6))
		}
	}

	// The pipeline's own images are Mats already. These four view the
	// simulation's buffers: no copy, the same bytes the rest of the game reads.
	p.heat = cv.view(s.heat.field.data, h, w)
	p.goal_src = cv.view(s.goal.data, h, w)
	p.units = cv.view(s.units.data, h, w)
	p.scratch = cv.view(s.units_next.data, h, w)

	cv.filter_2d(&p.coverage, &p.coverage, &p.kernel)
	inv_keep := f32(1) / max(s.heat.keep, f32(1e-6))
	for i in 0..<int(p.coverage.rows * p.coverage.cols) {
		p.coverage.data[i] = max(p.coverage.data[i] * inv_keep, f32(1e-6))
	}
}

// pipeline_masks builds the one label image and thresholds it into the masks
// the rest of the pipeline consumes.
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

	// thresholds are exact label boundaries: 0 -> walls, >1 -> units and
	// goal, >2 -> goal.
	cv.threshold_mask(&p.board, &p.walkable, 0.0, 1, cv.THRESH_BINARY)
	cv.threshold_mask(&p.board, &p.beyond, 1.0, 1, cv.THRESH_BINARY)
	cv.threshold_mask(&p.board, &p.goal_mask, 2.0, 1, cv.THRESH_BINARY)
	cv.threshold_mask(&p.board, &p.goal_inv, 2.0, 1, cv.THRESH_BINARY_INV)
	cv.threshold_mask(&p.board, &p.wall_mask, 0.0, 1, cv.THRESH_BINARY_INV)

	// units = (labels > 1) and (labels <= 2): the same label image feeds the
	// units and the objective.
	cv.and_masks(&p.beyond, &p.goal_inv, &p.units_mask)
}

// pipeline_heat is step 2: convolve the heat image with the diffusion kernel
// and clear the walls, then put the goal splat back. The filter runs in place,
// which OpenCV supports because its filter engine buffers the source rows it
// still needs.
pipeline_heat :: proc(s: ^Simulation) {
	for _ in 0..<HEAT_SUBSTEPS {
		pipeline_heat_substep(s)
	}
}

@(private)
pipeline_heat_substep :: proc(s: ^Simulation) {
	p := s.pipe
	cv.filter_2d(&p.heat, &p.heat, &p.kernel)
	cv.divide_images(&p.heat, &p.coverage, &p.heat)
	cv.set_to(&p.heat, 0, &p.wall_mask)
	cv.add_images(&p.heat, &p.goal_src, &p.heat)
}

// pipeline_prebake relaxes the seeded wash to the steady state of the screened
// Poisson operator before the first frame. Left alone, the wash is a free-space
// Green's function: smooth, wall-blind, and pointing straight at the goal
// through every wall. The steady state is the same operator constrained by the
// walls, and its only maximum sits at the goal, so its Sobel has to go around
// the maze. This is a loop over the diffusion step, but it runs on reset, so
// frame 0 already shows the converged field.
pipeline_prebake :: proc(s: ^Simulation) {
	pipeline_masks(s)
	steps := HEAT_PREBAKE
	for _ in 0..<steps {
		pipeline_heat_substep(s)
	}
}

// pipeline_seal extends the heat field across the walls before the Sobel.
// Sobel reads a 3x3 neighbourhood, so a wall cell holding 0 makes the gradient
// at every adjacent walkable cell point away from that wall instead of down
// the heat slope. Each wall cell is replaced by the average of the walkable
// cells in its own 3x3 window (the same inert-tap weighting the diffusion
// uses); walkable cells keep their heat unchanged.
pipeline_seal :: proc(s: ^Simulation) {
	p := s.pipe
	copy(cv.pixels(&p.sealed), s.heat.field.data)
	cv.filter_2d(&p.sealed, &p.avg, &p.box)
	cv.divide_images(&p.avg, &p.wall_cov, &p.avg)
	cv.copy_to(&p.avg, &p.sealed, &p.wall_mask)
}

// pipeline_flow is steps 3 and 4: Sobel X/Y over the sealed heat image, walls
// cleared, then the gradient becomes the unit vector field the resampling reads.
pipeline_flow :: proc(s: ^Simulation) {
	p := s.pipe

	cv.gradient_x(&p.sealed, &p.gx)
	cv.gradient_y(&p.sealed, &p.gy)
	cv.set_to(&p.gx, 0, &p.wall_mask)
	cv.set_to(&p.gy, 0, &p.wall_mask)

	w := s.dims[0]
	h := s.dims[1]

	for y in 0..<h {
		for x in 0..<w {
			i := x + y * w
			s.flow.field.data[i] = [2]f32{0, 0}
			if s.navmap.data[i] != maps.WALKABLE {
				continue
			}

			gx := p.gx.data[i]
			gy := p.gy.data[i]
			magnitude := math.sqrt(gx * gx + gy * gy)
			if magnitude <= 0 {
				continue
			}
			ux := gx / magnitude
			uy := gy / magnitude

			// Step 4, wall masking, in its continuous form: a unit cannot be
			// advected into a wall, so the component of the field that points
			// into a neighbouring wall is removed. What is left runs along the
			// wall, which is what lets a unit slide past an obstacle instead
			// of stopping dead against it. The direction is kept continuous:
			// quantizing it to one of eight neighbours is what made units hop
			// between two cells forever.
			if ux > 0 && (x + 1 >= w || s.navmap.data[x + 1 + y * w] != maps.WALKABLE) {
				ux = 0
			}
			if ux < 0 && (x - 1 < 0 || s.navmap.data[x - 1 + y * w] != maps.WALKABLE) {
				ux = 0
			}
			if uy > 0 && (y + 1 >= h || s.navmap.data[x + (y + 1) * w] != maps.WALKABLE) {
				uy = 0
			}
			if uy < 0 && (y - 1 < 0 || s.navmap.data[x + (y - 1) * w] != maps.WALKABLE) {
				uy = 0
			}
			slide := math.sqrt(ux * ux + uy * uy)
			if slide <= 0 {
				continue
			}
			ux /= slide
			uy /= slide

			s.flow.field.data[i] = [2]f32{ux, uy}
		}
	}
}

// resample_units is step 5: cv2.remap moves the units image downstream along
// the vector field. The map is built in scatter order, so it says where each
// source cell goes rather than where each destination comes from: a unit at
// (x,y) is written to the cell it reaches after one step, (x+vx, y+vy),
// rounded to the nearest cell. Building the map this way means a moved unit
// leaves nothing behind, so units stay point masses instead of smearing into
// a haze the way bilinear gather sampling does.
//
// Every destination that no unit claims is pointed at a wall cell, whose
// value is always zero. That is how the vacated cells are emptied without a
// clearing pass and without a second unit image in the simulation: the remap
// output lands in scratch and is copied back into the canonical units image.
// A step that would leave the map or enter a wall leaves its unit in place,
// so no unit is dragged through geometry.
resample_units :: proc(
	units: ^grid.Grid2D(f32),
	scratch: ^grid.Grid2D(f32),
	navmap: grid.Grid2D(u8),
	flow: ^nav.FlowField,
	map_x, map_y: ^cv.Mat(f32),
) {
	w := navmap.width
	h := navmap.height

	// A wall cell is never a valid destination, so sampling it always yields
	// zero. Use it as the trash cell for all unclaimed destinations.
	trash := 0
	for i in 0..<len(navmap.data) {
		if navmap.data[i] != maps.WALKABLE {
			trash = i
			break
		}
	}
	// A negative x marks a destination that no unit has claimed yet.
	for i in 0..<int(map_x.rows * map_x.cols) {
		map_x.data[i] = -1
		map_y.data[i] = 0
	}

	// Pass 1: every occupied cell reserves itself. A unit therefore always has
	// a cell to fall back on, and a neighbour can never claim it out from under
	// it; without this reservation a unit that stays put can be erased by the
	// unit behind it, which is where the earlier mass loss came from.
	for i in 0..<len(units.data) {
		if units.data[i] > 0 && navmap.data[i] == maps.WALKABLE {
			map_x.data[i] = f32(i % w)
			map_y.data[i] = f32(i / w)
		}
	}

	// Pass 2: a unit keeps its own cell and moves only into a cell nobody
	// holds. The flow converges on the goal, so this doubles as the congestion
	// rule: a unit whose destination is taken waits where it is instead of
	// being overwritten. The map stays one-to-one, so the remap conserves
	// every unit exactly.
	//
	// A queue has to be released in the order the units are facing, and a
	// single sweep in raster order always visits the front of the queue last,
	// which freezes the line. Repeating the sweep lets the vacated cells hand
	// the movement back along the queue, like a stream of units stepping
	// forward one after the other.
	for _ in 0..<8 {
		moved := 0
		for y in 0..<h {
			for x in 0..<w {
				i := x + y * w
				if units.data[i] <= 0 || navmap.data[i] != maps.WALKABLE {
					continue
				}
				// Already moved this frame: its reservation was released.
				if map_x.data[i] != f32(x) || map_y.data[i] != f32(y) {
					continue
				}
				dir := flow.field.data[i]
				// A cell with no flow is a local maximum of the heat field: the
				// unit there has nowhere to climb, so it stays put.
				if dir[0] == 0 && dir[1] == 0 {
					continue
				}
				dx := int(math.round(dir[0]))
				dy := int(math.round(dir[1]))
				// Flow shorter than half a cell still means "move that way", so
				// step on the dominant axis instead of rounding to zero.
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
				// The whole vector can point into a corner, where the diagonal
				// cell is a wall but one of the two axis neighbours is open.
				// Try the axes as well, which is how a unit slides along a wall
				// instead of freezing in front of it.
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

	// Everything still unclaimed reads the trash cell, i.e. zero.
	for i in 0..<int(map_x.rows * map_x.cols) {
		if map_x.data[i] < 0 {
			map_x.data[i] = f32(trash % w)
			map_y.data[i] = f32(trash / w)
		}
	}

	src := cv.view(units.data, h, w)
	dst := cv.view(scratch.data, h, w)

	// Nearest sampling keeps a unit on the single cell it reached. Bilinear
	// would split it across four cells and, applied every frame, would turn
	// each unit into a widening blur.
	cv.resample(&src, &dst, map_x, map_y, cv.INTER_NEAREST)

	copy(units.data, scratch.data)
}

// pipeline_step is one frame of the OpenCV experiment: masks, heat diffusion,
// vector field, wall masking, resampling.
pipeline_step :: proc(s: ^Simulation, units: ^grid.Grid2D(f32) = nil) {
	target := units
	if target == nil {
		target = &s.units
	}

	pipeline_masks(s)
	pipeline_heat(s)
	pipeline_seal(s)
	pipeline_flow(s)
	resample_units(target, &s.units_next, s.navmap, &s.flow, &s.pipe.map_x, &s.pipe.map_y)
}

