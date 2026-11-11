package main

import "core:fmt"
import "core:os"
import rl "vendor:raylib"

RECORD_SECONDS :: 10
RECORD_WIDTH :: W * WINDOW_SCALE
RECORD_HEIGHT :: H * WINDOW_SCALE

Recorder :: struct {
    outputs: [VISUALIZATION_COUNT]RecorderOutput,
    active:  bool,
    frames:  int,
    target:  rl.RenderTexture2D,
}

RecorderOutput :: struct {
    process:       os.Process,
    pipe:          ^os.File,
    visualization: Visualization,
}

visualization_name :: proc(v: Visualization) -> string {
    switch v {
    case .NONE:
        return "none"
    case .HEAT:
        return "heat"
    }
    return "unknown"
}

record_start :: proc(rec: ^Recorder) -> bool {
    if rec.active {
        return true
    }

    rec.target = rl.LoadRenderTexture(RECORD_WIDTH, RECORD_HEIGHT)
    if rec.target.id == 0 {
        fmt.eprintln("record: failed to create render texture")
        return false
    }

    rec.frames = 0
    rec.active = true
    rl.SetTraceLogLevel(.WARNING)
    for v in VISUALIZATIONS {
        if !record_start_output(rec, v) {
            record_stop(rec)
            return false
        }
    }

    fmt.println("recording all visualizations from the start")
    return true
}

@(private)
record_start_output :: proc(rec: ^Recorder, visualization: Visualization) -> bool {
    r, w, pipe_err := os.pipe()
    if pipe_err != nil {
        fmt.eprintln("record: pipe failed:", pipe_err)
        return false
    }

    size := fmt.tprintf("%dx%d", RECORD_WIDTH, RECORD_HEIGHT)
    fps := fmt.tprintf("%d", FPS)
    path := fmt.tprintf("procimg_%s.mp4", visualization_name(visualization))
    command := []string{
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pixel_format", "rgba",
        "-video_size", size,
        "-framerate", fps,
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        path,
    }

    desc := os.Process_Desc {
        command = command,
        stdin   = r,
        stderr  = os.stderr,
    }
    process, start_err := os.process_start(desc)
    os.close(r)
    if start_err != nil {
        os.close(w)
        fmt.eprintln("record: ffmpeg failed:", start_err)
        return false
    }

    rec.outputs[int(visualization)] = RecorderOutput {
        process       = process,
        pipe          = w,
        visualization = visualization,
    }
    fmt.println("recording to", path)
    return true
}

record_stop :: proc(rec: ^Recorder) {
    for &output in rec.outputs {
        if output.pipe != nil {
            os.close(output.pipe)
            _, _ = os.process_wait(output.process)
            output.pipe = nil
        }
    }

    if rec.target.id != 0 {
        rl.UnloadRenderTexture(rec.target)
        rec.target = {}
    }

    if rec.active {
        rec.active = false
        rl.SetTraceLogLevel(.INFO)
        fmt.println("recording saved")
    }
}

// Every visualization is rendered from the same simulation frame, so all
// outputs share an identical timeline.
record_capture :: proc(rec: ^Recorder, sim: ^Simulation, camera: rl.Camera2D) {
    if !rec.active {
        return
    }

    for v in VISUALIZATIONS {
        rl.BeginTextureMode(rec.target)
        rl.ClearBackground({0x18, 0x18, 0x18, 255})
        simulation_draw(sim, v, camera)
        rl.DrawText(rl.TextFormat("%s  %d/%d", visualization_name(v), rec.frames + 1, RECORD_SECONDS * FPS), 10, 10, 20, rl.WHITE)
        rl.EndTextureMode()

        img := rl.LoadImageFromTexture(rec.target.texture)
        rl.ImageFlipVertical(&img)
        ok := record_write(rec, v, img)
        rl.UnloadImage(img)
        if !ok {
            return
        }
    }

    rec.frames += 1
    if rec.frames >= RECORD_SECONDS * FPS {
        record_stop(rec)
    }
}

@(private)
record_write :: proc(rec: ^Recorder, visualization: Visualization, img: rl.Image) -> bool {
    output := &rec.outputs[int(visualization)]
    if output.pipe == nil {
        return false
    }

    pixels := cast([^]byte)img.data
    data := pixels[:img.width * img.height * 4]
    if !write_all(output.pipe, data) {
        fmt.eprintln("record: write failed for", visualization_name(visualization))
        record_stop(rec)
        return false
    }
    return true
}

@(private)
write_all :: proc(f: ^os.File, data: []byte) -> bool {
    remaining := data
    for len(remaining) > 0 {
        n, err := os.write(f, remaining)
        if err != nil || n <= 0 {
            return false
        }
        remaining = remaining[n:]
    }
    return true
}
