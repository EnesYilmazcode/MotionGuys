"""Render a recorded hexagon round to video on Modal, in parallel chunks, on CPU graphics.

Run:  npm run build
      modal run scripts/modal_render.py --seed 7 --out renders/hex-bots.mp4

Uses the same time warp and sound as render_hex.py, without the laptop's GPU or memory.
"""

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import modal
from modal_progress import bar, watch

from render_hex import FPS, RATE, soundtrack, warp

ROOT = Path(__file__).resolve().parent.parent
CHUNK_FRAMES = 40

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("playwright", "numpy", "modal-progress")
    .run_commands("playwright install --with-deps chromium")
    .add_local_dir(ROOT / "dist", "/dist")
    .add_local_python_source("render_hex")
)
app = modal.App("motionguys-render", image=image)


@app.function(cpu=8, memory=8192, timeout=1500, retries=1)
def render_chunk(number: int, times: list[float], seed: int, folder: str, width: int, height: int, query: str) -> tuple[int, bytes]:
    from playwright.sync_api import sync_playwright

    server = subprocess.Popen(["python", "-m", "http.server", "8000", "--directory", "/dist"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = Path(tempfile.mkdtemp()) / "chunk.mp4"
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
            page = browser.new_page(viewport={"width": width, "height": height})
            # CPU graphics can take far longer than the default 30 s on a heavy frame.
            page.set_default_timeout(300000)
            page.goto(f"http://localhost:8000/hex.html?capture=1&seed={seed}&dir={folder}{query}", timeout=180000)
            page.wait_for_selector("body[data-ready='1']", timeout=300000)
            # A fresh page starts at time 0, so let the start banner finish sliding away or each chunk opens with a flicker.
            page.evaluate(f"seekRound({times[0]})")
            page.wait_for_timeout(1500)
            for t in bar(times, desc=f"chunk {number}"):
                page.evaluate(f"seekRound({t})")
                encoder.stdin.write(page.screenshot(type="png"))
            browser.close()
    finally:
        encoder.stdin.close()
        server.terminate()
    if encoder.wait():
        raise RuntimeError("ffmpeg failed")
    return number, out.read_bytes()


@app.local_entrypoint()
def main(seed: int = 7, out: str = "renders/hex-bots.mp4", folder: str = "hex", width: int = 1920, height: int = 1080, frames: int = 0, start: float = 0.0, end: float = 0.0, follow: int = -1):
    target = ROOT / out
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "public" / folder / f"seed_{seed}.json").read_text())
    length = (meta["frames"] - 1) / meta["fps"]
    # A test round cut short has no winner.
    win = next((e["t"] for e in meta["events"] if e["type"] == "winner"), length)
    times = warp(length, win)
    # A short test render.
    times = times[:frames] if frames else times
    # A look at one moment, optionally following one robot.
    times = [t for t in times if t >= start and (not end or t <= end)]
    query = f"&follow={follow}" if follow >= 0 else ""
    chunks = [times[i:i + CHUNK_FRAMES] for i in range(0, len(times), CHUNK_FRAMES)]
    print(f"{len(times)} frames in {len(chunks)} chunks")
    with tempfile.TemporaryDirectory() as temp:
        folder_path = Path(temp)
        calls = [(n, chunk, seed, folder, width, height, query) for n, chunk in enumerate(chunks)]
        paths = {}
        with watch(app, name="render") as progress:
            for number, data in progress.track(render_chunk.starmap(calls, order_outputs=False), total=len(calls)):
                path = folder_path / f"{number:04d}.mp4"
                path.write_bytes(data)
                paths[number] = path
        if len(paths) != len(chunks):
            raise RuntimeError(f"only {len(paths)} of {len(chunks)} chunks came back")
        listing = folder_path / "list.txt"
        listing.write_text("".join(f"file '{paths[n].as_posix()}'\n" for n in range(len(chunks))))
        sound = folder_path / "sound.wav"
        with wave.open(str(sound), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(RATE)
            file.writeframes(soundtrack(meta, times).tobytes())
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(sound),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(target)],
            check=True,
        )
    print(f"{target}: {len(times) / FPS:.1f} s, {target.stat().st_size / 1e6:.1f} MB")
