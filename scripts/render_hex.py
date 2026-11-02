"""Render a recorded hexagon round to video, with sound made from its event log.

Run:  npm run dev -- --port 5199            in another terminal
      python scripts/render_hex.py --seed 5 --out renders/hex.mp4

Plays at real speed from start to finish. Needs playwright and ffmpeg.
"""

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

FPS = 30
RATE = 44100


def warp(duration, win, hold=4.0):
    """Video time to round time, as a list of round times, one per frame."""
    return [k / FPS for k in range(int(min(win + hold, duration) * FPS))]


def tone(freq, seconds, gain=0.2, decay=8.0, sweep=0.0):
    t = np.arange(int(seconds * RATE)) / RATE
    phase = 2 * np.pi * (freq * t + sweep * t * t / 2)
    return gain * np.exp(-decay * t) * (np.sin(phase) + 0.25 * np.sin(2 * phase))


def thump(gain, seed, lower=1.0):
    """A low, round drop. Noise bursts here sounded like a crash on every one of about 70 landings a round."""
    pitch = lower * (0.9 + 0.2 * np.random.default_rng(seed).random())
    t = np.arange(int(0.18 * RATE)) / RATE
    freq = pitch * (70 + 100 * np.exp(-30 * t))
    glide = np.sin(2 * np.pi * np.cumsum(freq) / RATE)
    out = (glide + 0.3 * np.sign(glide) * glide * glide) * np.exp(-20 * t) * np.minimum(t / 0.003, 1.0)
    return gain * out / np.abs(out).max()


def add(track, at, sound):
    start = int(at * RATE)
    if start >= len(track):
        return
    end = min(start + len(sound), len(track))
    track[start:end] += sound[: end - start]


def soundtrack(meta, times):
    """Soft, dry hits only. No sustained notes, no drones."""
    frame_of = np.array(times)
    video_time = lambda t: float(np.searchsorted(frame_of, t)) / FPS
    track = np.zeros(int(len(times) / FPS * RATE) + RATE)
    countdown = meta["arena"]["countdownSeconds"]
    for k in range(int(countdown)):
        add(track, k + 0.05, tone(523, 0.18, 0.14))
    add(track, countdown, tone(784, 0.35, 0.2, 5))
    last_touch = -1.0
    pitches = [660, 740, 830, 880, 990]
    for index, event in enumerate(meta["events"]):
        if event["t"] > frame_of[-1]:
            break
        at = video_time(event["t"])
        kind = event["type"]
        if kind == "tile_gone" and at - last_touch > 0.045:
            add(track, at, tone(pitches[index % 5], 0.07, 0.035, 40))
            last_touch = at
        elif kind == "fall":
            add(track, at, tone(420, 0.34, 0.09, 7, sweep=-700))
        elif kind == "land":
            add(track, at, thump(0.45, index))
        elif kind == "knock":
            add(track, at, thump(0.585, index))
            add(track, at, tone(180, 0.22, 0.2, 9, sweep=500))
        elif kind == "leap":
            add(track, at, tone(300, 0.2, 0.05, 11, sweep=900))
        elif kind == "splash":
            add(track, at, thump(0.675, index, lower=0.75))
            add(track, at, tone(200, 0.3, 0.1, 10, sweep=-200))
        elif kind == "winner":
            for k, freq in enumerate((523, 659, 784, 1047)):
                add(track, at + 0.12 * k, tone(freq, 0.5, 0.17, 5))
    peak = np.abs(track).max()
    return (np.clip(track / max(peak, 1e-6) * 0.85, -1, 1) * 32767).astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--dir", default="hex")
    parser.add_argument("--out", default="renders/hex.mp4")
    parser.add_argument("--url", default="http://localhost:5199/hex.html")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(f"public/{args.dir}/seed_{args.seed}.json").read_text())
    win = next(e["t"] for e in meta["events"] if e["type"] == "winner")
    times = warp((meta["frames"] - 1) / meta["fps"], win)
    wav_path = out.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(soundtrack(meta, times).tobytes())
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-", "-i", str(wav_path),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--use-angle=d3d11", "--enable-webgl", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(f"{args.url}?capture=1&seed={args.seed}&dir={args.dir}")
        page.wait_for_selector("body[data-ready='1']", timeout=120000)
        for index, t in enumerate(times):
            page.evaluate(f"seekRound({t})")
            encoder.stdin.write(page.screenshot(type="png"))
            if index % 150 == 0:
                print(f"{index}/{len(times)}", flush=True)
        browser.close()
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("ffmpeg failed")
    wav_path.unlink()
    print(f"{out}: {len(times) / FPS:.1f} s, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
