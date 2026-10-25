"""Normalize generated sheets using the Game Studio sprite pipeline.

This performs deterministic atlas extraction/packing, not image synthesis.
Original generated sheets are kept in assets/source, outside the web bundle.
"""
from pathlib import Path
import importlib.util
import json
import math
import random
import struct
import sys
import wave

UI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI / ".python-deps"))
from PIL import Image

spec = importlib.util.spec_from_file_location("normalize", Path(__file__).parent / "sprite-pipeline/normalize_sprite_strip.py")
normalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalizer)
DEST = UI / "public/assets/v2"
SOURCE = UI / "assets/source"


def grid(path, cols, rows):
    image = Image.open(path).convert("RGBA")
    return [image.crop((round(c * image.width / cols), round(r * image.height / rows),
                        round((c + 1) * image.width / cols), round((r + 1) * image.height / rows)))
            for r in range(rows) for c in range(cols)]


def pack(frames, path, normalize=True):
    if normalize:
        content = [normalizer.crop_to_content(frame, 24) for frame in frames]
        width, height = normalizer.max_content_size(content)
        scale = min(30 / width, 30 / height)
        frames = [normalizer.compose_frame(frame, 32, scale) for frame in content]
    else:
        frames = [frame.resize((32, 32), Image.Resampling.NEAREST) for frame in frames]
    strip = Image.new("RGBA", (len(frames) * 32, 32))
    for index, frame in enumerate(frames):
        strip.alpha_composite(frame, (index * 32, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(path, optimize=True)
    return frames


def sound(name, frequencies, duration, noise=0.):
    rate = 22050
    rng = random.Random(1022)
    samples = []
    for i in range(round(duration * rate)):
        t = i / rate
        phase = t / duration
        f = frequencies[min(len(frequencies) - 1, int(phase * len(frequencies)))]
        envelope = min(t / .008, 1.) * (1 - phase) ** 1.8
        tone = math.sin(2 * math.pi * f * t) * .65 + (rng.random() * 2 - 1) * noise
        samples.append(struct.pack("<h", max(-32767, min(32767, int(tone * envelope * 17000)))))
    with wave.open(str(DEST / f"audio/{name}.wav"), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"".join(samples))


ICONS = {
    "play": '<path d="m9 5 11 7-11 7z"/>',
    "watch": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
    "train": '<path d="M8 3h8M10 3v7L4 20h16l-6-10V3M8 15h8"/>',
    "pause": '<path d="M7 5h3v14H7zM15 5h3v14h-3z"/>',
    "step": '<path d="m5 5 11 7-11 7zM20 5v14"/>',
    "reset": '<path d="M4 9a8 8 0 1 1 0 7M4 3v6h6"/>',
    "settings": '<path d="m9 3-1 3-3 1 1 3-2 2 2 2-1 3 3 1 1 3h6l1-3 3-1-1-3 2-2-2-2 1-3-3-1-1-3Z"/><circle cx="12" cy="12" r="3"/>',
    "sound": '<path d="m3 9 5 0 5-4v14l-5-4H3ZM17 8a6 6 0 0 1 0 8M20 5a10 10 0 0 1 0 14"/>',
    "bomb": '<circle cx="11" cy="14" r="7"/><path d="m15 8 2-3 3 1M19 2v2M22 5h-2M8 12l2-2"/>',
    "range": '<path d="M12 3v18M3 12h18m-6-6-3-3-3 3m6 12-3 3-3-3M6 9l-3 3 3 3m12-6 3 3-3 3"/>',
    "alive": '<path d="M5 9V5h14v4M3 9h18v11H3zM8 13v3M16 13v3M10 3h4"/>',
    "danger": '<path d="m12 3 10 18H2ZM12 9v5M12 17v1"/>',
    "close": '<path d="m6 6 12 12M6 18 18 6"/>',
    "back": '<path d="m10 5-7 7 7 7M3 12h18"/>',
    "arrow": '<path d="M4 12h16m-6-6 6 6-6 6"/>',
    "up": '<path d="m5 15 7-7 7 7"/>',
    "down": '<path d="m5 9 7 7 7-7"/>',
    "left": '<path d="m15 5-7 7 7 7"/>',
    "right": '<path d="m9 5 7 7-7 7"/>',
    "check": '<path d="m4 12 5 5L20 6"/>',
    "home": '<path d="m3 10 9-7 9 7v11H3ZM9 21v-8h6v8"/>',
}


def main():
    for folder in ("characters", "environment", "fx", "ui", "audio"):
        (DEST / folder).mkdir(parents=True, exist_ok=True)
    preview = Image.new("RGBA", (32 * 12, 32 * 12), "#101820")
    characters = []
    for row, color in enumerate(("mint", "blue", "coral", "gold")):
        frames = pack(grid(SOURCE / f"robot-{color}.png", 6, 6), DEST / f"characters/{color}.png")
        frames[0].save(DEST / f"characters/{color}-avatar.png")
        for i, frame in enumerate(frames):
            preview.alpha_composite(frame, ((i % 12) * 32, (row * 3 + i // 12) * 32))
        characters.append({"id": color, "path": f"/assets/v2/characters/{color}.png", "frames": 36,
                           "avatar": f"/assets/v2/characters/{color}-avatar.png", "anchor": [0.5, 1]})
    (UI / "assets/previews").mkdir(parents=True, exist_ok=True)
    preview.resize((1152, 1152), Image.Resampling.NEAREST).save(UI / "assets/previews/characters.png")
    pack(grid(SOURCE / "terrain.png", 4, 3), DEST / "environment/terrain.png", normalize=False)
    pack(grid(SOURCE / "effects.png", 8, 9)[:68], DEST / "fx/effects.png")
    Image.open(SOURCE / "menu.png").convert("RGB").save(DEST / "ui/foundry.webp", quality=88, method=6)
    (DEST / "ui/logo.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path fill="#a6efc8" d="M18 24h28v8h8v20h-8v8H18v-8h-8V32h8z"/><path fill="#101820" d="M22 34h6v8h-6zm14 0h6v8h-6z"/><path fill="#f4c95d" d="M36 8h8v8h-8zm8 8h8v8h-8zM24 16h12v8H24z"/></svg>', encoding="utf-8")
    symbols = ''.join(f'<symbol id="{name}" viewBox="0 0 24 24">{content}</symbol>' for name, content in ICONS.items())
    (DEST / "ui/icons.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'+symbols+'</svg>', encoding="utf-8")
    sound("bomb", [130, 90], .16)
    sound("fuse", [680, 900], .09)
    sound("explosion", [105, 65, 32], .32, noise=.65)
    sound("pickup", [440, 660, 880], .22)
    sound("defeat", [360, 260, 130], .28)
    sound("win", [392, 494, 587, 784], .65)
    sound("draw", [330, 294, 262], .4)
    sound("click", [560], .045)
    files = ['/'+p.relative_to(UI / 'public').as_posix() for p in DEST.rglob('*') if p.is_file() and p.name != 'manifest.json']
    manifest = {"version": 2, "frameSize": 32, "characters": characters,
                "terrain": {"path": "/assets/v2/environment/terrain.png", "frames": 12},
                "effects": {"path": "/assets/v2/fx/effects.png", "frames": 68},
                "files": sorted(files), "provenance": "Generated with built-in ImageGen; normalized with Game Studio 0.1.2; docs/asset-prompts.md"}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared 4x36 character frames, 12 tiles, 68 FX frames, 8 original SFX and {len(files)} runtime files.")


if __name__ == "__main__":
    main()
