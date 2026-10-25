"""Production asset acceptance: actual packaged files, no legacy graphics."""
import hashlib
import json
from pathlib import Path
import unittest

UI = Path(__file__).resolve().parents[2]


class AssetTests(unittest.TestCase):
    def test_runtime_manifest_has_four_complete_character_sheets(self):
        path = UI / "public/assets/v2/manifest.json"
        self.assertTrue(path.is_file(), "A runtime asset manifest must be produced")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["characters"]), 4)
        from PIL import Image
        for character in data["characters"]:
            sheet = UI / "public" / character["path"].lstrip("/")
            with Image.open(sheet) as image:
                self.assertEqual(image.size, (32 * 36, 32))
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.getchannel("A").getextrema()[0], 0)
            self.assertTrue((UI / "public" / character["avatar"].lstrip("/")).is_file())

    def test_packaged_art_is_new_and_all_manifest_files_resolve(self):
        path = UI / "public/assets/v2/manifest.json"
        self.assertTrue(path.is_file(), "Asset manifest is absent")
        data = json.loads(path.read_text(encoding="utf-8"))
        legacy = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (UI.parent / "assets").glob("*")}
        for name in data["files"]:
            p = (UI / "public" / name.lstrip("/")).resolve()
            self.assertTrue(p.is_relative_to(UI.resolve()))
            self.assertTrue(p.is_file(), str(p))
            self.assertNotIn(hashlib.sha256(p.read_bytes()).hexdigest(), legacy)

    def test_normalized_preview_and_sprite_counts(self):
        path = UI / "public/assets/v2/manifest.json"
        self.assertTrue(path.is_file(), "Normalized assets have not been produced")
        from PIL import Image
        data = json.loads(path.read_text(encoding="utf-8"))
        for group in ("terrain", "effects"):
            entry = data[group]
            with Image.open(UI / "public" / entry["path"].lstrip("/")) as image:
                self.assertEqual(image.size, (32 * entry["frames"], 32))
        self.assertTrue((UI / "assets/previews/characters.png").is_file())


if __name__ == "__main__":
    unittest.main()
