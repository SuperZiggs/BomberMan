# Asset provenance and production prompts

All runtime game images are new assets generated for this task using the built-in **ImageGen** tool, not an API-key CLI. Runtime SVG logo/icons and 8 WAV effects are original code-authored assets in `tools/prepare_assets.py`. No legacy game PNG or Emulogic font is loaded by the web app.

Source images: `assets/source/`. Packaged outputs: `public/assets/v2/`. Whole-sheet sources remain outside the production build. Deterministic extraction and normalization uses the unmodified Game Studio 0.1.2 `normalize_sprite_strip.py`; other bundled pipeline helpers are preserved under `tools/sprite-pipeline/` (plugin manifest license: MIT). Previews: `assets/previews/characters.png`.

## Seed prompt

Production seed frame for an original 2D top-down pixel-art Bomberman robot-foundry game, not the existing Bomberman character. A single tiny mint-green industrial courier robot, squat compact rounded-square body, two dark stubby feet, small squared arms, charcoal face visor with two bright cream eyes, a small antenna and a distinct white triangle badge on its forehead. Friendly but sturdy, readable 32×32-pixel aesthetic with crisp hard pixel clusters, limited mint teal/cream/dark navy, top-left lighting, slightly top-down facing down. Full body, centered, generous transparent space; no floor, cast shadow, words, props, scenery or watermark. Identity reference for animation.

Generation output ID: `exec-0119bd5a-c3db-4fc1-80de-45fe40504c6a`, saved as `robot-seed.png`.

## Character prompt set

Use the seed as identity reference. One entire sheet per character, consistent same character/proportions/colors throughout 36 frames. Exact 6×6 equal square slots, transparent background, no floor, shadows, grid lines, labels or words, centered full-body sprites and transparent gutters. Crisp 32-pixel-style clusters, shared scale and bottom-center feet anchor. Row1: 2 down-facing idle, 4 down-facing walk. Row2: 2 up-facing idle, 4 up-facing walk. Row3: 2 left-facing idle, 4 left-facing walk. Row4: 2 right-facing idle, 4 right-facing walk. Row5: 2 down placement, 2 up placement, 2 left placement. Row6: 2 right placement then 4 defeat frames collapsing into smoke. No bomb prop in poses. Candidate production sprite sheet, not concept art; no frame overlaps or cropped character.

| File | Variant specification | Generation output ID |
|---|---|---|
| robot-mint.png | Exact mint seed, triangle forehead badge, compact round-square helmet, single antenna | exec-c67463d3-0b64-43f0-ace0-aebbdc2023d4 |
| robot-blue.png | Cobalt-blue sibling, white square badge, angular helmet, two small side fins | exec-fb6b39d9-275c-4fd8-92b4-07853889cb3b |
| robot-coral.png | Coral-red sibling, white circle badge, round helmet, side ear bolts | exec-d1f2da77-c756-449a-b70d-36316aaeb242 |
| robot-gold.png | Golden sibling, white diamond badge, hexagonal helmet, short double antenna | exec-6c555024-74e5-404d-81bf-ee823d977ecf |

Ruling: use one 6×6 sheet per character instead of a 36-slot-wide image so whole-sheet identity remains coherent at the image tool's supported canvas proportions. The sheet is unpacked as complete animation groups, then normalized with one shared scale; no independently generated frame is spliced in. 144 normalized character frames total. The seed was visually inspected before whole-sheet generation; all 4 sheets and the in-game screenshots were inspected afterward.

## Terrain prompt

Production tile atlas for a top-down pixel-art robot foundry. Exact 4×3 square tiles edge-to-edge, no gaps/margins/labels. Crisp 32-pixel clusters, industrial navy/blue-gray palette, upper-left highlights. Row1: four quiet walkable metal floors with central plates and small rivets. Row2: three bright beveled unbreakable metallic wall blocks and one tan destructible crate with orange diagonal brace. Row3: alternate darker crate and three corner/edge perimeter metal tiles with amber safety stripes. Floor darker/quieter than obstacles. Square footprint; no perspective convergence, characters, bombs or floating objects.

Output: `exec-fc9b8384-8ddb-46c3-9dfd-1ac9e457dbf5`, `terrain.png`; 12 normalized tiles. Only engine grid codes determine passability.

## Effects prompt

Production 8-column × 9-row transparent pixel-art atlas, no labels/grid. Each consecutive 4 slots is one coherent animation. Row1 mint/blue bomb fuse; row2 coral/gold bomb fuse; row3 ammo cartridge with plus/range outward ray; row4 explosion center/horizontal flame; row5 vertical flame/north end; row6 east end/south end; row7 west end/crate break; row8 defeat smoke/pickup sparkle; row9 amber warning pulse/4 empty slots. Crisp 32-pixel clusters, generous transparent padding, identical scale per animation. Bright amber/orange/cream explosions, no scene/ground/shadow. Effects expand/fade inside each slot.

Output: `exec-609da5b2-9968-4e55-b4a5-ff2a62beb9ea`, `effects.png`; 68 nonempty normalized frames. Flame lifetimes and ring warnings still follow Python state.

## Menu prompt

Wide 16:9 original pixel-art robot-foundry menu background. Dark industrial workshop at night, tiny illuminated arena on right half from elevated top-down angle, cute mint/golden robots, slate tiles, metal blocks, wood crates and small orange explosion. Warm amber overhead lamps, mint accents, quiet blue shadows, restrained pixel dust. Left half mostly empty very dark navy industrial wall for overlaid title. Detail concentrated on right. No text, logos, buttons, fake HUD or watermark; actual background plate, not UI mockup.

Output: `exec-4ebbdd56-35f2-4264-a036-4d78bd413efe`, `menu.png` → packaged `ui/foundry.webp`.

## Font and audio

Font: Be Vietnam Pro Regular/SemiBold/ExtraBold from [Google Fonts repository](https://github.com/google/fonts/tree/main/ofl/bevietnampro), self-hosted with `fonts/OFL.txt`. This replaces the old runtime font; it is not claimed as original font artwork.

Original synthesized SFX: bomb (130→90Hz), fuse (680→900Hz), explosion (105→65→32Hz + seeded noise), pickup (440→660→880Hz), defeat (360→260→130Hz), win (392→494→587→784Hz), draw (330→294→262Hz), click (560Hz). Mono 22,050Hz, 16-bit PCM with short amplitude envelopes; code reproducible in `prepare_assets.py`.
