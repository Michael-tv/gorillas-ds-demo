# Session: Real drag physics, empirical sprite-scale audit, gameplay metrics, resizable window

**Date:** 2026-09-05
**Goal:** Replace the original game's no-drag "wind is a constant sideways push" model with real
aerodynamic drag physics, figure out what real-world scale the game's own art implies (and validate
it empirically, not just by reading source), add the wind/distance/seed metrics from the earlier
`dosbox-modified` session onto the new physics build, and fix DOSBox's window so it can be resized
and maximized.

## Outcome (what to use)

Two new folders, alongside the two from the earlier session (`dosbox\`, `dosbox-modified\`):

- **`dosbox-modified-physics\`** — the byte-authentic original, with `PlotShot` rewritten to
  integrate real drag numerically. No UI/gameplay mods. Use this to see drag physics in isolation.
- **`dosbox-modified-physics-metrics\`** — the same physics, *plus* `dosbox-modified`'s three
  features (numeric wind speed, gorilla-to-gorilla distance, playing-field seed) *plus* a new
  **"Last Throw Distance to Target"** readout (horizontal distance from where the banana landed to
  the target gorilla, measured at the target's own floor level). **This is the one to actually
  play.**
- **`scale-reference\`** — screenshots proving the sprite pixel measurements below (not estimates).
- All four `dosbox*` folders now launch via **DOSBox Staging**, not mainline DOSBox 0.74 (see
  "Resizable window" below). Each folder is still fully self-contained.

Published Artifact (private, viewable by the user): **[Sprite Scale
Audit](https://claude.ai/code/artifact/993d9f9f-aea4-4f00-9253-1a4cb9ffab06)** — visual reference for
the meters-per-pixel calibration and what it implies for the gorilla/building/banana. Update it in
place (same URL) rather than re-publishing if these numbers ever change again.

## The physics rewrite (`dosbox-modified-physics\gorilla.bas`)

The original formula (`x# = ... + InitXVel#*t# + .5*(Wind/5)*t#^2`, same shape for `y#`) is a plain
constant-acceleration (SUVAT) closed form — wind is a fixed sideways acceleration, fully decoupled
from gravity or the shot's own speed, and range scales as *exactly* `V²` with no ceiling. Real drag
(`F = ½ρv²·Cd·A`, opposing the velocity vector) has no closed-form solution, so `PlotShot` now
integrates velocity/position incrementally (semi-implicit Euler) inside the existing per-tick loop.

Constants added near `gravity#`/`Wind`: `AirDensity# = 1.225`, `BananaMass# = .15` (real banana,
150g), `BananaCd# = .6`, `BananaArea#` from a 4cm diameter, and `MetersPerPixel# = .2` — calibrated
from the city-scape's own `WDifV = 15px` window-row spacing assuming a 3m real story height. `Wind`
is now a real m/s value that enters through the *relative* velocity used in the drag term (a
tailwind reduces relative airspeed and thus drag; a headwind increases it), not a bolted-on additive
term like the original. `Velocity` typed by the player is now literal m/s, no fudge factor.

Verified (numerically, not just by inspection) that drag acts correctly on **both** axes: a 40 m/s
45° throw's apex is 17% lower than the no-drag parabola predicts (proves vertical drag is real, not
just horizontal), a pure 90° throw has zero spurious horizontal drift (no axis cross-talk), and
wind coupling responds in the physically correct direction once you're careful about which sign is
tailwind vs headwind (`relVx# = vx# - Wind`).

## The empirical sprite-scale audit — don't trust the byte-budget guess

Early in the session, banana sprite size was *estimated* from its `GET`/`PUT` `DATA` storage budget
(9 LONGs = 36 bytes/frame ⇒ guessed ~8–16px). **This was wrong and the user was right to push back
on it twice.** The packed EGA bitmap header can't be reliably hand-decoded from the raw `DATA`
values — don't try again from source alone.

What actually worked: write a standalone QBasic program that `PUT`s the real sprite (or calls the
real `DrawGorilla` / window-drawing code directly), draws a pixel ruler next to it, runs it in
DOSBox, screenshots the window, then color-scans the PNG for exact bounding boxes. Cross-validated
against the building windows (which *are* exactly known from source, `WWidth/WHeight/WDifV`) —
the re-render matched those source constants to the pixel, confirming the method.

**Confirmed measurements** (see `scale-reference\`):
- Banana: **4×7px** (Left/Right frame) and **9×4px** (Down/Up frame) — both share a 4px short axis.
  At 0.2 m/px: ~0.8m diameter, 1.4–1.8m long. (Not the earlier 8–16px guess.)
- Gorilla: actual drawn silhouette is **23×26px**, not its 30×30px `GET` capture box (the box has
  slack around the arms-down pose). ⇒ 5.2m tall, not 6.0m. ~3× oversized vs a real ~1.7m gorilla.
- Building floor spacing: re-confirmed `WDifV=15px` exactly via re-render (this is the calibration
  anchor itself, not an independent check, but good to have visually confirmed anyway).

**Giant-banana mass — the right way to scale it:** if you blow the banana up and want a *physically
sensible* object (not the pixel art's own stubby proportions taken literally), anchor the scale
factor to length and keep the real banana's proportions, so mass scales as (linear factor)³, not
linearly and not via the sprite's own fatter cross-section. At the measured 1.8m length (10× a real
18cm banana), that's a clean **120 kg** (10³ × 120g) — matches the user's own back-of-envelope math
exactly. Two earlier numbers were floated and superseded, kept here so they don't get reintroduced:
9,600–76,800 kg (from treating the sprite's raw pixel size as *diameter*) and 600 kg (from taking
the sprite's own measured — stubbier-than-real — proportions literally). Both are answers to subtly
different questions; 120 kg is the one for "realistically-shaped giant banana."

## Resizable window: mainline DOSBox 0.74 can't do it — switched to DOSBox Staging

Checked directly via Win32 `GetWindowLong` style bits: the mainline DOSBox 0.74 window has no
`WS_MAXIMIZEBOX` and no `WS_THICKFRAME` in *any* output mode (`surface`, `openglnb`, etc.) — this is
a real limitation of that build, not a config problem. Alt+Enter fullscreen toggle does work in
mainline (confirmed by the user), but windowed drag-resize/maximize does not.

Installed **DOSBox Staging** (`winget install --id DOSBoxStaging.DOSBoxStaging`, lands at
`C:\Users\MichaelVictor\AppData\Local\Programs\DOSBox Staging\dosbox.exe`) and switched all four
`dosbox*` folders' `.bat`/`.conf` files to use it. Confirmed both via style bits
(`MaximizeBox=True`, `ResizeBorder=True`) and via the runtime log showing it correctly live-rescaling
DOS output as the window was resized (933×700 → 1345×1009), no errors. Mainline DOSBox 0.74 is left
installed, untouched, as a fallback — just point a `.bat` back at
`C:\Program Files (x86)\DOSBox-0.74-3\DOSBox.exe` to revert any one folder.

**Staging's config schema is different from mainline's** — don't copy mainline `.conf` syntax
verbatim:
- `output=surface` → `output=texturenb` (crisp/no-bilinear; `opengl` is Staging's default but
  applies an adaptive CRT shader that changes the look, so avoided here to keep the original crisp
  pixel-art appearance).
- `fullscreen=false`/`true` → `fullscreen=off`/`on`.
- `[render] aspect=true` → `[render] aspect=auto`.
- `[cpu] cycles=auto` → `[cpu] cpu_cycles=3000` (numeric or `max` only — no `auto` value for
  `cpu_cycles`; 3000 was mainline's own auto-detected value on this machine, confirmed from every
  mainline window title seen this session, so used as a like-for-like default).
- Added `window_size=1280x700` and `window_decorations=on` (the latter is Staging's default anyway,
  set explicitly for clarity).
- Staging auto-generates a full commented reference config on first run at
  `C:\Users\MichaelVictor\AppData\Local\DOSBox\dosbox-staging.conf` — read that directly if more
  options need tuning later; don't guess key names from memory of mainline DOSBox.

## Environment / automation gotchas (for next time)

- **QBasic requires `CONST` before any executable statement** at module level. Put a `CONST` after
  `SCREEN 9`/an assignment and `/RUN` silently errors and drops back to the DOS prompt almost
  instantly — looks like nothing happened unless you're watching the window title cycle
  `DOSBOX → QBASIC → DOSBOX` fast.
- **DOS 8.3 filenames**: any new throwaway `.bas`/`.conf` test file needs a ≤8-character base name
  (`bansize.bas` works, `sizetest2.bas` doesn't — "Bad file name").
- **DOSBox opens two windows** (the game render window and a "DOSBox Status Window"); when
  automating screenshots, filter by title prefix `DOSBox` *excluding* "Status Window", and when
  multiple DOSBox processes are alive at once, filter by the specific PID via
  `GetWindowThreadProcessId` too — title-only matching can grab a stale window from an earlier
  launch.
- **PowerShell tool calls don't share state** — any `Add-Type`-defined C# helper class must be
  redefined in full in every single call that uses it; it does not persist to the next call.
- The user sometimes has their own DOSBox window open and testing things directly — don't
  `Stop-Process` on DOSBox without being sure which PID you just launched yourself; asked once and
  was told no, because a different window was theirs.
