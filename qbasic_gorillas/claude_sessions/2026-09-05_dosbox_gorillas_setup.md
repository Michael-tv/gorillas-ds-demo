# Session: QBasic Gorillas — local authentic setup + mods

**Date:** 2026-09-05
**Goal:** Get a locally-runnable copy of the classic QBasic "Gorillas" game that matches the original exactly, then add: (1) numeric wind speed, (2) horizontal/vertical distance between gorillas, (3) a "Force" throw input instead of "Velocity", and (4) a way to specify/recall a specific playing field setup.

## Outcome (what to use)

Two folders, both in `C:\Projects\qbasic_gorillas\`:

- **`dosbox\`** — the genuine, byte-unmodified 1990 Microsoft `gorilla.bas`, run via the real MS-DOS `QBASIC.EXE` interpreter under DOSBox. No compiler, no rewrite — this is the actual original.
- **`dosbox-modified\`** — same base, with the four changes below.

Each folder is self-contained: `QBASIC.EXE`, `qbasic.hlp`, `gorilla.bas`, `play.conf`/`edit.conf`, and double-clickable **`Play Gorillas.bat`** (auto-runs the game) / **`Edit Gorillas.bat`** (opens the real QBasic IDE — press Shift+F5 to run after editing).

All other attempts were deleted per user request (see "Rejected approaches" below) — do not recreate them without reason.

## Rejected approaches (tried, abandoned — don't redo without reason)

1. **QB64PE compile of the qb64.com-adapted `GORILLA.BAS`.** Hit three real, distinct bugs on this machine:
   - Deprecated `$NoPrefix` metacommand (QB64PE 4.6 hard-errors on it) — fixed by removing it and using `_FullScreen`/`_SquarePixels`/`_Smooth`/`_Delay` explicitly. (`Timer(x)` with a fractional arg stayed bare, no underscore — `_Timer` doesn't exist as such.)
   - Blocking `PLAY` statements (missing the `MB` background flag) hang **forever** on this machine — fixed by prefixing every blocking `PLAY "..."` with `MB`.
   - A `GET`/`PUT` sprite-capture bug in `GorillaIntro`: three `DrawGorilla` poses are drawn back-to-back with only `CLS 2` between them, which doesn't reliably clear the graphics under QB64PE in Mode 9, so the arm/leg sprites captured into `GorD&`/`GorL&`/`GorR&` come out visually wrong (confirmed via user screenshots — the fix I applied, an explicit `LINE ...,BACKATTR,BF` erase-box between poses, did **not** resolve it). This bug is why the QB64PE route was abandoned — not fully root-caused.
2. **JS/HTML5 canvas port** (rebuilt from `github.com/michaeldeol/gorillas-js`, RequireJS stripped into one self-contained `index.html`). Fully functional, all 3 mods applied and verified via headless Chrome screenshots — but user rejected it: "significantly different in feel" from the original (different rendering technique, not the actual original code).
3. **`SWO-GS/gorillas` fork** ("GorillaStack", heavily expanded QB64-targeted fork with a `Gorillas.ini` settings file and a sound toggle). Compiled successfully after the same `DEF FN`→`FUNCTION` and `PLAY`→`MB` fixes, and it has the *exact same* Mode-9 sprite bug (their own code comment: "For some reason, the above CLS 2s don't work in CGA" — they only patched CGA, not EGA/VGA). Rejected by user: "looks significantly different to the original" (heavily reskinned setup screens, different UI).

## What actually made DOSBox work

- Installed **DOSBox 0.74-3** via `winget install --id DOSBox.DOSBox`.
- Sourced the genuine **`QBASIC.EXE`** from archive.org item `msdos-622` (`dos-622-disk1/QBASIC.EXE`, 194309 bytes).
- Sourced the genuine, byte-authentic **`gorilla.bas`** (1996-dated, untouched, single-line `DEF FnRan` intact) by downloading archive.org item `ms-dos-6.22-with-turbo-basic-1.0-qbasic-nibbles.bas-and-gorilla.bas-1.44-m` — a raw `.IMA` FAT12 floppy image — and reading it directly with **7-Zip** (`7z.exe l/e` opens FAT floppy images natively, no separate disk-image tool needed). Also pulled `qbasic.hlp` from the same image.
- `dosbox.conf`'s `[autoexec]` section does `mount c <folder>` + `c:` + `QBASIC.EXE [/RUN] GORILLA.BAS` — passing the run command via `-c` on the DOSBox command line was unreliable; putting it in the conf's autoexec was solid.
- **Gotcha for any future automated testing/driving of this app:** DOSBox 0.74 opens *two* windows — a "DOSBox Status Window" console and the actual game render window. `Process.MainWindowHandle` sometimes grabs the wrong one. Any driver script must `EnumWindows` and filter for a title that starts with `"DOSBox"` and does **not** contain `"Status Window"`.

## The four modifications (all in `dosbox-modified\gorilla.bas`)

1. **Wind Speed**: `SUB MakeCityScape`, right after the existing wind-arrow drawing block — `Center 24, "Wind Speed: " + LTRIM$(STR$(Wind))`.
2. **Distance**: `SUB PlaceGorillas`, right after both gorillas' `GorillaX`/`GorillaY` are set — `Center 22, "Distance  X:" + ... + "  Y:" + ...`.
3. **Force instead of Velocity**: `FUNCTION DoShot` — prompt changed to `"Force:"`, reads into `Force`, then `Velocity = Force / BanannaMass`. `CONST BanannaMass = 1` near the top (renamed from `GorillaMass` per user's explicit correction — it's the thrown banana's mass, not the gorilla's). At `BanannaMass = 1` this is a no-op vs. the original; raising it means more Force is needed for the same throw speed.
4. **Playing field seed** (added after the 3 requested mods, for "recall a specific setup"): `GetInputs` gained a `SeedVal#` out-parameter and a new prompt, `"Playing field seed (blank = random)?"`. `PlayGame` gained a matching `SeedVal#` parameter; the `RANDOMIZE (TIMER)` that used to be **inside** the per-round loop was moved to **once, before** the loop (`RANDOMIZE (SeedVal#)` if nonzero, else `RANDOMIZE (TIMER)`). This means a given seed reproduces the *entire session's* sequence of city skylines/wind/gorilla placements (rounds still vary from each other within a session, but the whole sequence repeats if you reuse the seed) — verified pixel-identical across two runs with the same seed, and different with a different seed.

All four were verified end-to-end by scripted-keystroke-driving DOSBox and screenshotting (PowerShell + `SendKeys` + `EnumWindows`), not just by code review.

## Environment notes

- Installed system-wide via winget on this machine (not project-local, left in place): **7-Zip** and **DOSBox 0.74-3**. User was told these remain unless they ask to uninstall.
- This is the user's actual working machine (VSCode extension environment), not an isolated sandbox — automated UI-driving scripts steal real keyboard focus on their live screen, which caused several flaky test runs. Prefer asking the user to look/test directly over more automated driving when it starts misbehaving.
