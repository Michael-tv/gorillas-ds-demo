# Session: New variant with triangular-distributed banana weight/size and effort variance

**Date:** 2026-09-06
**Scope:** New folder `dosbox-modified-physics-metrics-var\`, copied from
`dosbox-modified-physics-metrics\` (see the 2026-09-06 effort/force-model session for that
baseline). No other folder touched.

## Outcome (what to use)

`dosbox-modified-physics-metrics-var\` — same effort/force throw model and metrics readouts as
`dosbox-modified-physics-metrics\`, plus two sources of shot-to-shot randomness, both drawn from a
**triangular distribution** (values cluster near a "mode"/typical value, tapering linearly to zero
probability at a min/max — unlike a flat `RND` range where every value is equally likely):

1. **Banana weight and size vary per throw.** `BananaMassMin#/Mode#/Max# = .1/.15/.2` kg and
   `BananaDiamMin#/Mode#/Max# = .03/.04/.05` m. Sampled fresh in `DoShot` into the now-`DIM SHARED`
   (previously `CONST`) `BananaMass#`/`BananaDiam#`/`BananaArea#`, which `PlotShot`'s drag
   calculation (`DragK#`) and the effort-to-velocity conversion both read. `MaxGorillaForce#` stays
   a fixed `CONST` calibrated against the *typical* (Mode) mass, not the per-throw sampled one —
   it's a property of the gorilla's arm, not of whichever banana it happens to be holding.
2. **The gorilla can't apply exactly the intended effort every time.** `EffortVarianceMax# = 8`
   (percentage points). The player's typed `Effort%` becomes the triangular distribution's mode;
   actual applied effort is sampled within ±8 points of it (clamped ≥0 the same way the pre-existing
   `Force# < 0` clamp already worked), then that actual value — not the typed one — drives
   `Force#`/`Velocity`.

New `FUNCTION TriRand# (Lo#, Mode#, Hi#)` (standard inverse-CDF sampling) added near
`Fmt1$`/`GetNum#`. No on-screen display of the sampled values was added — deliberately kept to the
mechanic itself, not new UI, per the scope of what was asked.

## Verification approach — GUI keystroke injection didn't work in this sandbox, used a probe instead

Tried to drive the actual game through DOSBox via synthetic input (`SendKeys`, `keybd_event`,
`SendInput` with hardware scancodes) to reach a real throw and exercise the new code path. All
keyboard injection silently failed to reach the DOS keyboard buffer — the intro's "press any key"
`SparklePause` never advanced despite the window being genuinely foreground (`GetForegroundWindow`
confirmed) — even though synthetic **mouse** clicks did register (toggled DOSBox's own "mouse
captured" title-bar state). Root cause not fully diagnosed (candidates: UIPI blocking synthetic
`SendInput` keyboard events specifically, or DOSBox Staging's raw-input keyboard path filtering
injected events) and not worth chasing further for a smoke test — **note for next time: don't
assume keystroke injection into this DOSBox window works; mouse injection does.**

Instead, used the same principle as the earlier sprite-measurement session (render + observe,
don't guess): wrote a standalone `PROBE.BAS` (temporary, deleted after use — DOS 8.3 name) that
copied the exact new `CONST`/`DIM SHARED`/`TriRand#` block verbatim and looped calling it with
`PRINT`, no `INPUT`/`INKEY$` waits, so `QBASIC.EXE /RUN PROBE.BAS` runs to completion unattended and
returns to a screen you can just screenshot — sidesteps the input-injection problem entirely for
anything that doesn't need live interaction. Confirmed: `MaxGorillaForce# = 67.5` (matches hand
calc), 8 sampled masses/diameters all within bounds and clustering near the mode, effort samples
within ±8 of the 60% requested, the `Effort#=0` edge case clamps to exactly 0, and the whole probe
printed "PROBE DONE - NO ERRORS" — i.e. the new syntax (function definition, the `CONST` block
ordering, the arithmetic) is valid QBasic and the formulas behave as intended. Separately confirmed
the full `gorilla.bas` itself still parses and starts running correctly (reached the animated intro
screen normally) before the input-injection wall was hit — a compile-time error anywhere in the
module would have shown immediately rather than letting the intro render/animate.

**Residual, accepted risk:** the exact splice point inside `DoShot` (lines between reading `Effort#`
and computing `Velocity`) was verified by direct code inspection against the probe's already-tested
logic, not by an actual live throw — full interactive playtesting of a real round is still worth
doing by hand before relying on this build for real play.

## Follow-up, same session: moved to a new `-var-display` folder; `-var` reverted

The user asked for a version showing the sampled banana weight/area on screen, explicitly kept
**separate** from `dosbox-modified-physics-metrics-var\`. Then, mid-implementation, corrected the
plan further: `dosbox-modified-physics-metrics-var\` should **not** contain the triangular-distribution
logic at all.

Net result:
- **`dosbox-modified-physics-metrics-var\gorilla.bas`** was reverted byte-for-byte to
  `dosbox-modified-physics-metrics\gorilla.bas` (`diff` confirmed identical) — it no longer has
  `TriRand#`, the `BananaMass#`/`BananaDiam#`/`BananaArea#` variance, or the effort jitter. Only its
  `play.conf`/`edit.conf` mount paths still point at its own folder, so it remains a distinct,
  functional build, just currently equivalent in behavior to the plain metrics build. **Its purpose
  going forward is not yet defined by the user** — don't assume it's dead weight, but also don't
  assume "var" still means "variance" for this specific folder.
- **New `dosbox-modified-physics-metrics-var-display\`** (copied from the pre-revert `-var`, so it
  kept the triangular-distribution logic) is now the sole carrier of both: the `TriRand#`-based
  banana weight/size and effort variance, *and* a new persistent HUD line —
  `Center 20, "Banana Weight: " + ... + "g  Area: " + ... + "cm2"` — added right after "Last Throw
  Distance to Target" (row 21) in `PlayGame`'s per-throw loop, reading `BananaMass#`/`BananaArea#`
  directly (they already hold the just-thrown values; `DoShot` only resamples them on the *next*
  call, so no separate `Last*` variables were needed). Verified via the same unattended-probe
  technique as above (weight in grams, area in cm², both correct and in-range).

**Lesson:** when a user says "build X, but separate from Y," don't assume the shared ancestry means
shared logic stays duplicated forever — a follow-up correction removing that logic from the
original folder is a live possibility, especially once a second folder actually needs the logic
that "separate" implied should move, not just copy.
