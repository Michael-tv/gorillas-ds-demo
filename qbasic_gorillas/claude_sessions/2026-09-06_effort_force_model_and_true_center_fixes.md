# Session: Force/Effort throw model, true-center distance fixes, player-relative wind, meters display

**Date:** 2026-09-06
**Scope:** `dosbox-modified-physics-metrics\gorilla.bas` only, unless noted otherwise. Continues
directly from the 2026-09-05 physics-rewrite session.

**Goal:** Fix a runtime crash, then iteratively verify the drag-physics equations by hand-testing
specific throws (given distance/height/wind/angle, solve for the velocity/effort needed) — which
surfaced several real display/geometry bugs along the way — then replace the raw "Velocity" input
with a physically-motivated "Effort%" control.

## Outcome (what to use)

Same four folders as before; only `dosbox-modified-physics-metrics\gorilla.bas` got the bulk of
this session's changes. All four `dosbox*\gorilla.bas` got the crash fix (item 1 below).

Play the game the same way as before — `dosbox-modified-physics-metrics\` is still the one to
actually play. The on-screen prompt is now `Effort%:` (0-100) instead of `Velocity:`, and the
`Distance X/Y` and `Last Throw Distance to Target` readouts are in meters instead of raw pixels.

## 1. Crash fix: `Tossee = 3 - J` → `Tossee = 2 - J` (all four folders)

`J` alternates 0/1. `Tosser = J + 1` correctly alternates 1/2, but `Tossee = 3 - J` alternated 3/2 —
index **3** is out of range for `GorillaX`/`GorillaY`, both `DIM SHARED ...(1 TO 2)`. Only
`dosbox-modified-physics-metrics` actually dereferenced `GorillaX(Tossee)` (for the miss-distance
metric), so the other three folders had the same wrong formula silently dead until now.

## 2. Meters display, not raw pixels

`Distance X/Y` and `Last Throw Distance to Target` now multiply by `MetersPerPixel# = .2` and format
to one decimal place via a new `Fmt1$(n#)` helper (sign-aware, built with `CLNG`/`\`/`MOD` integer
math since this QBasic dialect has no `FORMAT$`/`PRINT USING`-to-string).

## 3. The real bug under "why did my calculated velocity overshoot?"

Hand-testing the drag equations against actual in-game throws repeatedly produced results that
didn't match a Python re-implementation of `PlotShot`'s exact integration (semi-implicit Euler,
`dt#=.1`, same drag/gravity formulas) — until tracing it down to **the on-screen distance numbers
themselves being wrong**, not the physics:

- `GorillaX(i)`/`GorillaY(i)` are the sprite's `PUT` anchor (top-left-ish), not its visual center.
- The banana's actual launch point (`PlotShot`'s `StartXPos`/`StartYPos`) is **not** simply
  `GorillaX(Tosser)`/`GorillaY(Tosser)` either — Player 2 gets an extra `+ Scl(25)` on X, and Y is
  always `GorillaY(Tosser) - Scl(4) - 3` for both players.
- The real hit-target center is `GorillaCenterX/Y`, matching the same offset `ExplodeGorilla` uses
  to animate a hit (`+XAdj` for X; `+ 7*SclY# + Scl(12)` for Y) — not the raw anchor.

Net effect on one real test case: displayed 78.1m/+2.3m, true launch-to-target-center was actually
**80.9m / -3.95m** (target center below the launch point, not above). A velocity solved against the
displayed numbers naturally overshot by exactly the observed amount.

**Fix:** added `GorillaCenterX(1 TO 2)` and `GorillaCenterY(1 TO 2)` (computed once in
`PlaceGorillas`), and rewrote `Distance X/Y` in `PlayGame`'s per-throw loop to measure from the
tosser's true launch point (replicating `PlotShot`'s `StartXPos`/`StartYPos` formula, including the
Player-2-only `+Scl(25)`) to `GorillaCenterX(Tossee)`/`GorillaCenterY(Tossee)`. `MissDist#` already
used `GorillaCenterX(Tossee)` from an earlier pass, so it only needed the meters conversion.

**Lesson for next time:** when a physics/geometry claim doesn't match observed behavior, check
whether the *displayed reference numbers* are measuring the same thing the engine actually uses
before assuming the integration itself is wrong.

## 4. Signed Y, player-relative wind

- `Distance X/Y`'s Y is signed: negative when the tosser is above the target, positive when below.
  Must be recomputed every throw (not once per round in `PlaceGorillas`) since `Tosser`/`Tossee`
  swap sides each turn.
- `Wind Speed:` is now shown **relative to whoever's currently throwing**: positive = blowing away
  from the tosser (tailwind), negative = blowing toward them (headwind), with `m/s` appended. Player
  1 throws toward +x so the real `Wind` sign is used as-is; Player 2 throws toward -x so it's negated
  (`WindRelative = -Wind`). This matters for hand-solving shots — the number on screen can now be fed
  straight into the drag equation without manually mirroring it for whichever player is up.

## 5. Force/Effort% throw model (replaces raw "Velocity")

The player-facing prompt is now `Effort%:` (0-100), not a raw m/s `Velocity:`. Physical model, per
the user's own spec: the gorilla (a ~5m ape) accelerates the banana over its arm-plus-stroke reach —
`ThrowStrokeM# = 4` (2m arm + 2m stroke) — via constant force, so by work-energy:

```
Force# = (Effort# / 100) * MaxGorillaForce#
Velocity = SQR(2 * Force# * ThrowStrokeM# / BananaMass#)     ' = MaxThrowVelocity# * SQR(Effort#/100)
```

**Calibrating `MaxThrowVelocity#` (100% effort) — don't guess, simulate real rounds:** first attempt
(70 m/s) was calibrated by searching the *optimal* angle across hand-picked extreme scenarios
(max screen distance/height/headwind), which missed that a **forced** steep angle (an obstruction
close to the tosser, still needing to reach a far target) can demand much more velocity at a fixed
angle than at the optimal one. Simulating that directly — full `MakeCityScape`/`PlaceGorillas`
random generation, both throw directions, computing the minimum angle needed to clear every
intervening building (straight-line-of-sight heuristic from the near edge), then checking
feasibility from that angle up to 85° — across 16,000 real generated shot-scenarios found **zero**
actually infeasible ones, because a building tall enough to force a steep angle is geometrically
close to the tosser, and the same round's total gorilla-to-gorilla distance is capped by the fixed
screen width — the adversarial "steep angle AND very long remaining distance" combination the naive
worst-case search worried about essentially doesn't co-occur in the real generator.

That same 16,000-sample run gave the actual velocity-needed distribution (unconstrained best-angle
search, respecting real obstructions): median 32.4 m/s, p90 37.9, p95 39.7, p99 43.8, **p99.9 53.4**,
max 73.8 (a single 1-in-16,000 case, 98.8m/67.6°-forced/-13 wind — already technically over the old
70 cap). Landed on **`MaxThrowVelocity# = 60`** (`MaxGorillaForce#` recomputes to 67.5N) — covers
p99.9 with real margin, sacrifices only that same already-borderline extreme tail, and gives a much
more usable 0-100 slider (median shot ≈29% effort instead of ≈46%).

**Gotcha:** QBasic's `CONST` expressions do not support the `^` (exponentiation) operator —
`CONST MaxGorillaForce# = MaxThrowVelocity# ^ 2 * ...` throws "Illegal function call" at runtime.
Use explicit multiplication (`MaxThrowVelocity# * MaxThrowVelocity#`) instead. `CONST` does support
simple arithmetic (`+ - * /`) and boolean operators (`NOT`, used elsewhere for `CONST FALSE = NOT
TRUE`) referencing earlier `CONST`s — just not `^`.

**Conversion for hand-testing:** `Effort% = 100 * (V / 60)²`.

## Workflow note: solving the inverse physics problem

Throughout this session, "given distance/height/wind/angle, what velocity/effort works" was solved
by a standalone Python re-implementation of `PlotShot`'s exact drag integration (same `dt#=.1`
semi-implicit Euler, same `DragK#`/`gravity#` formulas), then binary-searching velocity so the
trajectory's height matches the target at the target's x-distance. Scratch scripts were written to
the project root or a temp dir and deleted after use — not committed anywhere. If this needs doing
again, replicate `PlotShot` lines ~959-1030 exactly rather than reaching for a closed-form
projectile formula (drag has no closed form here).
