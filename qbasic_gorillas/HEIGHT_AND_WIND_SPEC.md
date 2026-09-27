# Building Height & Wind Speed Ranges

Derived from `MakeCityScape` in `gorilla.bas` (identical constants across
`dosbox`, `dosbox-modified`, `dosbox-modified-physics`, and
`dosbox-modified-physics-metrics`). `FnRan(x) = INT(RND(1) * x) + 1`, an
integer drawn uniformly from `[1, x]`.

## Wind speed

```
Wind = FnRan(10) - 5                     ' base: -4 .. 5
IF FnRan(3) = 1 THEN                     ' 1-in-3 chance of a gust
  IF Wind > 0 THEN Wind = Wind + FnRan(10)   ' +1 .. +10
  ELSE Wind = Wind - FnRan(10)                ' -1 .. -10 (also applies when Wind = 0)
END IF
```

| Case | Range |
|---|---|
| No gust (2/3 chance) | -4 to 5 |
| Gust, base wind > 0 | 2 to 15 |
| Gust, base wind <= 0 | -14 to -1 |
| **Overall** | **-14 to 15** |

Computed once per round in `MakeCityScape`, mode-independent (same formula
regardless of screen mode). Confirmed by exact enumeration and by a
2,000,000-sample simulation.

## Building height (`BHeight`)

Each building's height is a random walk seeded by a "slope" trend, then
jittered per building:

```
BHeight = FnRan(RandomHeight) + NewHt
IF BHeight < HtInc THEN BHeight = HtInc                        ' floor
IF BottomLine - BHeight <= MaxHeight + GHeight THEN
  BHeight = MaxHeight + GHeight - 5                            ' overflow clamp
```

`NewHt` starts at 15 or 130 depending on the slope case (upward/downward/"V"/
inverted-"V"), then drifts by `HtInc` or `2*HtInc` per building as the skyline
is built left to right. `MaxHeight` is declared but never assigned anywhere
in the source, so it is always `0` — the overflow clamp is effectively
`BHeight = GHeight - 5`.

Mode-dependent constants:

| Mode | `BottomLine` | `HtInc` | `DefBWidth` | `RandomHeight` | `GHeight` |
|---|---|---|---|---|---|
| 9 (EGA/VGA, 640x350 — used in DOSBox) | 335 | 10 | 37 | 120 | 25 |
| 1 (CGA fallback, 320x200) | 190 | 6 | 18 | 54 | 12 |

Because `NewHt`'s drift depends on building position and the random slope
case, there's no closed-form min/max — the range below comes from a
2,000,000-skyline simulation of the exact walk:

| Mode | Height range (px) | Height range (m, at 0.2 m/px) |
|---|---|---|
| 9 (EGA/VGA) | 10 – 295 (up to ~310 before the overflow clamp forces it down to 20) | 2.0 – 59.0 |
| 1 (CGA) | 6 – 157 | 1.2 – 31.4 |

The meter conversion uses `MetersPerPixel# = .2` (the same constant the
banana-physics code uses for real-world scaling: one building "story" is
`WDifV` = 15px on the 640x350 reference screen, calibrated to a real 3m
story).
