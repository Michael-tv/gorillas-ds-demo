'                         Q B a s i c   G o r i l l a s
'
'                   Copyright (C) Microsoft Corporation 1990
'
' Your mission is to hit your opponent with the exploding banana
' by varying the angle and power of your throw, taking into account
' wind speed, gravity, and the city skyline.
'
' Speed of this game is determined by the constant SPEEDCONST.  If the
' program is too slow or too fast adjust the "CONST SPEEDCONST = 500" line
' below.  The larger the number the faster the game will go.
'
' To run this game, press Shift+F5.
'
' To exit QBasic, press Alt, F, X.
'
' To get help on a BASIC keyword, move the cursor to the keyword and press
' F1 or click the right mouse button.
'

'Set default data type to integer for faster game play
DEFINT A-Z

'Sub Declarations
DECLARE SUB DoSun (Mouth)
DECLARE SUB SetScreen ()
DECLARE SUB EndGame ()
DECLARE SUB Center (Row, Text$)
DECLARE SUB Intro ()
DECLARE SUB SparklePause ()
DECLARE SUB GetInputs (Player1$, Player2$, NumGames, SeedVal#)
DECLARE SUB PlayGame (Player1$, Player2$, NumGames, SeedVal#)
DECLARE SUB DoExplosion (x#, y#)
DECLARE SUB MakeCityScape (BCoor() AS ANY)
DECLARE SUB PlaceGorillas (BCoor() AS ANY)
DECLARE SUB UpdateScores (Record(), PlayerNum, Results)
DECLARE SUB DrawGorilla (x, y, arms)
DECLARE SUB GorillaIntro (Player1$, Player2$)
DECLARE SUB Rest (t#)
DECLARE SUB VictoryDance (Player)
DECLARE SUB ClearGorillas ()
DECLARE SUB DrawBan (xc#, yc#, r, bc)
DECLARE FUNCTION Scl (n!)
DECLARE FUNCTION Fmt1$ (n#)
DECLARE FUNCTION GetNum# (Row, Col)
DECLARE FUNCTION TriRand# (Lo#, Mode#, Hi#)
DECLARE FUNCTION DoShot (PlayerNum, x, y)
DECLARE FUNCTION ExplodeGorilla (x#, y#)
DECLARE FUNCTION Getn# (Row, Col)
'Velocity is Velocity# here, not the original INTEGER. DEFINT A-Z made it an
'integer, and QuickBASIC rounds on assignment, so the SQR() in DoShot could
'only ever produce whole m/s -- about 45 distinct values. That is invisible
'while playing, but this build's whole point is emitting launch speed as a
'regression target, and a 45-value target has a hard error floor. See
'claude_sessions for the reasoning. dosbox-datagen only; the six original
'folders keep the integer.
DECLARE FUNCTION PlotShot (StartX, StartY, Angle#, Velocity#, PlayerNum)
DECLARE FUNCTION CalcDelay! ()

'--- dosbox-datagen additions ---
DECLARE SUB ReadCfg ()
DECLARE SUB CsvOpenFile ()
DECLARE SUB CsvCloseFile ()
DECLARE SUB LogThrow (Tosser, Tossee, LaunchX#, LaunchY#, DistX#, DistY#, WindRel)
DECLARE FUNCTION Fmt4$ (n#)
DECLARE FUNCTION UniRand# (Lo#, Hi#)
DECLARE SUB Dbg (s$)

'Make all arrays Dynamic
'$DYNAMIC

'User-Defined TYPEs
TYPE XYPoint
  XCoor AS INTEGER
  YCoor AS INTEGER
END TYPE

'Constants
CONST SPEEDCONST = 500
CONST TRUE = -1
CONST FALSE = NOT TRUE
CONST HITSELF = 1
CONST BACKATTR = 0
CONST OBJECTCOLOR = 1
CONST WINDOWCOLOR = 14
CONST SUNATTR = 3
CONST SUNHAPPY = FALSE
CONST SUNSHOCK = TRUE
CONST RIGHTUP = 1
CONST LEFTUP = 2
CONST ARMSDOWN = 3

'Global Variables
DIM SHARED GorillaX(1 TO 2)  'Location of the two gorillas
DIM SHARED GorillaY(1 TO 2)
DIM SHARED GorillaCenterX(1 TO 2)  'Horizontal center of each gorilla sprite
DIM SHARED GorillaCenterY(1 TO 2)  'Vertical center of each gorilla sprite (impact point)
DIM SHARED LastShotX#, LastShotY#  'Where the last banana came to rest
DIM SHARED LastBuilding

DIM SHARED pi#
DIM SHARED LBan&(x), RBan&(x), UBan&(x), DBan&(x) 'Graphical picture of banana
DIM SHARED GorD&(120)        'Graphical picture of Gorilla arms down
DIM SHARED GorL&(120)        'Gorilla left arm raised
DIM SHARED GorR&(120)        'Gorilla right arm raised

DIM SHARED gravity#
DIM SHARED Wind

'Banana aerodynamics (PlotShot integrates drag numerically since drag force
'depends on velocity and has no closed-form solution). Air itself stays
'real-world (the environment isn't rescaled, only the thrown banana is - see
'BananaLengthScale# below).
CONST AirDensity# = 1.225      'kg/m^3, sea-level air density
CONST BananaCd# = .6           'drag coefficient, tumbling-average (not a sphere)
'Calibration: EGA city scape draws one building story every WDifV=15px on the
'640x350 reference screen (see MakeCityScape); a real story is ~3m tall.
CONST MetersPerPixel# = .2     'm per reference pixel, 3m story / 15px story

'This game's own art isn't real-banana-sized: the empirical sprite audit
'(scale-reference\, render+screenshot method) measured the banana sprite at
'~1.4-1.8m long at this game's MetersPerPixel# calibration, consistent with
'the ~5.2m gorillas and 3m building stories - a giant banana, not an 18cm
'real one. BananaLengthScale# is that linear scale factor (~1.8m measured /
'~.18m real), applied to diameter linearly and to mass by the cube (mass
'scales with volume for a self-similar shape). Real bananas also aren't
'uniform, so weight and size vary shot-to-shot: a triangular distribution
'(values cluster near the Mode, tapering linearly to zero probability at
'Min/Max) sampled fresh each throw - see TriRand# and its use in DoShot.
'BananaMass#/BananaDiam#/BananaArea# below hold the current throw's sampled
'values (SHARED so PlotShot's drag calc sees them too).
CONST BananaLengthScale# = 10  'giant-banana linear scale factor (see above)
CONST BananaMassMin# = .1 * BananaLengthScale# * BananaLengthScale# * BananaLengthScale#  'kg, small banana
CONST BananaMassMode# = .15 * BananaLengthScale# * BananaLengthScale# * BananaLengthScale#  'kg, typical banana
CONST BananaMassMax# = .2 * BananaLengthScale# * BananaLengthScale# * BananaLengthScale#  'kg, large banana
CONST BananaDiamMin# = .03 * BananaLengthScale#  'm, thin banana
CONST BananaDiamMode# = .04 * BananaLengthScale#  'm, typical banana
CONST BananaDiamMax# = .05 * BananaLengthScale#  'm, thick banana
DIM SHARED BananaMass#         'kg, current throw's sampled banana mass
DIM SHARED BananaDiam#         'm, current throw's sampled banana diameter
DIM SHARED BananaArea#         'm^2, frontal area computed from BananaDiam#

'Throw force model: the gorilla applies a constant force over its arm's
'reach (a ~5m gorilla's arm plus stroke, ~2m + 2m) accelerating the banana
'from rest, so v = SQR(2 * Force * ThrowStrokeM# / BananaMass#) (work-energy
'over that stroke). The player enters "Effort" as a 0-100% fraction of
'MaxGorillaForce#. Calibrated by simulating 16,000 random rounds (real
'city/wind generation, respecting building obstructions) against the giant
'banana's own drag (its Area/Mass ratio - and so DragK# - is ~10x lower than
'a real banana's, so it carries much farther per m/s): p99.9 of shots need
'<=39.2 m/s (max sampled 41.3), so 44 m/s at 100% effort covers that with
'the same ~12% margin the original real-banana calibration used, sacrificing
'only the same kind of already-borderline extreme tail. MaxGorillaForce# is
'a fixed property of the gorilla, so it's calibrated against the banana's
'typical (Mode) mass, not the per-throw sampled one.
CONST ThrowStrokeM# = 4        'm, gorilla arm + stroke length force acts over
CONST MaxThrowVelocity# = 44   'm/s, launch speed at 100% effort against a typical banana
CONST MaxGorillaForce# = MaxThrowVelocity# * MaxThrowVelocity# * BananaMassMode# / (2 * ThrowStrokeM#)  'Newtons

'A gorilla can't apply exactly the intended effort every time - a triangular
'jitter of up to +/-EffortVarianceMax# percentage points is applied to the
'player's requested Effort% fresh each throw (see DoShot).
CONST EffortVarianceMax# = 8   'percentage points, max deviation either way

'=====================================================================
'Batch data-generation controls -- dosbox-datagen build only
'
'This build can run unattended: generate one random playing field
'("board") after another, throw at it, and log every throw to a CSV,
'with nobody at the keyboard. It is still a playable game -- with
'DataMode = FALSE and the defaults below it behaves exactly like the
'dosbox-modified-physics-metrics-var-display build it was copied from.
'
'Every setting is read at startup from GORCFG.TXT on the mounted drive,
'because QBASIC.EXE /RUN gives no access to COMMAND$ -- there is no way
'to pass arguments on the command line. See SUB ReadCfg.
'
'IMPORTANT: CONST must appear before any executable statement at module
'level. Putting one after, say, the PEEK below makes /RUN fail silently
'and drop straight back to the DOS prompt.
'=====================================================================

'Input modes -- what the thrower actually controls.
CONST IMEFFORT = 0             'Effort% 0-100, through the MaxGorillaForce# model
CONST IMVELOCITY = 1           'raw launch speed in m/s: no force model, no cap

'Per-throw outcome codes (LastOutcome). 4 is reserved for "unreachable",
'6 for a MaxTicks abort; both are filtered out downstream.
CONST OCEDGE = 0               'left the left/right screen edge -- flight censored
CONST OCTERRAIN = 1            'hit a building or a window
CONST OCGORILLA = 2            'hit a gorilla
CONST OCSELF = 3               'Velocity < 2, scored a self-hit before probing
CONST OCGROUND = 5             'fell past the bottom of the field -- flew clear
CONST OCMAXTICKS = 6           'hit the per-throw tick ceiling

'File numbers
CONST CSVCHAN = 1
CONST CFGCHAN = 2

'--- Mode and feature switches ---
DIM SHARED DataMode            'TRUE = unattended batch run
DIM SHARED InputMode           'IMEFFORT or IMVELOCITY
DIM SHARED Animate             'FALSE strips every pause -- see SUB Rest
DIM SHARED UseEffortJitter     'apply the +/-EffortVarianceMax# jitter
DIM SHARED UseBananaVar        'sample banana mass/size fresh per throw
DIM SHARED ShowHud             'draw the metrics readouts
DIM SHARED Destruct            'FALSE suppresses explosions: they permanently
                               'carve the framebuffer, and the framebuffer IS
                               'the collision model (POINT in PlotShot)
DIM SHARED FastDraw            'TRUE skips drawing that provably cannot affect
                               'collision (banana, windows, sun). Kept separate
                               'from Animate so the two can be tested apart.

'--- Batch parameters ---
DIM SHARED CfgBoards
DIM SHARED CfgThrows
DIM SHARED CfgSeed#
DIM SHARED CfgGravity#         'm/s^2; non-9.8 runs are how "gravity" outliers
                               'are produced, and gravity is a genuine input
                               'the interactive game already prompts for
DIM SHARED CfgAngleMin#, CfgAngleMax#
DIM SHARED CfgEffortMin#, CfgEffortMax#
DIM SHARED CfgVelMin#, CfgVelMax#
DIM SHARED CfgAlternate        'swap tosser between throws on the same board
DIM SHARED CfgFlushEvery
DIM SHARED CfgDt#              'physics timestep; .1 is the game's own value
DIM SHARED CfgMaxTicks         '0 = no ceiling
DIM SHARED OutFile$

'--- Per-throw values lifted out of DoShot and PlotShot so PlayGame can
'--- log them. This follows the file's own existing idiom: LastShotX# and
'--- LastShotY# are already SHARED for exactly this reason.
DIM SHARED AngleRaw#           'as sampled/typed, before the Player 2 mirror
DIM SHARED AngleSim#           'after the mirror -- what PlotShot integrated
DIM SHARED EffortReq#          'requested effort% (-1 when IMVELOCITY)
DIM SHARED EffortAct#          'post-jitter effort% (-1 when IMVELOCITY)
DIM SHARED ForceUsed#          'newtons (-1 when IMVELOCITY)
DIM SHARED VelocityUsed#       'm/s actually integrated
DIM SHARED DragKLast#          '1/m, the drag constant PlotShot used
DIM SHARED LastShotXm#         'signed downrange metres where it came to rest
DIM SHARED LastShotYm#         'signed vertical metres, +up from the launch point
DIM SHARED LastFlightT#        'seconds of simulated flight
DIM SHARED LastOutcome
DIM SHARED LastPointVal
DIM SHARED BoardNum, ThrowNum
DIM SHARED GroundY             'screen y of the ground line, set by MakeCityScape
DIM SHARED RowsSinceFlush
DIM SHARED CsvIsOpen
DIM SHARED RowsWritten&
DIM SHARED CfgFound           'set at module level before ReadCfg runs
DIM SHARED FastDrawSet        'was FASTDRAW given explicitly in the config?

'Screen Mode Variables
DIM SHARED ScrHeight
DIM SHARED ScrWidth
DIM SHARED Mode
DIM SHARED MaxCol

'Screen Color Variables
DIM SHARED ExplosionColor
DIM SHARED SunColor
DIM SHARED BackColor
DIM SHARED SunHit

DIM SHARED SunHt
DIM SHARED GHeight
DIM SHARED MachSpeed AS SINGLE

  DEF FnRan (x) = INT(RND(1) * x) + 1
  DEF SEG = 0                         ' Set NumLock to ON
  KeyFlags = PEEK(1047)
  IF (KeyFlags AND 32) = 0 THEN
    POKE 1047, KeyFlags OR 32
  END IF
  DEF SEG

  'Does the config file exist? Tested here at module level rather than inside
  'ReadCfg, because QBasic 1.1 has no ON ERROR RESUME NEXT (that is
  'QuickBASIC/VB), and RETURNing out of an error handler is not valid either --
  'so neither an inline guard nor a GOSUB can clear the error state. RESUME
  '<label> can, and only works where the handler is in this same scope.
  CfgFound = 1
  ON ERROR GOTO CfgErr
  OPEN "GORCFG.TXT" FOR INPUT AS #CFGCHAN
  CLOSE #CFGCHAN
  GOTO CfgDone
CfgErr:
  CfgFound = 0
  RESUME CfgDone
CfgDone:
  ON ERROR GOTO 0

  'ReadCfg comes first: InitVars' CalcDelay calibration and Intro's key wait
  'both depend on whether this is a batch run.
  ReadCfg

  GOSUB InitVars
  Intro
  GetInputs Name1$, Name2$, NumGames, SeedVal#
  GorillaIntro Name1$, Name2$
  PlayGame Name1$, Name2$, NumGames, SeedVal#
 
  DEF SEG = 0                         ' Restore NumLock state
  POKE 1047, KeyFlags
  DEF SEG

  'END alone is not enough to finish an unattended run. Under QBASIC /RUN,
  'reaching END hands control back to the QBasic editor, which then sits there
  'with the program listed and waits for a keypress -- so DOSBox would never
  'reach its own exit. SYSTEM closes any open files and returns to DOS, letting
  'the [autoexec] script continue to its `exit`.
  IF DataMode THEN SYSTEM
END


CGABanana:
  'BananaLeft
  DATA 327686, -252645316, 60
  'BananaDown
  DATA 196618, -1057030081, 49344
  'BananaUp
  DATA 196618, -1056980800, 63
  'BananaRight
  DATA 327686,  1010580720, 240

EGABanana:
  'BananaLeft
  DATA 458758,202116096,471604224,943208448,943208448,943208448,471604224,202116096,0
  'BananaDown
  DATA 262153, -2134835200, -2134802239, -2130771968, -2130738945,8323072, 8323199, 4063232, 4063294
  'BananaUp
  DATA 262153, 4063232, 4063294, 8323072, 8323199, -2130771968, -2130738945, -2134835200,-2134802239
  'BananaRight
  DATA 458758, -1061109760, -522133504, 1886416896, 1886416896, 1886416896,-522133504,-1061109760,0

InitVars:
  pi# = 4 * ATN(1#)

  'This is a clever way to pick the best graphics mode available
  ON ERROR GOTO ScreenModeError
  Mode = 9
  SCREEN Mode
  ON ERROR GOTO PaletteError
  IF Mode = 9 THEN PALETTE 4, 0   'Check for 64K EGA
  ON ERROR GOTO 0

  'CalcDelay spins for a hard half second timing the emulated CPU. Rest then
  'scales its waits BY that figure (t2# = MachSpeed * t# / SPEEDCONST), so a
  'faster CPU produces LONGER pauses -- which is why raising cpu_cycles does
  'not speed this game up. With Animate off, Rest returns immediately and
  'MachSpeed is never read, so the half second is pure waste.
  IF Animate THEN
    MachSpeed = CalcDelay
  ELSE
    MachSpeed = 1
  END IF

  IF Mode = 9 THEN
    ScrWidth = 640
    ScrHeight = 350
    GHeight = 25
    RESTORE EGABanana
    REDIM LBan&(8), RBan&(8), UBan&(8), DBan&(8)

    FOR i = 0 TO 8
      READ LBan&(i)
    NEXT i

    FOR i = 0 TO 8
      READ DBan&(i)
    NEXT i

    FOR i = 0 TO 8
      READ UBan&(i)
    NEXT i

    FOR i = 0 TO 8
      READ RBan&(i)
    NEXT i

    SunHt = 39

  ELSE

    ScrWidth = 320
    ScrHeight = 200
    GHeight = 12
    RESTORE CGABanana
    REDIM LBan&(2), RBan&(2), UBan&(2), DBan&(2)
    REDIM GorL&(20), GorD&(20), GorR&(20)

    FOR i = 0 TO 2
      READ LBan&(i)
    NEXT i
    FOR i = 0 TO 2
      READ DBan&(i)
    NEXT i
    FOR i = 0 TO 2
      READ UBan&(i)
    NEXT i
    FOR i = 0 TO 2
      READ RBan&(i)
    NEXT i

    MachSpeed = MachSpeed * 1.3
    SunHt = 20
  END IF
RETURN

ScreenModeError:
  IF Mode = 1 THEN
    CLS
    LOCATE 10, 5
    PRINT "Sorry, you must have CGA, EGA color, or VGA graphics to play GORILLA.BAS"
    END
  ELSE
    Mode = 1
    RESUME
  END IF

PaletteError:
  Mode = 1            '64K EGA cards will run in CGA mode.
  RESUME NEXT

REM $STATIC
'CalcDelay:
'  Checks speed of the machine.
FUNCTION CalcDelay!

  s! = TIMER
  DO
    i! = i! + 1
  LOOP UNTIL TIMER - s! >= .5
  CalcDelay! = i!

END FUNCTION

' Center:
'   Centers and prints a text string on a given row
' Parameters:
'   Row - screen row number
'   Text$ - text to be printed
'
SUB Center (Row, Text$)
  Col = MaxCol \ 2
  LOCATE Row, Col - (LEN(Text$) / 2 + .5)
  PRINT Text$;
END SUB

' DoExplosion:
'   Produces explosion when a shot is fired
' Parameters:
'   X#, Y# - location of explosion
'
SUB DoExplosion (x#, y#)

  'The erase pass below is what actually carves terrain: it paints BACKATTR
  'over the blast radius. Callers gate this whole SUB on Destruct; the guards
  'here only cover pacing, for a destructive run that still wants to be fast.
  IF Animate THEN PLAY "MBO0L32EFGEFDC"
  Radius = ScrHeight / 50
  IF Mode = 9 THEN Inc# = .5 ELSE Inc# = .41
  FOR c# = 0 TO Radius STEP Inc#
    CIRCLE (x#, y#), c#, ExplosionColor
  NEXT c#
  FOR c# = Radius TO 0 STEP (-1 * Inc#)
    CIRCLE (x#, y#), c#, BACKATTR
    'Bare spin loop, so Rest's early exit does not cover it.
    IF Animate THEN
      FOR i = 1 TO 100
      NEXT i
    END IF
    Rest .005
  NEXT c#
END SUB

' DoShot:
'   Controls banana shots by accepting player input and plotting
'   shot angle
' Parameters:
'   PlayerNum - Player
'   x, y - Player's gorilla position
'
FUNCTION DoShot (PlayerNum, x, y)

  'Input shot
  IF PlayerNum = 1 THEN
    LocateCol = 1
  ELSE
    IF Mode = 9 THEN
      LocateCol = 66
    ELSE
      LocateCol = 26
    END IF
  END IF

  'This banana's actual weight and size vary from a triangular distribution
  'around the typical values - see TriRand#. Sampled and shown before Angle/
  'Effort are asked for, since a heavier or bigger banana changes how much
  'velocity a given effort produces and the player should get to factor
  'that in, not find out after committing to a throw.
  '
  'With BananaVar off, mass and size are pinned to the typical (Mode) values.
  'Note this is still the GIANT banana -- BananaLengthScale# applies either
  'way -- so it is not the same thing as the real-banana dosbox-metrics build.
  IF UseBananaVar THEN
    BananaMass# = TriRand#(BananaMassMin#, BananaMassMode#, BananaMassMax#)
    BananaDiam# = TriRand#(BananaDiamMin#, BananaDiamMode#, BananaDiamMax#)
  ELSE
    BananaMass# = BananaMassMode#
    BananaDiam# = BananaDiamMode#
  END IF
  BananaArea# = pi# * (BananaDiam# / 2) * (BananaDiam# / 2)

  IF ShowHud THEN
    LOCATE 2, LocateCol
    PRINT "B:" + Fmt1$(BananaMass#) + "kg/" + Fmt1$(BananaArea#) + "m2";
  END IF

  '--- Angle ---
  IF DataMode THEN
    AngleRaw# = UniRand#(CfgAngleMin#, CfgAngleMax#)
  ELSE
    LOCATE 3, LocateCol
    PRINT "Angle:";
    AngleRaw# = GetNum#(3, LocateCol + 7)
  END IF

  '--- Launch speed: either straight from the thrower, or through the force
  '--- model. In VELOCITY mode nothing caps the speed, so this variant can
  '--- cover the full range the Python generator samples; in EFFORT mode the
  '--- ceiling is MaxThrowVelocity#, which is a deliberate calibration.
  IF InputMode = IMVELOCITY THEN

    IF DataMode THEN
      VelocityUsed# = UniRand#(CfgVelMin#, CfgVelMax#)
    ELSE
      LOCATE 4, LocateCol
      PRINT "Velocity:";
      VelocityUsed# = GetNum#(4, LocateCol + 10)
      IF VelocityUsed# < 0 THEN VelocityUsed# = 0
    END IF
    'Sentinels, not blanks: an empty CSV field reads back as 0.0, and 0 is a
    'legal effort, so a blank would be indistinguishable from a real value.
    EffortReq# = -1
    EffortAct# = -1
    ForceUsed# = -1

  ELSE

    IF DataMode THEN
      EffortReq# = UniRand#(CfgEffortMin#, CfgEffortMax#)
    ELSE
      LOCATE 4, LocateCol
      PRINT "Effort%:";
      EffortReq# = GetNum#(4, LocateCol + 9)
    END IF

    'The gorilla can't apply exactly the intended effort every time - a
    'triangular jitter around the requested Effort% - see TriRand#.
    IF UseEffortJitter THEN
      EffortAct# = TriRand#(EffortReq# - EffortVarianceMax#, EffortReq#, EffortReq# + EffortVarianceMax#)
    ELSE
      EffortAct# = EffortReq#
    END IF

    'Convert effort (0-100% of MaxGorillaForce#) to a launch velocity via
    'work-energy over the throwing stroke: Force*ThrowStrokeM# = .5*m*v^2
    ForceUsed# = (EffortAct# / 100) * MaxGorillaForce#
    IF ForceUsed# < 0 THEN ForceUsed# = 0
    VelocityUsed# = SQR(2 * ForceUsed# * ThrowStrokeM# / BananaMass#)

  END IF

  'Player 2 throws toward -x. AngleRaw# stays as aimed (0-90) because that is
  'what gets logged and what the pipeline's 0-90 range check expects;
  'AngleSim# is the mirrored value actually integrated.
  AngleSim# = AngleRaw#
  IF PlayerNum = 2 THEN
    AngleSim# = 180 - AngleRaw#
  END IF

  'Erase input
  IF ShowHud THEN
    FOR i = 1 TO 5
      LOCATE i, 1
      PRINT SPACE$(30 \ (80 \ MaxCol));
      LOCATE i, (50 \ (80 \ MaxCol))
      PRINT SPACE$(30 \ (80 \ MaxCol));
    NEXT
  END IF

  'Pass a scratch copy of the angle: PlotShot converts its Angle# parameter to
  'radians in place, and QBasic passes by reference, so handing it AngleSim#
  'directly would overwrite the very value we are about to log.
  SunHit = FALSE
  AngleArg# = AngleSim#
  PlayerHit = PlotShot(x, y, AngleArg#, VelocityUsed#, PlayerNum)
  IF PlayerHit = 0 THEN
    DoShot = FALSE
  ELSE
    DoShot = TRUE
    IF PlayerHit = PlayerNum THEN PlayerNum = 3 - PlayerNum
    IF Animate THEN VictoryDance PlayerNum
  END IF

END FUNCTION

' DoSun:
'   Draws the sun at the top of the screen.
' Parameters:
'   Mouth - If TRUE draws "O" mouth else draws a smile mouth.
'
SUB DoSun (Mouth)

  'set position of sun
  x = ScrWidth \ 2: y = Scl(25)

  'clear old sun
  LINE (x - Scl(22), y - Scl(18))-(x + Scl(22), y + Scl(18)), BACKATTR, BF

  'draw new sun:
  'body
  CIRCLE (x, y), Scl(12), SUNATTR
  PAINT (x, y), SUNATTR

  'rays
  LINE (x - Scl(20), y)-(x + Scl(20), y), SUNATTR
  LINE (x, y - Scl(15))-(x, y + Scl(15)), SUNATTR

  LINE (x - Scl(15), y - Scl(10))-(x + Scl(15), y + Scl(10)), SUNATTR
  LINE (x - Scl(15), y + Scl(10))-(x + Scl(15), y - Scl(10)), SUNATTR

  LINE (x - Scl(8), y - Scl(13))-(x + Scl(8), y + Scl(13)), SUNATTR
  LINE (x - Scl(8), y + Scl(13))-(x + Scl(8), y - Scl(13)), SUNATTR

  LINE (x - Scl(18), y - Scl(5))-(x + Scl(18), y + Scl(5)), SUNATTR
  LINE (x - Scl(18), y + Scl(5))-(x + Scl(18), y - Scl(5)), SUNATTR

  'mouth
  IF Mouth THEN  'draw "o" mouth
    CIRCLE (x, y + Scl(5)), Scl(2.9), 0
    PAINT (x, y + Scl(5)), 0, 0
  ELSE           'draw smile
    CIRCLE (x, y), Scl(8), 0, (210 * pi# / 180), (330 * pi# / 180)
  END IF

  'eyes
  CIRCLE (x - 3, y - 2), 1, 0
  CIRCLE (x + 3, y - 2), 1, 0
  PSET (x - 3, y - 2), 0
  PSET (x + 3, y - 2), 0

END SUB

'DrawBan:
'  Draws the banana
'Parameters:
'  xc# - Horizontal Coordinate
'  yc# - Vertical Coordinate
'  r - rotation position (0-3). (  \_/  ) /-\
'  bc - if TRUE then DrawBan draws the banana ELSE it erases the banana
SUB DrawBan (xc#, yc#, r, bc)

SELECT CASE r
  CASE 0
    IF bc THEN PUT (xc#, yc#), LBan&, PSET ELSE PUT (xc#, yc#), LBan&, XOR
  CASE 1
    IF bc THEN PUT (xc#, yc#), UBan&, PSET ELSE PUT (xc#, yc#), UBan&, XOR
  CASE 2
    IF bc THEN PUT (xc#, yc#), DBan&, PSET ELSE PUT (xc#, yc#), DBan&, XOR
  CASE 3
    IF bc THEN PUT (xc#, yc#), RBan&, PSET ELSE PUT (xc#, yc#), RBan&, XOR
END SELECT

END SUB

'DrawGorilla:
'  Draws the Gorilla in either CGA or EGA mode
'  and saves the graphics data in an array.
'Parameters:
'  x - x coordinate of gorilla
'  y - y coordinate of the gorilla
'  arms - either Left up, Right up, or both down
SUB DrawGorilla (x, y, arms)
  DIM i AS SINGLE   ' Local index must be single precision

  'draw head
  LINE (x - Scl(4), y)-(x + Scl(2.9), y + Scl(6)), OBJECTCOLOR, BF
  LINE (x - Scl(5), y + Scl(2))-(x + Scl(4), y + Scl(4)), OBJECTCOLOR, BF

  'draw eyes/brow
  LINE (x - Scl(3), y + Scl(2))-(x + Scl(2), y + Scl(2)), 0

  'draw nose if ega
  IF Mode = 9 THEN
    FOR i = -2 TO -1
      PSET (x + i, y + 4), 0
      PSET (x + i + 3, y + 4), 0
    NEXT i
  END IF

  'neck
  LINE (x - Scl(3), y + Scl(7))-(x + Scl(2), y + Scl(7)), OBJECTCOLOR

  'body
  LINE (x - Scl(8), y + Scl(8))-(x + Scl(6.9), y + Scl(14)), OBJECTCOLOR, BF
  LINE (x - Scl(6), y + Scl(15))-(x + Scl(4.9), y + Scl(20)), OBJECTCOLOR, BF

  'legs
  FOR i = 0 TO 4
    CIRCLE (x + Scl(i), y + Scl(25)), Scl(10), OBJECTCOLOR, 3 * pi# / 4, 9 * pi# / 8
    CIRCLE (x + Scl(-6) + Scl(i - .1), y + Scl(25)), Scl(10), OBJECTCOLOR, 15 * pi# / 8, pi# / 4
  NEXT

  'chest
  CIRCLE (x - Scl(4.9), y + Scl(10)), Scl(4.9), 0, 3 * pi# / 2, 0
  CIRCLE (x + Scl(4.9), y + Scl(10)), Scl(4.9), 0, pi#, 3 * pi# / 2

  FOR i = -5 TO -1
    SELECT CASE arms
      CASE 1
        'Right arm up
        CIRCLE (x + Scl(i - .1), y + Scl(14)), Scl(9), OBJECTCOLOR, 3 * pi# / 4, 5 * pi# / 4
        CIRCLE (x + Scl(4.9) + Scl(i), y + Scl(4)), Scl(9), OBJECTCOLOR, 7 * pi# / 4, pi# / 4
        GET (x - Scl(15), y - Scl(1))-(x + Scl(14), y + Scl(28)), GorR&
      CASE 2
        'Left arm up
        CIRCLE (x + Scl(i - .1), y + Scl(4)), Scl(9), OBJECTCOLOR, 3 * pi# / 4, 5 * pi# / 4
        CIRCLE (x + Scl(4.9) + Scl(i), y + Scl(14)), Scl(9), OBJECTCOLOR, 7 * pi# / 4, pi# / 4
        GET (x - Scl(15), y - Scl(1))-(x + Scl(14), y + Scl(28)), GorL&
      CASE 3
        'Both arms down
        CIRCLE (x + Scl(i - .1), y + Scl(14)), Scl(9), OBJECTCOLOR, 3 * pi# / 4, 5 * pi# / 4
        CIRCLE (x + Scl(4.9) + Scl(i), y + Scl(14)), Scl(9), OBJECTCOLOR, 7 * pi# / 4, pi# / 4
        GET (x - Scl(15), y - Scl(1))-(x + Scl(14), y + Scl(28)), GorD&
    END SELECT
  NEXT i
END SUB

'ExplodeGorilla:
'  Causes gorilla explosion when a direct hit occurs
'Parameters:
'  X#, Y# - shot location
FUNCTION ExplodeGorilla (x#, y#)
  YAdj = Scl(12)
  XAdj = Scl(5)
  SclX# = ScrWidth / 320
  SclY# = ScrHeight / 200
  'Attribution is by screen half, not by which sprite was actually struck.
  'That holds in practice because gorillas sit on buildings 2-3 and
  'LastBuilding-1/-2, but it is why the CSV logs the landing position and both
  'gorilla centres rather than trusting this number.
  IF x# < ScrWidth / 2 THEN PlayerHit = 1 ELSE PlayerHit = 2

  'With Destruct off, score the hit and leave the framebuffer alone.
  '
  'The third loop below paints BACKATTR over the gorilla, permanently erasing
  'it. Because POINT reads the framebuffer for collision, a gorilla erased on
  'throw 3 cannot be hit again on throws 4..N of the same board -- the target
  'geometry would silently change mid-board, and only for boards where an
  'early throw happened to connect. For a playable game the destruction is the
  'whole point; for data generation it is a bias.
  IF NOT Destruct THEN
    ExplodeGorilla = PlayerHit
    EXIT FUNCTION
  END IF

  IF Animate THEN PLAY "MBO0L16EFGEFDC"

  FOR i = 1 TO 8 * SclX#
    CIRCLE (GorillaX(PlayerHit) + 3.5 * SclX# + XAdj, GorillaY(PlayerHit) + 7 * SclY# + YAdj), i, ExplosionColor, , , -1.57
    LINE (GorillaX(PlayerHit) + 7 * SclX#, GorillaY(PlayerHit) + 9 * SclY# - i)-(GorillaX(PlayerHit), GorillaY(PlayerHit) + 9 * SclY# - i), ExplosionColor
  NEXT i

  FOR i = 1 TO 16 * SclX#
    IF i < (8 * SclX#) THEN CIRCLE (GorillaX(PlayerHit) + 3.5 * SclX# + XAdj, GorillaY(PlayerHit) + 7 * SclY# + YAdj), (8 * SclX# + 1) - i, BACKATTR, , , -1.57
    CIRCLE (GorillaX(PlayerHit) + 3.5 * SclX# + XAdj, GorillaY(PlayerHit) + YAdj), i, i MOD 2 + 1, , , -1.57
  NEXT i

  FOR i = 24 * SclX# TO 1 STEP -1
    CIRCLE (GorillaX(PlayerHit) + 3.5 * SclX# + XAdj, GorillaY(PlayerHit) + YAdj), i, BACKATTR, , , -1.57
    'A bare spin loop, not a Rest call, so it needs its own gate: roughly
    '24 x 200 empty iterations of emulated CPU per gorilla hit.
    IF Animate THEN
      FOR Count = 1 TO 200
      NEXT
    END IF
  NEXT i

  ExplodeGorilla = PlayerHit
END FUNCTION

'GetInputs:
'  Gets user inputs at beginning of game
'Parameters:
'  Player1$, Player2$ - player names
'  NumGames - number of games to play
SUB GetInputs (Player1$, Player2$, NumGames, SeedVal#)
  COLOR 7, 0
  CLS

  'A batch run takes all five of these from the config file instead of the
  'keyboard. Note there is deliberately no interactive fallback: if the config
  'is missing, ReadCfg has already ended the program. Falling back to prompts
  'here would hang an unattended run until its harness timed out, which is a
  'far more confusing failure than exiting immediately.
  IF DataMode THEN
    Player1$ = "P1"
    Player2$ = "P2"
    NumGames = CfgBoards        'a flat board count, not "first to N" -- see PlayGame
    gravity# = CfgGravity#
    SeedVal# = CfgSeed#
    EXIT SUB
  END IF

  LOCATE 8, 15
  LINE INPUT "Name of Player 1 (Default = 'Player 1'): "; Player1$
  IF Player1$ = "" THEN
    Player1$ = "Player 1"
  ELSE
    Player1$ = LEFT$(Player1$, 10)
  END IF

  LOCATE 10, 15
  LINE INPUT "Name of Player 2 (Default = 'Player 2'): "; Player2$
  IF Player2$ = "" THEN
    Player2$ = "Player 2"
  ELSE
    Player2$ = LEFT$(Player2$, 10)
  END IF

  DO
    LOCATE 12, 56: PRINT SPACE$(25);
    LOCATE 12, 13
    INPUT "Play to how many total points (Default = 3)"; game$
    NumGames = VAL(LEFT$(game$, 2))
  LOOP UNTIL NumGames > 0 AND LEN(game$) < 3 OR LEN(game$) = 0
  IF NumGames = 0 THEN NumGames = 3

  DO
    LOCATE 14, 53: PRINT SPACE$(28);
    LOCATE 14, 17
    INPUT "Gravity in Meters/Sec (Earth = 9.8)"; grav$
    gravity# = VAL(grav$)
  LOOP UNTIL gravity# > 0 OR LEN(grav$) = 0
  IF gravity# = 0 THEN gravity# = 9.8

  LOCATE 16, 53: PRINT SPACE$(28);
  LOCATE 16, 11
  INPUT "Playing field seed (blank = random)"; seed$
  SeedVal# = VAL(seed$)
END SUB

'GetNum:
'  Gets valid numeric input from user
'Parameters:
'  Row, Col - location to echo input
FUNCTION GetNum# (Row, Col)
  Result$ = ""
  Done = FALSE
  WHILE INKEY$ <> "": WEND   'Clear keyboard buffer

  DO WHILE NOT Done

    LOCATE Row, Col
    PRINT Result$; CHR$(95); "    ";

    Kbd$ = INKEY$
    SELECT CASE Kbd$
      CASE "0" TO "9"
        Result$ = Result$ + Kbd$
      CASE "."
        IF INSTR(Result$, ".") = 0 THEN
          Result$ = Result$ + Kbd$
        END IF
      CASE CHR$(13)
        IF VAL(Result$) > 360 THEN
          Result$ = ""
        ELSE
          Done = TRUE
        END IF
      CASE CHR$(8)
        IF LEN(Result$) > 0 THEN
          Result$ = LEFT$(Result$, LEN(Result$) - 1)
        END IF
      CASE ELSE
        IF LEN(Kbd$) > 0 THEN
          BEEP
        END IF
      END SELECT
  LOOP

  LOCATE Row, Col
  PRINT Result$; " ";

  GetNum# = VAL(Result$)
END FUNCTION

'GorillaIntro:
'  Displays gorillas on screen for the first time
'  allows the graphical data to be put into an array
'Parameters:
'  Player1$, Player2$ - The names of the players
'
SUB GorillaIntro (Player1$, Player2$)
  'Do NOT skip this SUB in batch mode, however much it looks like pure intro.
  'The three DrawGorilla calls below are the only place GorD&/GorL&/GorR& ever
  'get filled -- DrawGorilla renders each pose and GETs it back off the
  'framebuffer. Without them every PUT draws garbage, POINT never returns
  'OBJECTCOLOR, and no gorilla hit can ever register. Only the menu and the
  'keypress wait are skipped; Char$ is forced to "P" so the optional "view
  'intro" animation (and its Rest/PLAY calls) is bypassed.
  IF DataMode THEN
    Char$ = "P"
  ELSE
    LOCATE 16, 34: PRINT "--------------"
    LOCATE 18, 34: PRINT "V = View Intro"
    LOCATE 19, 34: PRINT "P = Play Game"
    LOCATE 21, 35: PRINT "Your Choice?"

    DO WHILE Char$ = ""
      Char$ = INKEY$
    LOOP
  END IF

  IF Mode = 1 THEN
    x = 125
    y = 100
  ELSE
    x = 278
    y = 175
  END IF

  SCREEN Mode
  SetScreen

  IF Mode = 1 THEN Center 5, "Please wait while gorillas are drawn."

  VIEW PRINT 9 TO 24

  IF Mode = 9 THEN PALETTE OBJECTCOLOR, BackColor
 
  DrawGorilla x, y, ARMSDOWN
  CLS 2
  DrawGorilla x, y, LEFTUP
  CLS 2
  DrawGorilla x, y, RIGHTUP
  CLS 2
 
  VIEW PRINT 1 TO 25
  IF Mode = 9 THEN PALETTE OBJECTCOLOR, 46
 
  IF UCASE$(Char$) = "V" THEN
    Center 2, "Q B A S I C   G O R I L L A S"
    Center 5, "             STARRING:               "
    P$ = Player1$ + " AND " + Player2$
    Center 7, P$

    PUT (x - 13, y), GorD&, PSET
    PUT (x + 47, y), GorD&, PSET
    Rest 1

    PUT (x - 13, y), GorL&, PSET
    PUT (x + 47, y), GorR&, PSET
    PLAY "t120o1l16b9n0baan0bn0bn0baaan0b9n0baan0b"
    Rest .3

    PUT (x - 13, y), GorR&, PSET
    PUT (x + 47, y), GorL&, PSET
    PLAY "o2l16e-9n0e-d-d-n0e-n0e-n0e-d-d-d-n0e-9n0e-d-d-n0e-"
    Rest .3

    PUT (x - 13, y), GorL&, PSET
    PUT (x + 47, y), GorR&, PSET
    PLAY "o2l16g-9n0g-een0g-n0g-n0g-eeen0g-9n0g-een0g-"
    Rest .3

    PUT (x - 13, y), GorR&, PSET
    PUT (x + 47, y), GorL&, PSET
    PLAY "o2l16b9n0baan0g-n0g-n0g-eeen0o1b9n0baan0b"
    Rest .3

    FOR i = 1 TO 4
      PUT (x - 13, y), GorL&, PSET
      PUT (x + 47, y), GorR&, PSET
      PLAY "T160O0L32EFGEFDC"
      Rest .1
      PUT (x - 13, y), GorR&, PSET
      PUT (x + 47, y), GorL&, PSET
      PLAY "T160O0L32EFGEFDC"
      Rest .1
    NEXT
  END IF
END SUB

'Intro:
'  Displays game introduction
SUB Intro

  'The SCREEN 0 / WIDTH / MaxCol setup below is load-bearing for the rest of
  'the program, so a batch run still executes it -- only the text and the key
  'wait are skipped.
  SCREEN 0
  WIDTH 80, 25
  MaxCol = 80
  COLOR 15, 0
  CLS

  IF DataMode THEN
    Center 4, "GORILLAS -- BATCH DATA GENERATION"
    Center 6, "Boards: " + LTRIM$(STR$(CfgBoards)) + "   Throws/board: " + LTRIM$(STR$(CfgThrows))
    IF InputMode = IMVELOCITY THEN
      Center 7, "Input mode: VELOCITY (m/s)"
    ELSE
      Center 7, "Input mode: EFFORT (%)"
    END IF
    Center 9, "Writing " + OutFile$
    IF Mode = 1 THEN MaxCol = 40
    EXIT SUB
  END IF

  Center 4, "Q B a s i c    G O R I L L A S"
  COLOR 7
  Center 6, "Copyright (C) Microsoft Corporation 1990"
  Center 8, "Your mission is to hit your opponent with the exploding"
  Center 9, "banana by varying the angle and power of your throw, taking"
  Center 10, "into account wind speed, gravity, and the city skyline."
  Center 11, "The wind speed is shown by a directional arrow at the bottom"
  Center 12, "of the playing field, its length relative to its strength."
  Center 24, "Press any key to continue"

  PLAY "MBT160O1L8CDEDCDL4ECC"
  SparklePause
  IF Mode = 1 THEN MaxCol = 40
END SUB

'MakeCityScape:
'  Creates random skyline for game
'Parameters:
'  BCoor() - a user-defined type array which stores the coordinates of
'  the upper left corner of each building.
SUB MakeCityScape (BCoor() AS XYPoint)

  x = 2

  'Set the sloping trend of the city scape. NewHt is new building height
  Slope = FnRan(6)
  SELECT CASE Slope
    CASE 1: NewHt = 15                 'Upward slope
    CASE 2: NewHt = 130                'Downward slope
    CASE 3 TO 5: NewHt = 15            '"V" slope - most common
    CASE 6: NewHt = 130                'Inverted "V" slope
  END SELECT

  IF Mode = 9 THEN
    BottomLine = 335                   'Bottom of building
    HtInc = 10                         'Increase value for new height
    DefBWidth = 37                     'Default building height
    RandomHeight = 120                 'Random height difference
    WWidth = 3                         'Window width
    WHeight = 6                        'Window height
    WDifV = 15                         'Counter for window spacing - vertical
    WDifh = 10                         'Counter for window spacing - horizontal
  ELSE
    BottomLine = 190
    HtInc = 6
    NewHt = NewHt * 20 \ 35            'Adjust for CGA
    DefBWidth = 18
    RandomHeight = 54
    WWidth = 1
    WHeight = 2
    WDifV = 5
    WDifh = 4
  END IF

  'Expose the ground line so the height datum is explicit in the logged data
  'rather than a constant the reader has to know. Every launch_height_m and
  'landing_height_m is measured from here.
  GroundY = BottomLine

  CurBuilding = 1
  DO

    SELECT CASE Slope
      CASE 1
        NewHt = NewHt + HtInc
      CASE 2
        NewHt = NewHt - HtInc
      CASE 3 TO 5
        IF x > ScrWidth \ 2 THEN
          NewHt = NewHt - 2 * HtInc
        ELSE
          NewHt = NewHt + 2 * HtInc
        END IF
      CASE 4
        IF x > ScrWidth \ 2 THEN
          NewHt = NewHt + 2 * HtInc
        ELSE
          NewHt = NewHt - 2 * HtInc
        END IF
    END SELECT

    'Set width of building and check to see if it would go off the screen
    BWidth = FnRan(DefBWidth) + DefBWidth
    IF x + BWidth > ScrWidth THEN BWidth = ScrWidth - x - 2

    'Set height of building and check to see if it goes below screen
    BHeight = FnRan(RandomHeight) + NewHt
    IF BHeight < HtInc THEN BHeight = HtInc

    'Check to see if Building is too high
    IF BottomLine - BHeight <= MaxHeight + GHeight THEN BHeight = MaxHeight + GHeight - 5

    'Set the coordinates of the building into the array
    BCoor(CurBuilding).XCoor = x
    BCoor(CurBuilding).YCoor = BottomLine - BHeight

    IF Mode = 9 THEN BuildingColor = FnRan(3) + 4 ELSE BuildingColor = 2

    'Draw the building, outline first, then filled
    LINE (x - 1, BottomLine + 1)-(x + BWidth + 1, BottomLine - BHeight - 1), BACKGROUND, B
    LINE (x, BottomLine)-(x + BWidth, BottomLine - BHeight), BuildingColor, BF

    'Draw the windows.
    '
    'FastDraw skips only the LINE, never the loop or the FnRan call. Two
    'separate reasons, and both matter:
    '
    '  - The FnRan draw must happen regardless. It is consumed once per window,
    '    so the number of draws per board depends on the skyline. Skipping it
    '    would shift the whole RNG stream and a given seed would stop
    '    reproducing the same boards.
    '  - Skipping the LINE cannot affect collision. Windows are filled rects
    '    painted ON TOP of the building rect that was already filled solid on
    '    the line above, so every window pixel is over ground that is non-zero
    '    already. Impact is identical either way; the only observable
    '    difference is the logged pointval (5/6/7 instead of 8/14).
    '
    'This is the single biggest drawing cost on the board: roughly 7 columns x
    '19 rows per building across ~12 buildings, so order 1600 filled LINEs.
    c = x + 3
    DO
      FOR i = BHeight - 3 TO 7 STEP -WDifV
        IF Mode <> 9 THEN
          WinColr = (FnRan(2) - 2) * -3
        ELSEIF FnRan(4) = 1 THEN
          WinColr = 8
        ELSE
          WinColr = WINDOWCOLOR
        END IF
        IF NOT FastDraw THEN
          LINE (c, BottomLine - i)-(c + WWidth, BottomLine - i + WHeight), WinColr, BF
        END IF
      NEXT
      c = c + WDifh
    LOOP UNTIL c >= x + BWidth - 3

    x = x + BWidth + 2

    CurBuilding = CurBuilding + 1

  LOOP UNTIL x > ScrWidth - HtInc

  LastBuilding = CurBuilding - 1

  'Set Wind speed
  Wind = FnRan(10) - 5
  IF FnRan(3) = 1 THEN
    IF Wind > 0 THEN
      Wind = Wind + FnRan(10)
    ELSE
      Wind = Wind - FnRan(10)
    END IF
  END IF

  'Draw Wind speed arrow
  IF Wind <> 0 THEN
    WindLine = Wind * 3 * (ScrWidth \ 320)
    LINE (ScrWidth \ 2, ScrHeight - 5)-(ScrWidth \ 2 + WindLine, ScrHeight - 5), ExplosionColor
    IF Wind > 0 THEN ArrowDir = -2 ELSE ArrowDir = 2
    LINE (ScrWidth / 2 + WindLine, ScrHeight - 5)-(ScrWidth / 2 + WindLine + ArrowDir, ScrHeight - 5 - 2), ExplosionColor
    LINE (ScrWidth / 2 + WindLine, ScrHeight - 5)-(ScrWidth / 2 + WindLine + ArrowDir, ScrHeight - 5 + 2), ExplosionColor
  END IF

END SUB

'PlaceGorillas:
'  PUTs the Gorillas on top of the buildings.  Must have drawn
'  Gorillas first.
'Parameters:
'  BCoor() - user-defined TYPE array which stores upper left coordinates
'  of each building.
SUB PlaceGorillas (BCoor() AS XYPoint)
    
  IF Mode = 9 THEN
    XAdj = 14
    YAdj = 30
  ELSE
    XAdj = 7
    YAdj = 16
  END IF

  'Place gorillas on second or third building from edge
  FOR i = 1 TO 2
    IF i = 1 THEN BNum = FnRan(2) + 1 ELSE BNum = LastBuilding - FnRan(2)

    BWidth = BCoor(BNum + 1).XCoor - BCoor(BNum).XCoor
    GorillaX(i) = BCoor(BNum).XCoor + BWidth / 2 - XAdj
    GorillaY(i) = BCoor(BNum).YCoor - YAdj
    GorillaCenterX(i) = GorillaX(i) + XAdj
    'True vertical center of the arms-down silhouette within its GET/PUT box
    '(GorillaY(i) is the box's top row) - confirmed empirically by rendering
    'GorD& next to a pixel ruler and screenshotting: in mode 9 the silhouette
    'occupies rows 1-29 of the 30-row box (1px slack top and bottom), so its
    'midpoint is exactly YAdj/2 - NOT the old "7*SclY#+Scl(12)" formula (~24),
    'which was borrowed from ExplodeGorilla's hit-animation-centering circle
    '(tuned to look good for a burst effect) and actually lands near the
    'sprite's feet, not its middle.
    GorillaCenterY(i) = GorillaY(i) + YAdj / 2
    PUT (GorillaX(i), GorillaY(i)), GorD&, PSET
  NEXT i

END SUB

'PlayGame:
'  Main game play routine
'Parameters:
'  Player1$, Player2$ - player names
'  NumGames - number of games to play
SUB PlayGame (Player1$, Player2$, NumGames, SeedVal#)
  DIM BCoor(0 TO 30) AS XYPoint
  DIM TotalWins(1 TO 2)

  'Seed once for the whole session so a fixed seed reproduces the same
  'sequence of playing fields (buildings, wind, gorilla spots) across rounds
  IF SeedVal# <> 0 THEN
    RANDOMIZE (SeedVal#)
  ELSE
    RANDOMIZE (TIMER)
  END IF

  IF DataMode THEN CsvOpenFile

  J = 1

  FOR i = 1 TO NumGames

    BoardNum = i
    ThrowNum = 0
    CLS
    CALL MakeCityScape(BCoor())
    CALL PlaceGorillas(BCoor())
    'DoSun is two PAINT flood fills plus a circle and eight lines per board.
    'The sun never blocks the banana -- the probe below sets ShotInSun but
    'never Impact -- so skipping it changes no trajectory and no outcome, only
    'the logged pointval and whether SunHit can fire.
    IF NOT FastDraw THEN DoSun SUNHAPPY
    Hit = FALSE

    'Round shape differs by mode.
    '
    'Interactively a round runs until somebody is hit, which is the game. In a
    'batch run that is unusable: with randomly aimed throws a ~30x30 sprite on
    'a 640x350 field is hit rarely, so a round would run for hundreds of throws
    'and "generate N boards" would stall on the first one. Instead each board
    'gets exactly CfgThrows throws and is then abandoned, hit or not.
    DO
      ThrowNum = ThrowNum + 1

      'Tosser alternation. Keeping it on means each board contributes both
      'throw directions, both wind signs and both launch/target height pairs,
      'and exercises the Player 2 coordinate mirror.
      IF CfgAlternate OR NOT DataMode THEN
        J = 1 - J
      END IF

      IF ShowHud THEN
        LOCATE 1, 1
        PRINT Player1$
        LOCATE 1, (MaxCol - 1 - LEN(Player2$))
        PRINT Player2$
        Center 23, LTRIM$(STR$(TotalWins(1))) + ">Score<" + LTRIM$(STR$(TotalWins(2)))
      END IF
      Tosser = J + 1: Tossee = 2 - J

      'Wind speed relative to the tosser: positive blows away from the tosser
      '(downrange, a tailwind), negative blows toward the tosser (a
      'headwind). Player 1 throws toward +x, Player 2 toward -x, so the sign
      'is mirrored for Player 2. Recomputed each throw since Tosser swaps.
      IF Tosser = 1 THEN WindRelative = Wind ELSE WindRelative = -Wind
      IF ShowHud THEN Center 24, "Wind Speed: " + LTRIM$(STR$(WindRelative)) + " m/s"

      'Distance from the banana's actual launch point (matches StartXPos/
      'StartYPos in PlotShot) to the target's true center (GorillaCenterX/Y -
      'X is the building's horizontal midpoint, Y is the silhouette's true
      'vertical midpoint) - shown in real-world meters. Y is signed: negative
      'when the tosser is above the target, positive when below. Recomputed
      'each throw since Tosser/Tossee swap sides.
      LaunchX# = GorillaX(Tosser)
      IF Tosser = 2 THEN LaunchX# = LaunchX# + Scl(25)
      LaunchY# = GorillaY(Tosser) - Scl(4) - 3

      GorillaDistX# = ABS(GorillaCenterX(Tossee) - LaunchX#) * MetersPerPixel#
      GorillaDistY# = (LaunchY# - GorillaCenterY(Tossee)) * MetersPerPixel#
      IF ShowHud THEN
        Center 22, "Distance  X:" + Fmt1$(GorillaDistX#) + "m  Y:" + Fmt1$(GorillaDistY#) + "m"
      END IF

      'Plot the shot.  Hit is true if Gorilla gets hit.
      Hit = DoShot(Tosser, GorillaX(Tosser), GorillaY(Tosser))

      'Everything the CSV needs is in scope right here, already in metres --
      'the readouts above are effectively a hand-written feature extractor, so
      'this just tees them to a file.
      IF DataMode THEN
        CALL LogThrow(Tosser, Tossee, LaunchX#, LaunchY#, GorillaDistX#, GorillaDistY#, WindRelative)
      END IF

      'Show how far the last throw landed from the target, measured against
      'the target gorilla's true horizontal center (horizontal distance only),
      'in real-world meters
      IF ShowHud THEN
        MissDist# = ABS(LastShotX# - GorillaCenterX(Tossee)) * MetersPerPixel#
        Center 21, "Last Throw Distance to Target: " + Fmt1$(MissDist#) + "m"
      END IF

      'Reset the sun, if it got hit
      IF SunHit AND NOT FastDraw THEN DoSun SUNHAPPY

      IF Hit = TRUE THEN CALL UpdateScores(TotalWins(), Tosser, Hit)

      'A batch board is done after CfgThrows throws regardless of hits; an
      'interactive round ends only on a hit, as it always did.
      IF DataMode THEN
        IF ThrowNum >= CfgThrows THEN EXIT DO
      ELSE
        IF Hit <> FALSE THEN EXIT DO
      END IF
    LOOP

    IF Animate THEN SLEEP 1
  NEXT i

  IF DataMode THEN CsvCloseFile

  SCREEN 0
  WIDTH 80, 25
  COLOR 7, 0
  MaxCol = 80
  CLS

  IF DataMode THEN
    'No SparklePause: it spins on INKEY$ forever, so an unattended run could
    'never terminate and DOSBox would sit at this screen until the harness
    'killed it. Printing the row count gives the harness something to scrape
    'and a human something to glance at.
    Center 8, "BATCH COMPLETE"
    Center 10, "Boards: " + LTRIM$(STR$(NumGames))
    Center 11, "Rows written: " + LTRIM$(STR$(RowsWritten&))
    Center 12, "Output: " + OutFile$
    COLOR 7, 0
    EXIT SUB
  END IF

  Center 8, "GAME OVER!"
  Center 10, "Score:"
  LOCATE 11, 30: PRINT Player1$; TAB(50); TotalWins(1)
  LOCATE 12, 30: PRINT Player2$; TAB(50); TotalWins(2)
  Center 24, "Press any key to continue"
  SparklePause
  COLOR 7, 0
  CLS
END SUB

'PlayGame:
'  Plots banana shot across the screen
'Parameters:
'  StartX, StartY - starting shot location
'  Angle - shot angle
'  Velocity - shot velocity
'  PlayerNum - the banana thrower
FUNCTION PlotShot (StartX, StartY, Angle#, Velocity#, PlayerNum)

  Angle# = Angle# / 180 * pi#  'Convert degree angle to radians
  Radius = Mode MOD 7

  vx# = COS(Angle#) * Velocity#     'm/s - Velocity# is real-world m/s
  vy# = SIN(Angle#) * Velocity#     'm/s, +up

  Xm# = 0                           'meters traveled, physics space
  Ym# = 0                           'meters traveled, +up, physics space
  dt# = CfgDt#                      'physics timestep, seconds (matches t# below)

  DragK# = .5 * AirDensity# * BananaCd# * BananaArea# / BananaMass#  '1/m
  DragKLast# = DragK#               'logged so the mapping can be cross-checked

  'Default outcome. Overwritten below by whichever exit actually happens.
  LastOutcome = OCEDGE
  LastPointVal = 0
  Ticks = 0

  oldx# = StartX
  oldy# = StartY

  'draw gorilla toss
  IF PlayerNum = 1 THEN
    PUT (StartX, StartY), GorL&, PSET
  ELSE
    PUT (StartX, StartY), GorR&, PSET
  END IF
  
  'throw sound
  IF Animate THEN PLAY "MBo0L32A-L64CL16BL64A+"
  Rest .1

  'redraw gorilla
  PUT (StartX, StartY), GorD&, PSET

  adjust = Scl(4)                   'For scaling CGA

  xedge = Scl(9) * (2 - PlayerNum)  'Find leading edge of banana for check

  Impact = FALSE
  ShotInSun = FALSE
  OnScreen = TRUE
  PlayerHit = 0
  NeedErase = FALSE

  StartXPos = StartX
  StartYPos = StartY - adjust - 3

  IF PlayerNum = 2 THEN
    StartXPos = StartXPos + Scl(25)
    direction = Scl(4)
  ELSE
    direction = Scl(-4)
  END IF

  IF Velocity# < 2 THEN             'Shot too slow - hit self
    x# = StartX
    y# = StartY
    pointval = OBJECTCOLOR
    LastOutcome = OCSELF
  END IF

  DO WHILE (NOT Impact) AND OnScreen

  Rest .02
  Ticks = Ticks + 1

  'Erase old banana, if necessary
  IF NeedErase THEN
    NeedErase = FALSE
    CALL DrawBan(oldx#, oldy#, oldrot, FALSE)
  END IF

  'Drag opposes velocity relative to the wind (Wind is real-world m/s)
  relVx# = vx# - Wind
  relVy# = vy#
  relSpeed# = SQR(relVx# ^ 2 + relVy# ^ 2)

  ax# = -DragK# * relSpeed# * relVx#
  ay# = -gravity# - DragK# * relSpeed# * relVy#

  vx# = vx# + ax# * dt#
  vy# = vy# + ay# * dt#

  Xm# = Xm# + vx# * dt#
  Ym# = Ym# + vy# * dt#

  x# = StartXPos + Xm# / MetersPerPixel#
  y# = StartYPos - (Ym# / MetersPerPixel#) * (ScrHeight / 350)
         
  'Leaving the field is split by WHICH edge, because the two mean opposite
  'things for the data. Buildings are drawn upward from the ground line, and
  'the rows below it are background, so a banana that clears every building
  'keeps falling and exits through the BOTTOM with Impact still FALSE. Those
  'are the throws that flew clear and landed -- the closest analogue to the
  'Python generator's "landed on a platform", and the most useful rows here.
  'A left/right exit is different: the flight is censored, there is no landing
  'point, and the row has to be discarded. The original code could not tell
  'them apart because it had no reason to.
  IF (x# >= ScrWidth - Scl(10)) OR (x# <= 3) THEN
    OnScreen = FALSE
    LastOutcome = OCEDGE
  ELSEIF (y# >= ScrHeight - 3) THEN
    OnScreen = FALSE
    LastOutcome = OCGROUND
  END IF

  'Optional ceiling on flight length. A near-vertical lob at high speed can
  'run for hundreds of ticks; those rows are rarely useful and cost the most
  'time to produce. Off by default (MaxTicks = 0) because it changes which
  'throws appear in the data.
  IF CfgMaxTicks > 0 AND Ticks >= CfgMaxTicks THEN
    OnScreen = FALSE
    LastOutcome = OCMAXTICKS
  END IF

          
  IF OnScreen AND y# > 0 THEN

    'check it
    LookY = 0
    LookX = Scl(8 * (2 - PlayerNum))
    DO
      pointval = POINT(x# + LookX, y# + LookY)
      IF pointval = 0 THEN
        Impact = FALSE
        IF ShotInSun = TRUE THEN
          IF ABS(ScrWidth \ 2 - x#) > Scl(20) OR y# > SunHt THEN ShotInSun = FALSE
        END IF
      ELSEIF pointval = SUNATTR AND y# < SunHt THEN
        IF NOT SunHit AND Animate THEN DoSun SUNSHOCK
        SunHit = TRUE
        ShotInSun = TRUE
      ELSE
        Impact = TRUE
      END IF
      LookX = LookX + direction
      LookY = LookY + Scl(6)
    LOOP UNTIL Impact OR LookX <> Scl(4)
   
    'Drawing the banana is provably irrelevant to collision, so FastDraw skips
    'it. Look at the loop order: the previous frame's banana is ERASED at the
    'top, before POINT is sampled, and the new one is drawn here, AFTER. The
    'banana is therefore never on screen during its own probe and can never be
    'what POINT hits. Skipping it costs the visible trajectory and two sprite
    'operations per tick, and changes no physics and no logged value.
    IF NOT ShotInSun AND NOT Impact AND NOT FastDraw THEN
      'plot it
      rot = (t# * 10) MOD 4
      CALL DrawBan(x#, y#, rot, TRUE)
      NeedErase = TRUE
    END IF
            
    oldx# = x#
    oldy# = y#
    oldrot = rot

  END IF


  t# = t# + dt#

  LOOP

  'An impact overrides whichever edge-exit code was provisionally set above.
  'Note the ELSEIF deliberately has no "AND Impact" guard -- that is how the
  'Velocity# < 2 self-hit, which sets pointval before the loop ever runs, gets
  'scored without probing anything.
  IF pointval <> OBJECTCOLOR AND Impact THEN
    LastOutcome = OCTERRAIN
    IF Destruct THEN CALL DoExplosion(x# + adjust, y# + adjust)
  ELSEIF pointval = OBJECTCOLOR THEN
    IF LastOutcome <> OCSELF THEN LastOutcome = OCGORILLA
    PlayerHit = ExplodeGorilla(x#, y#)
  END IF

  LastShotX# = x#
  LastShotY# = y#

  'Metres straight out of the integrator, rather than converting the pixel
  'position back through MetersPerPixel#. Xm#/Ym# never went through the pixel
  'grid, so this avoids the quantisation the screen coordinates carry, and it
  'makes landing distance directly comparable with the Python generator, which
  'also takes its landing point from the integrator.
  '
  'Both are signed: Xm# is negative for Player 2, who throws toward -x.
  LastShotXm# = Xm#
  LastShotYm# = Ym#
  LastFlightT# = t#
  LastPointVal = pointval

  PlotShot = PlayerHit

END FUNCTION

'Rest:
'  pauses the program
SUB Rest (t#)
  'One early exit here de-blocks all five call sites at once: the per-tick
  'pause in PlotShot's physics loop, the post-throw pause, DoExplosion's
  'fifteen, and VictoryDance's eight.
  '
  'It has to skip the whole SUB, not just shorten the wait. TIMER is driven by
  'the 18.2 Hz BIOS tick, so "LOOP UNTIL TIMER - s# > t2#" cannot return until
  'the tick counter moves -- 0 to 55 ms, ~27 ms on average, even for t# = 0.
  'Across the 20-60 ticks of a typical arc that alone is 0.5-1.6 s per throw.
  IF NOT Animate THEN EXIT SUB

  s# = TIMER
  t2# = MachSpeed * t# / SPEEDCONST
  DO
  LOOP UNTIL TIMER - s# > t2#
END SUB

'Scl:
'  Pass the number in to scaling for cga.  If the number is a decimal, then we
'  want to scale down for cga or scale up for ega.  This allows a full range
'  of numbers to be generated for scaling.
'  (i.e. for 3 to get scaled to 1, pass in 2.9)
FUNCTION Scl (n!)

  IF n! <> INT(n!) THEN
      IF Mode = 1 THEN n! = n! - 1
  END IF
  IF Mode = 1 THEN
      Scl = CINT(n! / 2 + .1)
  ELSE
      Scl = CINT(n!)
  END IF

END FUNCTION

'TriRand#:
'  Samples a triangular distribution: values cluster near Mode# and taper
'  linearly to zero probability at Lo#/Hi# (unlike a flat RND range, where
'  every value is equally likely). Standard inverse-CDF sampling.
'Parameters:
'  Lo#, Mode#, Hi# - the distribution's minimum, most-likely, and maximum
'
FUNCTION TriRand# (Lo#, Mode#, Hi#)
  u# = RND(1)
  Fc# = (Mode# - Lo#) / (Hi# - Lo#)
  IF u# < Fc# THEN
    TriRand# = Lo# + SQR(u# * (Hi# - Lo#) * (Mode# - Lo#))
  ELSE
    TriRand# = Hi# - SQR((1 - u#) * (Hi# - Lo#) * (Hi# - Mode#))
  END IF
END FUNCTION

'Fmt1$:
'  Formats a number to exactly one decimal place, sign preserved.
FUNCTION Fmt1$ (n#)

  sign$ = ""
  v# = n#
  IF v# < 0 THEN
    sign$ = "-"
    v# = -v#
  END IF
  Tenths& = CLNG(v# * 10 + .5)
  Fmt1$ = sign$ + LTRIM$(STR$(Tenths& \ 10)) + "." + LTRIM$(STR$(Tenths& MOD 10))

END FUNCTION

'Fmt4$:
'  Formats a number to exactly four decimal places, sign preserved, for the
'  CSV. Fmt1$ is too coarse to log as an ML feature, and the two obvious
'  alternatives are both traps:
'
'    PRINT #1, USING "###.###,###.###"  -- a comma next to a numeric field is
'      a digit-GROUPING specifier in QBasic, not a literal separator.
'    LTRIM$(STR$(x#))  -- on a double this can emit D-exponent notation
'      (1.23D-05), which Python's float() will not parse.
'
'  Unlike Fmt1$ this rounds once, via CLNG's own round-to-nearest, rather than
'  adding .5 and truncating (which double-rounds).
'
'  CLNG overflows past +/-2147483647, i.e. |n#| > 214748 at four decimals.
'  Every quantity logged here is under 1000, so that is unreachable -- but it
'  is the one numeric landmine in the CSV writer, so do not reuse this for
'  something larger without widening it.
FUNCTION Fmt4$ (n#)

  sign$ = ""
  v# = n#
  IF v# < 0 THEN
    sign$ = "-"
    v# = -v#
  END IF
  T& = CLNG(v# * 10000)
  f$ = LTRIM$(STR$(T& MOD 10000))
  f$ = STRING$(4 - LEN(f$), "0") + f$
  Fmt4$ = sign$ + LTRIM$(STR$(T& \ 10000)) + "." + f$

END FUNCTION

'UniRand#:
'  A uniform double in [Lo#, Hi#]. Used to sample angle and effort/velocity in
'  batch mode, replacing the two GetNum# keyboard prompts.
FUNCTION UniRand# (Lo#, Hi#)
  UniRand# = Lo# + RND(1) * (Hi# - Lo#)
END FUNCTION

'Dbg:
'  Appends a progress marker to D:DIAG.TXT, re-opening the file each time so
'  the line survives even if the program later hangs -- QBasic only flushes on
'  CLOSE, so a buffered marker is no use for diagnosing a stall.
'
'  Left in, uncalled, because it earned its keep: it is how the "config parsed
'  but only the first key took effect" bug was found, and it is the only
'  practical way to see inside an unattended DOSBox run (the screen cannot be
'  scraped and keystrokes cannot be injected into this window). Sprinkle calls
'  where needed, run, then read D:DIAG.TXT from the host.
SUB Dbg (s$)
  OPEN "D:DIAG.TXT" FOR APPEND AS #3
  PRINT #3, s$
  CLOSE #3
END SUB

'ReadCfg:
'  Reads GORCFG.TXT from the mounted drive and sets every batch switch.
'
'  QBASIC.EXE /RUN gives no access to COMMAND$, so a config file is the only
'  way to parameterise an unattended run. Parsed as KEY=VALUE rather than by
'  line position, because a positional file goes silently wrong the moment a
'  line is added or reordered.
'
'  Defaults are set FIRST and then overridden, so an older config file missing
'  newer keys still runs. If the file is absent the program prints NO CFG and
'  ends: there is deliberately no fall-back to the interactive prompts, which
'  would leave an unattended run hanging until its harness timed out.
SUB ReadCfg

  'Defaults reproduce the dosbox-modified-physics-metrics-var-display build.
  DataMode = FALSE
  InputMode = IMEFFORT
  Animate = TRUE
  UseEffortJitter = TRUE
  UseBananaVar = TRUE
  ShowHud = TRUE
  Destruct = TRUE
  FastDraw = FALSE
  FastDrawSet = 0

  CfgBoards = 10
  CfgThrows = 24
  CfgSeed# = 0                  '0 means "seed from TIMER" -- see PlayGame
  CfgGravity# = 9.8
  CfgAngleMin# = 10
  CfgAngleMax# = 88             'must stay <= 90: the pipeline's range check
                                'deletes any launch_angle_deg outside 0-90
  CfgEffortMin# = 20
  CfgEffortMax# = 100
  CfgVelMin# = 10               '>= 2: PlotShot treats Velocity < 2 as a self-hit
  CfgVelMax# = 80               'matches the Python generator's SPEED_RANGE
  CfgAlternate = TRUE
  CfgFlushEvery = 50
  CfgDt# = .1                   'the game's own timestep; changing it changes
                                'the physics, so it is logged per row
  CfgMaxTicks = 0
  OutFile$ = "THROWS.CSV"

  'CfgFound was established at module level, where the error state can be
  'cleared properly -- see the comment there.
  IF CfgFound = 0 THEN EXIT SUB   'no config file: stay a normal playable game

  OPEN "GORCFG.TXT" FOR INPUT AS #CFGCHAN

  DO WHILE NOT EOF(CFGCHAN)
    LINE INPUT #CFGCHAN, ln$

    'Strip trailing comments so "BOARDS=5   ' how many" works.
    q = INSTR(ln$, "'")
    IF q > 0 THEN ln$ = LEFT$(ln$, q - 1)
    ln$ = LTRIM$(RTRIM$(ln$))

    IF LEN(ln$) > 0 THEN
      p = INSTR(ln$, "=")
      IF p > 1 THEN
        k$ = UCASE$(LTRIM$(RTRIM$(LEFT$(ln$, p - 1))))
        v$ = LTRIM$(RTRIM$(MID$(ln$, p + 1)))

        'An empty value means "use the default", not "use zero". Without this
        'test VAL("") would quietly set the key to 0 -- which for ANGLEMIN or
        'DT is a legal-looking but wrong value, and for DT would be silently
        'clamped back later. Skipping the whole key is what makes commenting a
        'value out, or writing "ANGLEMAX=", behave the way anyone would expect.
        IF LEN(v$) > 0 THEN
        v# = VAL(v$)

        IF k$ = "DATAMODE" THEN DataMode = (v# <> 0)
        IF k$ = "INPUTMODE" THEN
          IF UCASE$(LEFT$(v$, 1)) = "V" THEN InputMode = IMVELOCITY ELSE InputMode = IMEFFORT
        END IF
        IF k$ = "ANIMATE" THEN Animate = (v# <> 0)
        IF k$ = "EFFORTJITTER" THEN UseEffortJitter = (v# <> 0)
        IF k$ = "BANANAVAR" THEN UseBananaVar = (v# <> 0)
        IF k$ = "HUD" THEN ShowHud = (v# <> 0)
        IF k$ = "DESTRUCT" THEN Destruct = (v# <> 0)
        IF k$ = "FASTDRAW" THEN
          FastDraw = (v# <> 0)
          FastDrawSet = 1       'remember it was given, so the default below
        END IF                  'does not quietly overrule it
        IF k$ = "BOARDS" THEN CfgBoards = v#
        IF k$ = "THROWS" THEN CfgThrows = v#
        IF k$ = "SEED" THEN CfgSeed# = v#
        IF k$ = "GRAVITY" THEN CfgGravity# = v#
        IF k$ = "ANGLEMIN" THEN CfgAngleMin# = v#
        IF k$ = "ANGLEMAX" THEN CfgAngleMax# = v#
        IF k$ = "EFFORTMIN" THEN CfgEffortMin# = v#
        IF k$ = "EFFORTMAX" THEN CfgEffortMax# = v#
        IF k$ = "VELMIN" THEN CfgVelMin# = v#
        IF k$ = "VELMAX" THEN CfgVelMax# = v#
        IF k$ = "ALTERNATE" THEN CfgAlternate = (v# <> 0)
        IF k$ = "FLUSHEVERY" THEN CfgFlushEvery = v#
        IF k$ = "DT" THEN CfgDt# = v#
        IF k$ = "MAXTICKS" THEN CfgMaxTicks = v#
        IF k$ = "OUTFILE" THEN OutFile$ = UCASE$(v$)
        END IF
      END IF
    END IF
  LOOP
  CLOSE #CFGCHAN

  'Clamp to what the rest of the program and the downstream pipeline require.
  IF CfgBoards < 1 THEN CfgBoards = 1
  IF CfgBoards > 32000 THEN CfgBoards = 32000   'BoardNum is an INTEGER
  IF CfgThrows < 1 THEN CfgThrows = 1
  IF CfgAngleMax# > 90 THEN CfgAngleMax# = 90
  IF CfgAngleMin# < 0 THEN CfgAngleMin# = 0
  IF CfgVelMin# < 2 THEN CfgVelMin# = 2
  IF CfgDt# <= 0 THEN CfgDt# = .1
  IF CfgFlushEvery < 1 THEN CfgFlushEvery = 1

  'Animation is a pacing choice, not a rendering one -- the skyline is drawn
  'either way, because POINT reads it for collision. But an unanimated batch
  'run is only ever machine-consumed, so the collision-neutral drawing is
  'pointless there and FastDraw follows Animate by default.
  '
  'Only when FASTDRAW was not given at all, though. Applying this
  'unconditionally silently discarded an explicit FASTDRAW=0 -- which also
  'makes it impossible to run the two halves of the lever-safety comparison
  '(identical seed, drawing on vs off, assert the data matches).
  IF DataMode AND NOT Animate AND FastDrawSet = 0 THEN FastDraw = TRUE

END SUB

'CsvOpenFile:
'  Opens the throw log and writes the header.
'
'  FOR OUTPUT truncates, so the header is always written exactly once and a
'  stale file from a previous run can never be appended to by accident.
'  LogThrow then re-opens FOR APPEND when it flushes.
SUB CsvOpenFile

  OPEN OutFile$ FOR OUTPUT AS #CSVCHAN
  PRINT #CSVCHAN, "board,throw,tosser,tossee,seed,gravity_ms2,buildings,input_mode,flags,";
  PRINT #CSVCHAN, "wind_ms,wind_rel_ms,angle_deg,angle_sim_deg,effort_pct,actual_effort_pct,";
  'drag_k is deliberately absent. It is ~4e-4, so at Fmt4$'s four decimals it
  'would log as 0.0004 -- one significant figure, useless as the cross-check it
  'was meant to be. It is exactly .5*AirDensity#*BananaCd#*area/mass, and both
  'area and mass are logged to full precision, so recompute it downstream.
  PRINT #CSVCHAN, "force_n,velocity_ms,banana_mass_kg,banana_diam_m,banana_area_m2,";
  PRINT #CSVCHAN, "launch_px_x,launch_px_y,target_px_x,target_px_y,ground_px_y,";
  PRINT #CSVCHAN, "launch_height_m,target_height_m,target_dist_m,target_dy_m,";
  PRINT #CSVCHAN, "land_dx_m,land_dy_m,land_px_x,land_px_y,flight_s,outcome,pointval,timer_s"
  CsvIsOpen = TRUE
  RowsSinceFlush = 0
  RowsWritten& = 0

END SUB

'CsvCloseFile:
'  Closes the throw log. QBasic only flushes on CLOSE or END, so this is what
'  makes the last rows reach disk.
SUB CsvCloseFile
  IF CsvIsOpen THEN
    CLOSE #CSVCHAN
    CsvIsOpen = FALSE
  END IF
END SUB

'LogThrow:
'  Appends one row describing the throw that just completed.
'
'  Called from PlayGame, which already has the launch point, the target centre
'  and the relative wind in scope and in metres -- the readouts it draws are
'  effectively a hand-written feature extractor, so this tees them to a file
'  rather than recomputing anything.
'
'  Every CfgFlushEvery rows the file is closed and re-opened FOR APPEND. That
'  is the only way to get QBasic to flush mid-run, and it is what makes a run
'  that later times out still yield usable rows.
SUB LogThrow (Tosser, Tossee, LaunchX#, LaunchY#, DistX#, DistY#, WindRel)

  IF NOT CsvIsOpen THEN EXIT SUB

  'flags packs the switches that change the physics or the RNG stream, so two
  'concatenated runs can never be silently mixed.
  flags = 0
  IF UseEffortJitter THEN flags = flags + 1
  IF UseBananaVar THEN flags = flags + 2
  IF Destruct THEN flags = flags + 4
  IF FastDraw THEN flags = flags + 8
  IF Animate THEN flags = flags + 16

  IF InputMode = IMVELOCITY THEN im$ = "V" ELSE im$ = "E"

  r$ = LTRIM$(STR$(BoardNum)) + "," + LTRIM$(STR$(ThrowNum)) + ","
  r$ = r$ + LTRIM$(STR$(Tosser)) + "," + LTRIM$(STR$(Tossee)) + ","
  r$ = r$ + Fmt4$(CfgSeed#) + "," + Fmt4$(gravity#) + ","
  r$ = r$ + LTRIM$(STR$(LastBuilding)) + "," + im$ + "," + LTRIM$(STR$(flags)) + ","
  r$ = r$ + LTRIM$(STR$(Wind)) + "," + LTRIM$(STR$(WindRel)) + ","
  r$ = r$ + Fmt4$(AngleRaw#) + "," + Fmt4$(AngleSim#) + ","
  r$ = r$ + Fmt4$(EffortReq#) + "," + Fmt4$(EffortAct#) + ","
  r$ = r$ + Fmt4$(ForceUsed#) + "," + Fmt4$(VelocityUsed#) + ","
  r$ = r$ + Fmt4$(BananaMass#) + "," + Fmt4$(BananaDiam#) + "," + Fmt4$(BananaArea#) + ","
  r$ = r$ + Fmt4$(LaunchX#) + "," + Fmt4$(LaunchY#) + ","
  r$ = r$ + LTRIM$(STR$(GorillaCenterX(Tossee))) + "," + LTRIM$(STR$(GorillaCenterY(Tossee))) + ","
  r$ = r$ + LTRIM$(STR$(GroundY)) + ","
  r$ = r$ + Fmt4$((GroundY - LaunchY#) * MetersPerPixel#) + ","
  r$ = r$ + Fmt4$((GroundY - GorillaCenterY(Tossee)) * MetersPerPixel#) + ","
  r$ = r$ + Fmt4$(DistX#) + "," + Fmt4$(DistY#) + ","
  r$ = r$ + Fmt4$(LastShotXm#) + "," + Fmt4$(LastShotYm#) + ","
  r$ = r$ + Fmt4$(LastShotX#) + "," + Fmt4$(LastShotY#) + ","
  r$ = r$ + Fmt4$(LastFlightT#) + ","
  r$ = r$ + LTRIM$(STR$(LastOutcome)) + "," + LTRIM$(STR$(LastPointVal)) + ","
  r$ = r$ + Fmt4$(TIMER)

  PRINT #CSVCHAN, r$
  RowsWritten& = RowsWritten& + 1
  RowsSinceFlush = RowsSinceFlush + 1

  IF RowsSinceFlush >= CfgFlushEvery THEN
    CLOSE #CSVCHAN
    OPEN OutFile$ FOR APPEND AS #CSVCHAN
    RowsSinceFlush = 0
  END IF

END SUB

'SetScreen:
'  Sets the appropriate color statements
SUB SetScreen

  IF Mode = 9 THEN
    ExplosionColor = 2
    BackColor = 1
    PALETTE 0, 1
    PALETTE 1, 46
    PALETTE 2, 44
    PALETTE 3, 54
    PALETTE 5, 7
    PALETTE 6, 4
    PALETTE 7, 3
    PALETTE 9, 63       'Display Color
  ELSE
    ExplosionColor = 2
    BackColor = 0
    COLOR BackColor, 2

  END IF

END SUB

'SparklePause:
'  Creates flashing border for intro and game over screens
SUB SparklePause

  COLOR 4, 0
  A$ = "*    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    "
  WHILE INKEY$ <> "": WEND 'Clear keyboard buffer

  WHILE INKEY$ = ""
    FOR A = 1 TO 5
      LOCATE 1, 1                             'print horizontal sparkles
      PRINT MID$(A$, A, 80);
      LOCATE 22, 1
      PRINT MID$(A$, 6 - A, 80);

      FOR b = 2 TO 21                         'Print Vertical sparkles
        c = (A + b) MOD 5
        IF c = 1 THEN
          LOCATE b, 80
          PRINT "*";
          LOCATE 23 - b, 1
          PRINT "*";
        ELSE
          LOCATE b, 80
          PRINT " ";
          LOCATE 23 - b, 1
          PRINT " ";
        END IF
      NEXT b
    NEXT A
  WEND
END SUB

'UpdateScores:
'  Updates players' scores
'Parameters:
'  Record - players' scores
'  PlayerNum - player
'  Results - results of player's shot
SUB UpdateScores (Record(), PlayerNum, Results)
  IF Results = HITSELF THEN
    Record(ABS(PlayerNum - 3)) = Record(ABS(PlayerNum - 3)) + 1
  ELSE
    Record(PlayerNum) = Record(PlayerNum) + 1
  END IF
END SUB

'VictoryDance:
'  gorilla dances after he has eliminated his opponent
'Parameters:
'  Player - which gorilla is dancing
SUB VictoryDance (Player)

  FOR i# = 1 TO 4
    PUT (GorillaX(Player), GorillaY(Player)), GorL&, PSET
    PLAY "MFO0L32EFGEFDC"
    Rest .2
    PUT (GorillaX(Player), GorillaY(Player)), GorR&, PSET
    PLAY "MFO0L32EFGEFDC"
    Rest .2
  NEXT
END SUB

