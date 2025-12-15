# Main Weapon Vibes Reference

> [!CAUTION]
> **Community Consensus — Not Gospel**
> 
> This document represents **aggregated community wisdom** from Sendou.ink build data and competitive player consensus. It is intended to be **directionally correct** and capture the general "vibe" of each weapon — not to be treated as a bible.
> 
> **Use this for:**
> - Understanding why certain abilities cluster around certain weapons
> - Getting a feel for a weapon's intended playstyle
> - Labeling build features with weapon-appropriate context
> - Identifying when a build deviates from typical patterns (which may be innovation OR error)
> 
> **Do NOT use this to:**
> - Declare builds "wrong" — meta evolves, players innovate
> - Ignore kit-specific considerations (sub/special matter!)
> - Override empirical data from actual build statistics

---

## Schema Legend

Each weapon is tagged with parse-friendly tokens for downstream analysis:

| Field | Values | Meaning |
|-------|--------|---------|
| **Ink (FEEL)** | `STARVING` / `HUNGRY` / `AVERAGE` / `EFFICIENT` / `PAINTER` | How ink-hungry the main feels |
| **Ink (ISM)** | `MANDATORY` / `HIGH` / `MED` / `LOW` / `NONE` | Typical Ink Saver Main investment |
| **Move (FEEL)** | `TURRET` / `STIFF` / `STRAFE` / `MOBILE` / `FREE` | Mobility profile while firing |
| **Move (RSU)** | `MANDATORY` / `HIGH` / `MED` / `LOW` / `NONE` | Typical Run Speed Up investment |
| **JumpAim (FEEL)** | `LOTTERY` / `SHAKY` / `OKAY` / `STABLE` / `PERFECT` | Jump-shot accuracy baseline |
| **JumpAim (IA)** | `MANDATORY` / `HIGH` / `MED` / `LOW` / `NONE` | Typical Intensify Action investment |
| **Range** | `MELEE` / `CLOSE` / `MID` / `LONG` / `SNIPER` | Effective engagement range |
| **Approach** | `AMBUSH` / `FLANK` / `FLEX` / `POKE` / `ZONE` | Primary engagement pattern |
| **NinjaSquid** | `CORE` / `GOOD` / `MEH` / `BAD` / `NO` | Stealth-swimming affinity |
| **Role (Lane)** | `FRONT` / `MID` / `BACK` / `FLEX` | Typical positional lane |
| **Role (Job)** | `SLAYER` / `SUPPORT` / `ANCHOR` / `SKIRMISH` / `ASSASSIN` | Primary team function |
| **DeathTol** | `HIGH` / `MED` / `LOW` | How acceptable trading/dying is |

---

## Data Source Notes

Key reconciliations from Sendou.ink build data:

- **Nautilus** is *not* a Ninja Squid weapon: NS appears ~1% on both kits, so stealth is `NO/BAD`, not `CORE`
- **.52 Gal** strongly supports "shaky jump aim + stealth brawler": **IA** and **Ninja Squid** both far above global averages
- **Squeezer** is "execution, not gear crutches": **IA is near-zero** and **ISM is below average** in builds
- **Splattershot Jr.** trends "stay alive + paint," not "trade": **QR very low**, **NS below average**
- **Sploosh** really does lean stealth: NS appears well above global average
- **Splatlings** broadly show "RSU is life" patterns

---

## Shooters

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Sploosh-o-matic | Point-blank shark that wins by surprise and speed. | FEEL=HUNGRY; ISM=MED | FEEL=FREE; RSU=NONE | FEEL=OKAY; IA=LOW | MELEE | AMBUSH | GOOD | LANE=FRONT; JOB=ASSASSIN | HIGH | |
| Splattershot Jr. | Big-tank painter/support that survives and enables. | FEEL=PAINTER; ISM=NONE | FEEL=MOBILE; RSU=LOW | FEEL=STABLE; IA=LOW | CLOSE | FLEX | BAD | LANE=MID; JOB=SUPPORT | LOW | Larger ink tank (main trait). |
| Splash-o-matic | "No excuses" consistent shooter—hits what you aim at. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | FLEX | MEH | LANE=FLEX; JOB=SLAYER | MED | |
| Aerospray | Paint firehose that avoids fair fights and farms space. | FEEL=PAINTER; ISM=NONE | FEEL=FREE; RSU=NONE | FEEL=LOTTERY; IA=LOW | CLOSE | ZONE | NO | LANE=MID; JOB=SUPPORT | MED | |
| Splattershot | True generalist—takes most fights if played cleanly. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=OKAY; IA=MED | MID | FLEX | MEH | LANE=FLEX; JOB=SLAYER | MED | |
| .52 Gal | Two-shot bully: swing corners, punish mistakes, reset often. | FEEL=EFFICIENT; ISM=LOW | FEEL=STIFF; RSU=LOW | FEEL=SHAKY; IA=HIGH | MID | POKE | GOOD | LANE=FRONT; JOB=SLAYER | MED | |
| N-ZAP | Speedy "feet painter" that skirmishes and enables pushes. | FEEL=EFFICIENT; ISM=NONE | FEEL=MOBILE; RSU=LOW | FEEL=OKAY; IA=LOW | MID | FLEX | MEH | LANE=MID; JOB=SUPPORT | MED | |
| Splattershot Pro | Patient poke rifle—wins by spacing and clean confirms. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=MED | FEEL=STABLE; IA=LOW | LONG | POKE | BAD | LANE=MID; JOB=SKIRMISH | MED | |
| .96 Gal | Heavy lane-holder: big threat, slow pivots, hates pressure. | FEEL=HUNGRY; ISM=LOW | FEEL=STIFF; RSU=MED | FEEL=LOTTERY; IA=MED | LONG | ZONE | BAD | LANE=BACK; JOB=ANCHOR | LOW | |
| Jet Squelcher | Safe chip/paint backliner—controls lanes, struggles to finish. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=STABLE; IA=LOW | LONG | ZONE | NO | LANE=BACK; JOB=SUPPORT | LOW | |
| Splattershot Nova | Support painter that annoys and chips more than it duels. | FEEL=PAINTER; ISM=NONE | FEEL=MOBILE; RSU=LOW | FEEL=SHAKY; IA=LOW | LONG | ZONE | NO | LANE=MID; JOB=SUPPORT | MED | |
| L-3 Nozzlenose | Rhythm burst skirmisher—wins by footwork and timing. | FEEL=AVERAGE; ISM=LOW | FEEL=STRAFE; RSU=MED | FEEL=STABLE; IA=LOW | MID | POKE | MEH | LANE=MID; JOB=SKIRMISH | MED | Burst-fire cadence. |
| H-3 Nozzlenose | High-stakes burst pick—misses hurt, hits swing fights. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=MED | FEEL=STABLE; IA=LOW | LONG | POKE | BAD | LANE=MID; JOB=SKIRMISH | LOW | "All-bullets" burst reward. |
| Squeezer | Tap-mode marksman: precision pokes with a spray backup. | FEEL=AVERAGE; ISM=LOW | FEEL=STRAFE; RSU=HIGH | FEEL=PERFECT; IA=NONE | LONG | POKE | BAD | LANE=MID; JOB=SKIRMISH | LOW | Two firing modes (tap pinpoint vs hold spray). |

## Splatlings

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Mini Splatling | Aggro splatling—wins by RSU strafing and tempo. | FEEL=AVERAGE; ISM=LOW | FEEL=STRAFE; RSU=MANDATORY | FEEL=STABLE; IA=LOW | MID | FLEX | NO | LANE=MID; JOB=SKIRMISH | MED | |
| Heavy Splatling | Classic sustained anchor—holds space with long sprays. | FEEL=AVERAGE; ISM=LOW | FEEL=STRAFE; RSU=MANDATORY | FEEL=STABLE; IA=LOW | LONG | ZONE | NO | LANE=BACK; JOB=ANCHOR | LOW | |
| Hydra Splatling | Boss-turret: huge threat, huge commitment, hates dives. | FEEL=HUNGRY; ISM=MED | FEEL=TURRET; RSU=MANDATORY | FEEL=STABLE; IA=LOW | LONG | ZONE | NO | LANE=BACK; JOB=ANCHOR | LOW | |
| Ballpoint Splatling | Technical flex—switches between brawl-mode and beam-mode. | FEEL=HUNGRY; ISM=LOW | FEEL=STRAFE; RSU=MANDATORY | FEEL=STABLE; IA=LOW | LONG | FLEX | NO | LANE=FLEX; JOB=ANCHOR | LOW | Two fire modes (short/long). |
| Nautilus | "Stored charge" splatling—duels midline with strong strafes. | FEEL=AVERAGE; ISM=LOW | FEEL=STRAFE; RSU=HIGH | FEEL=STABLE; IA=LOW | MID | FLEX | NO | LANE=MID; JOB=SKIRMISH | MED | Charge storage changes pacing. |

## Rollers

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Carbon Roller | Fast assassin—pick one target, disappear, repeat. | FEEL=HUNGRY; ISM=LOW | FEEL=FREE; RSU=NONE | FEEL=PERFECT; IA=NONE | CLOSE | AMBUSH | CORE | LANE=FRONT; JOB=ASSASSIN | HIGH | |
| Splat Roller | Iconic shark—threatens corners and punishes oversteps. | FEEL=HUNGRY; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | CLOSE | AMBUSH | CORE | LANE=FRONT; JOB=ASSASSIN | HIGH | |
| Dynamo Roller | Slow artillery—controls zones with huge, committed swings. | FEEL=STARVING; ISM=HIGH | FEEL=TURRET; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | ZONE | NO | LANE=BACK; JOB=ANCHOR | LOW | High-commitment windups. |
| Flingza Roller | Hybrid painter/poker—plays lanes more than stealth kills. | FEEL=AVERAGE; ISM=LOW | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | ZONE | BAD | LANE=MID; JOB=SUPPORT | MED | Distinct horizontal vs vertical feel. |
| Big Swig Roller | Paint specialist—wins by coverage, not duels. | FEEL=PAINTER; ISM=NONE | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | ZONE | NO | LANE=MID; JOB=SUPPORT | MED | |

## Chargers

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Classic Squiffer | Aggro charger—plays midline duels with fast tempo. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | FLEX | BAD | LANE=MID; JOB=SKIRMISH | MED | Fast charge identity. |
| Splat Charger | Standard sniper—wins by sightlines and discipline. | FEEL=AVERAGE; ISM=LOW | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | SNIPER | ZONE | NO | LANE=BACK; JOB=ANCHOR | LOW | |
| Splatterscope | More committed sniper—tunnel vision, stronger hold angles. | FEEL=AVERAGE; ISM=LOW | FEEL=TURRET; RSU=LOW | FEEL=PERFECT; IA=NONE | SNIPER | ZONE | NO | LANE=BACK; JOB=ANCHOR | LOW | |
| E-liter 4K | Extreme-range anchor—punishes movement with hard picks. | FEEL=STARVING; ISM=HIGH | FEEL=TURRET; RSU=LOW | FEEL=PERFECT; IA=NONE | SNIPER | ZONE | NO | LANE=BACK; JOB=ANCHOR | LOW | |
| E-liter 4K Scope | Maximum commitment—pure hold-and-delete turret. | FEEL=STARVING; ISM=HIGH | FEEL=TURRET; RSU=LOW | FEEL=PERFECT; IA=NONE | SNIPER | ZONE | NO | LANE=BACK; JOB=ANCHOR | LOW | |
| Bamboozler 14 | "Rifle charger"—harasses with rapid, repeated shots. | FEEL=AVERAGE; ISM=MED | FEEL=STRAFE; RSU=MED | FEEL=PERFECT; IA=NONE | LONG | POKE | BAD | LANE=MID; JOB=SKIRMISH | MED | No OHKO loop (two-tap pressure). |
| Goo Tuber | Lurker charger—stores charge to ambush or re-peek. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | LONG | AMBUSH | MEH | LANE=FLEX; JOB=ASSASSIN | MED | Charge hold enables sharking. |
| Snipewriter | Support sniper—multi-shot pressure that manages space. | FEEL=AVERAGE; ISM=LOW | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | SNIPER | ZONE | NO | LANE=BACK; JOB=SUPPORT | LOW | Multi-shot per charge. |

## Blasters

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Luna Blaster | Corner assassin—gets close, explodes you, escapes. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=SHAKY; IA=HIGH | CLOSE | AMBUSH | GOOD | LANE=FRONT; JOB=ASSASSIN | HIGH | |
| Blaster | Midline jumper—plays ledges for directs/indirects. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=SHAKY; IA=HIGH | MID | POKE | MEH | LANE=MID; JOB=SLAYER | MED | |
| Range Blaster | Elevation zoner—slow, lethal, and IA-hungry. | FEEL=HUNGRY; ISM=MED | FEEL=TURRET; RSU=LOW | FEEL=SHAKY; IA=HIGH | LONG | ZONE | BAD | LANE=BACK; JOB=SKIRMISH | LOW | |
| Clash Blaster | Disruption spam—wins by chaos, not precision. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=OKAY; IA=LOW | CLOSE | FLEX | MEH | LANE=FRONT; JOB=SKIRMISH | HIGH | |
| Rapid Blaster | Tempo poker—reliable pressure from safer ranges. | FEEL=AVERAGE; ISM=LOW | FEEL=STIFF; RSU=LOW | FEEL=OKAY; IA=MED | LONG | POKE | BAD | LANE=MID; JOB=SKIRMISH | MED | |
| Rapid Blaster Pro | Long-range denial—plays slow and punishes peeks. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=OKAY; IA=MED | LONG | ZONE | NO | LANE=BACK; JOB=SKIRMISH | LOW | |
| S-BLAST '92 | Two-mode blaster—ground for range, jump for big close threat. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=OKAY; IA=MED | LONG | FLEX | MEH | LANE=MID; JOB=SLAYER | MED | Ground vs jump mode split. |

## Sloshers

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Slosher | Angle bully—hits over cover and forces awkward fights. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | FLEX | MEH | LANE=MID; JOB=SKIRMISH | MED | |
| Tri-Slosher | Close brawler—wide hits, fast pressure, messy fights. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | CLOSE | AMBUSH | GOOD | LANE=FRONT; JOB=SLAYER | HIGH | |
| Sloshing Machine | Precision bucket—slower, harsher punishes, strong angles. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | POKE | BAD | LANE=MID; JOB=SLAYER | MED | |
| Bloblobber | Bubble zoner—wins by denying space, not chasing. | FEEL=AVERAGE; ISM=LOW | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | LONG | ZONE | NO | LANE=BACK; JOB=SUPPORT | LOW | Predictive ricochet pressure. |
| Explosher | Artillery anchor—slow, expensive shots that own areas. | FEEL=STARVING; ISM=HIGH | FEEL=TURRET; RSU=LOW | FEEL=PERFECT; IA=NONE | LONG | ZONE | NO | LANE=BACK; JOB=ANCHOR | LOW | |
| Dread Wringer | Double-tap slosher—midline pressure with a steady rhythm. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | FLEX | MEH | LANE=MID; JOB=SKIRMISH | MED | Two-hit cadence. |

## Dualies

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Dapple Dualies | Knife-fight dualies—roll in, delete, or explode. | FEEL=AVERAGE; ISM=LOW | FEEL=FREE; RSU=LOW | FEEL=OKAY; IA=LOW | MELEE | AMBUSH | CORE | LANE=FRONT; JOB=ASSASSIN | HIGH | |
| Splat Dualies | All-round rolls—works everywhere if you manage commits. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=OKAY; IA=LOW | MID | FLEX | MEH | LANE=FLEX; JOB=SKIRMISH | MED | |
| Glooga Dualies | High-commit turret rolls—roll = "fight to the end." | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=MED | FEEL=STABLE; IA=LOW | MID | POKE | BAD | LANE=MID; JOB=SLAYER | MED | Strong post-roll mode. |
| Dualie Squelchers | Slide skirmisher—pokes safely and refuses hard commits. | FEEL=HUNGRY; ISM=MED | FEEL=MOBILE; RSU=LOW | FEEL=STABLE; IA=LOW | LONG | POKE | NO | LANE=MID; JOB=SKIRMISH | LOW | Post-roll mobility identity. Ink-hungry due to constant poke/slide pressure. |
| Tetra Dualies | Chaos engager—creates openings by being hard to pin down. | FEEL=HUNGRY; ISM=MED | FEEL=FREE; RSU=LOW | FEEL=OKAY; IA=LOW | MID | FLANK | GOOD | LANE=FRONT; JOB=SKIRMISH | HIGH | 4-roll pressure loop. |
| Douser Dualies | Long-range "soft backliner" with limited escape. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=MED | FEEL=STABLE; IA=LOW | LONG | POKE | NO | LANE=BACK; JOB=SUPPORT | LOW | 1-roll pacing. |

## Brellas

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Splat Brella | Skirmish shield—wins by timing canopy + close confirms. | FEEL=AVERAGE; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=OKAY; IA=LOW | CLOSE | FLEX | MEH | LANE=MID; JOB=SKIRMISH | MED | Shield timing is core loop. |
| Tenta Brella | Front tank—pushes space with shield-first commitment. | FEEL=HUNGRY; ISM=MED | FEEL=TURRET; RSU=LOW | FEEL=OKAY; IA=LOW | CLOSE | ZONE | BAD | LANE=FRONT; JOB=SUPPORT | HIGH | Big shield "escort" identity. |
| Undercover Brella | Harasser—stays annoying under fire, rarely hard-wins duels. | FEEL=EFFICIENT; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=OKAY; IA=LOW | CLOSE | FLANK | MEH | LANE=FRONT; JOB=SKIRMISH | HIGH | Fires while shielded. |
| Recycled Brella | Aggro brella—plays burst windows and shield throws. | FEEL=HUNGRY; ISM=MED | FEEL=MOBILE; RSU=LOW | FEEL=OKAY; IA=LOW | CLOSE | FLANK | MEH | LANE=FRONT; JOB=SLAYER | MED | Shield is more disposable/offensive. |

## Brushes

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Inkbrush | Pure disruption—runs routes, touches points, forces turns. | FEEL=EFFICIENT; ISM=LOW | FEEL=FREE; RSU=NONE | FEEL=PERFECT; IA=NONE | MELEE | FLANK | GOOD | LANE=FRONT; JOB=SKIRMISH | HIGH | Extreme mobility identity. |
| Octobrush | Ambush brush—slower but scarier close engagements. | FEEL=AVERAGE; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | CLOSE | AMBUSH | CORE | LANE=FRONT; JOB=ASSASSIN | HIGH | |
| Painbrush | Heavy brush—committed swings that win space, not chases. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | POKE | MEH | LANE=MID; JOB=SLAYER | MED | Slow windup "hammer" feel. |

## Stringers

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Tri-Stringer | Area-denial archer—controls landings with delayed damage. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | LONG | ZONE | NO | LANE=BACK; JOB=SUPPORT | LOW | Explosive delay is the "vibe." |
| REEF-LUX 450 | Mobile snap bow—paints and skirmishes on the move. | FEEL=PAINTER; ISM=LOW | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | FLEX | MEH | LANE=MID; JOB=SUPPORT | MED | Charge storage changes peeks. |
| Wellstring V | Heavy spread bow—wins by covering zones, not duels. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | LONG | ZONE | NO | LANE=MID; JOB=SUPPORT | LOW | Multi-arrow "shotgun bow." |

## Splatanas

| Weapon | Vibe | Ink (FEEL;ISM) | Move (FEEL;RSU) | JumpAim (FEEL;IA) | Range | Approach | NinjaSquid | Role (Lane;Job) | DeathTol | Notes |
|--------|------|----------------|-----------------|-------------------|-------|----------|------------|-----------------|----------|-------|
| Splatana Stamper | Midline duelist—threatens burst combos and angles. | FEEL=HUNGRY; ISM=MED | FEEL=MOBILE; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | FLANK | MEH | LANE=MID; JOB=SLAYER | MED | Charged slash "commit" moments. |
| Splatana Wiper | Hyper skirmisher—chips, kites, and re-engages nonstop. | FEEL=AVERAGE; ISM=LOW | FEEL=FREE; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | FLANK | GOOD | LANE=FRONT; JOB=SKIRMISH | HIGH | |
| Decavitator | Burst assassin—big threats in short windows, hates being seen early. | FEEL=HUNGRY; ISM=MED | FEEL=STIFF; RSU=LOW | FEEL=PERFECT; IA=NONE | MID | AMBUSH | GOOD | LANE=FRONT; JOB=ASSASSIN | HIGH | Heavy, window-based pressure. |

---

## Usage in Labeling

When labeling SAE features with weapon context:

```
Feature: High RSU activation on Nautilus build
Label: weapon_appropriate, splatling_strafe_standard, expected_pattern

Feature: Ninja Squid on Splat Charger
Label: weapon_mismatch, backline_no_ns, investigate_kit_reason

Feature: IA=HIGH on .52 Gal
Label: weapon_appropriate, shaky_jump_compensation, community_standard

Feature: IA=HIGH on Squeezer
Label: weapon_deviation, perfect_aim_weapon, unusual_investment
```

## Quick Lookup by Trait

### Weapons that want Ninja Squid (CORE/GOOD)
- **CORE**: Carbon Roller, Splat Roller, Octobrush, Dapple Dualies
- **GOOD**: Sploosh, .52 Gal, Luna Blaster, Tri-Slosher, Tetra Dualies, Inkbrush, Splatana Wiper, Decavitator

### Weapons that need RSU (MANDATORY/HIGH)
- **MANDATORY**: All Splatlings (Mini, Heavy, Hydra, Ballpoint)
- **HIGH**: Nautilus, Squeezer

### Weapons that want Intensify Action (HIGH)
- .52 Gal, Luna Blaster, Blaster, Range Blaster

### Death-Tolerant Weapons (HIGH)
- Sploosh, Carbon Roller, Splat Roller, Luna Blaster, Clash Blaster, Tri-Slosher, Dapple Dualies, Tetra Dualies, Tenta Brella, Undercover Brella, Inkbrush, Octobrush, Splatana Wiper, Decavitator

### Pure Backline / Anchor (LOW DeathTol)
- .96 Gal, Jet Squelcher, Heavy Splatling, Hydra Splatling, Dynamo Roller, All Chargers (except Squiffer/Bamboozler), Range Blaster, Rapid Blaster Pro, Bloblobber, Explosher, Dualie Squelchers, Douser Dualies, Tri-Stringer, Wellstring V
