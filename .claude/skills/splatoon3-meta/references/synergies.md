# Ability Synergies and Anti-Synergies

## Critical Synergies (Must-Have Pairings)

### Ninja Squid → Swim Speed Up (MANDATORY)
**Requirement**: NS requires SSU ≥1.0 (10 AP)
**Reason**: NS imposes 10% swim speed penalty; SSU offsets this
**Without pairing**: Suboptimal mobility, incomplete build
**Detection**: NS present AND SSU < 10 → flag as `incomplete_build`, `missing_synergy`

---

### カムバゾンビステジャン (Zombie Package)
**Components**: Comeback + Quick Respawn + Stealth Jump
**Synergy strength**: CRITICAL
**Reason**: Complete pressure loop—fast respawn, stat boost on return, safe re-entry
**Labels**: `zombie_package`, `complete_synergy`, `japanese_meta_template`

---

### Anchor Control Package
**Components**: Respawn Punisher + Object Shredder
**Synergy strength**: STRONG
**Reason**: Punish kills + destroy defensive objects = complete backline dominance
**Weapon context**: Chargers, Hydra
**Labels**: `anchor_package`, `backline_dominance`

---

### Special Economy Loop
**Components**: Special Charge Up + Special Saver + (Tenacity optional)
**Synergy strength**: STRONG
**Reason**: Fast charge + death protection = maximize special uptime
**Labels**: `special_economy`, `farm_cycle`

---

## Standard Synergies

| Ability A | Ability B | Relationship | Notes |
|-----------|-----------|--------------|-------|
| Comeback | Quick Respawn | Zombie synergy | Both reward death-accepting play |
| Comeback | Stealth Jump | Re-entry synergy | Safe return + stat boost |
| QR | Stealth Jump | Pressure loop | Fast respawn + safe jump |
| QSJ | Stealth Jump | Jump optimization | Escape + safe aggressive jumps |
| LDE | ISS | Late-game bombs | Stacking efficiency for endgame |
| LDE | Ink Recovery | Late-game sustain | Maximum endgame efficiency |
| Ink Recovery | NS | Swim recovery | More swimming = more recovery opportunity |
| Drop Roller | Inkjet | Landing safety | Roll out of special return |
| Drop Roller | Zipcaster | Landing safety | Roll out of special return |
| RSU | Ink Resistance | Strafe fighting | Move through enemy ink while strafing |
| SPU (Beakon) | ISS | Beacon spam | More beakons + team jump benefit |
| SCU | SPU | Special focus | Fast charge + enhanced special |
| Haunt | Aggressive play | Info advantage | Track flankers for revenge |
| Thermal Ink | Bloblobber | Wall tracking | Tag enemies behind walls |

---

## Critical Anti-Synergies (Never Combine)

### Respawn Punisher + Quick Respawn
**Severity**: CRITICAL CONFLICT
**Reason**: RP reduces QR effectiveness by ~85%
**Philosophy conflict**: RP = "never die" vs QR = "death is acceptable"
**Detection**: RP present AND QR present → flag as `critical_conflict`, `build_error`

---

### Respawn Punisher + Frontline Weapon
**Severity**: STRATEGIC MISMATCH
**Reason**: Self-penalty (+1.13s) exceeds opponent penalty (+0.75s)
**Risk**: Frontline weapons die; RP punishes YOUR deaths more
**Detection**: RP present AND weapon_class in [roller, brush, short_shooter] → flag as `strategic_mismatch`

---

### Respawn Punisher + Special Saver
**Severity**: MODERATE CONFLICT
**Reason**: RP significantly reduces SS effectiveness
**Detection**: RP present AND SS present → flag as `reduced_effectiveness`

---

## Moderate Anti-Synergies

| Conflict | Reason | Severity | Flag |
|----------|--------|----------|------|
| NS + Tacticooler team | Cooler reveals NS users | MECHANIC_OVERRIDE | `tacticooler_reveals_stealth` |
| Thermal Ink + Quick-kill weapon | Tracking useless if target dies | WASTED_SLOT | `tracking_redundant` |
| Opening Gambit + Support play | Aggressive ability on passive role | STRATEGIC_MISMATCH | `ability_role_mismatch` |
| Tenacity + Frontline weapon | Only activates when team down | UNRELIABLE | `inconsistent_activation` |
| High SCU + High death rate | Special Saver more efficient | MISALLOCATION | `efficiency_mismatch` |
| Static ISM/ISS + LDE | LDE replaces need for static investment | REDUNDANCY | `redundant_efficiency` |

---

## Slot Competition Matrices

### Head Slot (Main-Only)
| Ability | Competes With | Resolution Criteria |
|---------|---------------|---------------------|
| Opening Gambit | LDE, Tenacity, Comeback | Choose by death frequency + game phase focus |
| Last-Ditch Effort | Opening Gambit, Tenacity, Comeback | Best for low-death support/painting |
| Tenacity | Opening Gambit, LDE, Comeback | Best for backline anchors |
| Comeback | Opening Gambit, LDE, Tenacity | Best for death-prone frontline |

**Selection heuristic**:
- High deaths expected → Comeback
- Low deaths, late-game focus → LDE
- Anchor, rarely dies → Tenacity
- Organized rush strategy → Opening Gambit

---

### Clothing Slot (Main-Only)
| Ability | Competes With | Resolution Criteria |
|---------|---------------|---------------------|
| Ninja Squid | RP, Thermal, Haunt, Doubler | Best for close-range stealth approach |
| Respawn Punisher | NS, Thermal, Haunt, Doubler | Best for anchors with good K/D |
| Thermal Ink | NS, RP, Haunt, Doubler | Best for artillery/poke weapons |
| Haunt | NS, RP, Thermal, Doubler | Counter-pick vs stealth meta |
| Ability Doubler | All | Rarely optimal; Splatfest only |

**Selection heuristic**:
- Close-range slayer → Ninja Squid
- Backline anchor → Respawn Punisher
- Bloblobber/Stringer → Thermal Ink
- Counter stealth meta → Haunt

---

### Shoes Slot (Main-Only)
| Ability | Competes With | Resolution Criteria |
|---------|---------------|---------------------|
| Stealth Jump | Object Shredder, Drop Roller | Default for frontline (~90% usage) |
| Object Shredder | Stealth Jump, Drop Roller | Tower Control, Rainmaker priority |
| Drop Roller | Stealth Jump, Object Shredder | Inkjet/Zipcaster recall protection |

**Selection heuristic**:
- Frontline, any mode → Stealth Jump
- Tower/Rainmaker, backline → Object Shredder
- Inkjet weapon → Consider Drop Roller

---

## Synergy Chain Analysis

### Complete Zombie Chain
```
Comeback (re-entry stats)
    ↓ synergizes with
Quick Respawn (faster return)
    ↓ synergizes with
Stealth Jump (safe arrival)
    ↓ synergizes with
Quick Super Jump (faster jump)
    ↓ enables
PRESSURE LOOP: Die → Fast respawn → Stat boost → Safe jump → Engage → Repeat
```
**Labels**: `complete_synergy_chain`, `zombie_loop`

---

### Anchor Control Chain
```
Respawn Punisher (punish kills)
    ↓ synergizes with
Object Shredder (destroy defenses)
    ↓ synergizes with
ISM/Ink Recovery (sustain pressure)
    ↓ synergizes with
QSJ (escape when threatened)
    ↓ enables
CONTROL LOOP: Zone → Punish → Destroy objects → Sustain → Escape if needed
```
**Labels**: `anchor_control_chain`, `backline_loop`

---

### Special Farm Chain
```
Special Charge Up (fast charge)
    ↓ synergizes with
Special Saver (death protection)
    ↓ synergizes with
Special Power Up (enhanced special)
    ↓ synergizes with
Tenacity (passive charge when behind)
    ↓ enables
FARM LOOP: Paint → Charge → Use special → If die, preserved gauge → Repeat
```
**Labels**: `special_farm_chain`, `support_loop`

---

## Build Validation Rules

### Critical Validations (Flag as Error)
```python
if has(RP) and has(QR):
    flag("CRITICAL_CONFLICT", "RP_QR_incompatible")

if has(NS) and SSU < 10:
    flag("INCOMPLETE_BUILD", "NS_missing_SSU_offset")

if has(RP) and weapon_role == "frontline":
    flag("STRATEGIC_MISMATCH", "RP_on_frontline")
```

### Warning Validations (Flag for Review)
```python
if has(ThermalInk) and weapon_ttk < 0.3:
    flag("POTENTIAL_WASTE", "tracking_on_quick_kill_weapon")

if SCU > 15 and death_rate > "high":
    flag("EFFICIENCY_CONCERN", "SCU_vs_SS_tradeoff")

if has(Tenacity) and weapon_role == "frontline":
    flag("INCONSISTENT_VALUE", "tenacity_on_aggressive_role")
```

### Positive Validations (Flag as Optimal)
```python
if has(Comeback) and has(QR) and has(StealthJump):
    flag("META_BUILD", "zombie_package_complete")

if QSJ == 3 and SubRes == 3 and InkRes == 3 and SS == 3:
    flag("OPTIMAL_UTILITY", "omamori_package_complete")

if has(NS) and SSU >= 10:
    flag("SYNERGY_COMPLETE", "NS_SSU_pairing_valid")
```

---

## Diminishing Returns Reference

For stacking decisions:

| AP Investment | % of Max Effect | Efficiency Rating |
|---------------|-----------------|-------------------|
| 3 (0.1) | ~10% | HIGHEST |
| 6 (0.2) | ~18% | HIGH |
| 10 (1.0) | ~30% | GOOD |
| 18 (1.6) | ~50% | MODERATE |
| 30 (3.0) | ~70% | LOW |
| 57 (5.7) | 100% | VERY LOW |

**Implication**: First sub of any ability provides disproportionate value. Pure stacking (3+ mains) is almost always suboptimal.

**Exception**: RSU on Splatlings (need 2.0+ for viable strafe)
