# Weapon Archetypes and Build Templates

## Archetype Detection Logic

### Zombie Slayer
**Signature**: QR ≥1.0 + Comeback + Stealth Jump

**Indicators**:
- Quick Respawn ≥12 AP (1.2)
- Comeback (head main)
- Stealth Jump (shoes main)
- Often: SSU for chase, QSJ for escape

**Philosophy**: Death is a resource. Weaponize respawn for relentless pressure.

**Weapons**: Tetra Dualies, Splatana Wiper, aggressive Splattershot variants

**Labels**: `zombie_archetype`, `death_mitigation_hard`, `aggressive_reentry`, `pressure_loop`

---

### Stealth Slayer
**Signature**: Ninja Squid + SSU ≥1.0 + Stealth Jump

**Indicators**:
- Ninja Squid (clothing main)
- Swim Speed Up ≥10 AP (mandatory offset)
- Stealth Jump (shoes main)
- Often: LDE or Comeback

**Philosophy**: Win by getting close unseen. Approach concealment over raw aggression.

**Weapons**: Carbon Roller, Splat Roller, Inkbrush, short-range shooters

**Labels**: `stealth_approach`, `flanker`, `close_range_specialist`

**BUILD CHECK**: NS without SSU ≥1.0 = incomplete build

---

### Anchor/Backline
**Signature**: Respawn Punisher + (Object Shredder OR high ISM)

**Indicators**:
- Respawn Punisher (clothing main)
- Object Shredder OR Stealth Jump (context-dependent)
- ISM for sustain
- QSJ for escape (NOT QR)
- Tenacity or LDE (head)

**Philosophy**: Never die. Make every kill devastating. Control through threat.

**Weapons**: E-liter 4K, Splat Charger, Hydra Splatling, Explosher

**Labels**: `anchor_playstyle`, `backline`, `punishment`, `stay_alive`

**BUILD CHECK**: RP + QR = critical conflict

---

### Splatling Turret
**Signature**: RSU ≥2.0 + Ink Resistance

**Indicators**:
- Run Speed Up ≥20 AP (2.0)
- Ink Resistance Up (often 0.1-0.2)
- LDE or Comeback (head)
- Stealth Jump or Object Shredder (shoes)

**Philosophy**: Mobile turret. Strafe during charge/fire. Outmaneuver while shooting.

**Weapons**: Barrel Splatling, Heavy Splatling, Ballpoint Splatling, Hydra

**Labels**: `splatling_archetype`, `strafe_fighting`, `turret_mobility`

**BUILD CHECK**: RSU <2.0 on Splatling = suboptimal mobility

---

### Support/Beacon Carrier
**Signature**: Sub Power Up + ISS + Comeback

**Indicators**:
- Sub Power Up (benefits team beakon jumps)
- Ink Saver Sub (beacon spam)
- Comeback (head main)
- Stealth Jump (standard)

**Philosophy**: Team utility over personal kills. Beakons as infrastructure.

**Weapons**: Splattery, weapons with Squid Beakon

**Labels**: `support_role`, `team_utility`, `beacon_infrastructure`

---

### Special Farmer
**Signature**: SCU ≥1.0 + Special Saver + Tenacity

**Indicators**:
- Special Charge Up ≥10 AP (1.0)
- Special Saver (often heavy)
- Tenacity or SCU (head)
- Special Power Up (if special scales well)

**Philosophy**: Special weapon is win condition. Maximize uptime and impact.

**Weapons**: Tacticooler carriers (N-ZAP '85), Reef-Lux, special-dependent kits

**Labels**: `special_focused`, `farm_cycle`, `support_anchor`

---

### Bomb Spam / Zoner
**Signature**: ISS ≥1.0 + Ink Recovery + LDE

**Indicators**:
- Ink Saver Sub ≥10 AP (1.0)
- Ink Recovery Up
- Last-Ditch Effort (head)
- Sub Power Up (if bomb scales)

**Philosophy**: Sub weapon as primary tool. Area denial through constant pressure.

**Weapons**: Splattershot Jr., weapons with Splash Bomb/Burst Bomb/Fizzy

**Labels**: `zoner`, `sub_spam`, `area_denial`

---

## Weapon Class → Core Ability Mappings

### Chargers
**Core**: Respawn Punisher, ISM, Ink Recovery, QSJ
**Archetype**: Anchor
**RSU**: Not needed (stationary)
**Death philosophy**: Never die; punish kills

### Splatlings
**Core**: Run Speed Up (2.0+), Ink Resistance, Ink Recovery
**Archetype**: Turret
**RSU**: ESSENTIAL (strafe during charge/fire)
**Death philosophy**: Positioning over aggression

### Rollers (Splat/Carbon)
**Core**: Ninja Squid, Swim Speed Up (1.0+), Stealth Jump
**Archetype**: Stealth Slayer
**SSU**: MANDATORY (offset NS penalty)
**Death philosophy**: Ambush; avoid fair fights

### Brushes
**Core**: Ninja Squid, Swim Speed Up, Quick Respawn
**Archetype**: Zombie Stealth hybrid
**Death philosophy**: Aggressive pressure; death acceptable

### Blasters
**Core**: Intensify Action (1.0+), Ink Resistance, Comeback
**Archetype**: Combat slayer
**IA**: ESSENTIAL (jump accuracy)
**Death philosophy**: Trade-focused; Comeback for recovery

### Shooters (Slayer)
**Core**: Stealth Jump, SSU, Ninja Squid OR Comeback
**Archetype**: Varies by loadout
**IA**: Recommended 0.2 for jump shots
**Death philosophy**: Kit-dependent

### Shooters (Support)
**Core**: SCU, Sub Power Up, LDE
**Archetype**: Support/Farmer
**Death philosophy**: Stay alive; farm specials

### Dualies
**Core**: Intensify Action, Swim Speed Up, Stealth Jump
**Archetype**: Aggressive slayer
**Death philosophy**: Slide-dodge focused; moderate risk

### Sloshers
**Core**: Comeback OR LDE, Stealth Jump, SSU
**Archetype**: Varies
**Death philosophy**: Range-dependent

### Stringers
**Core**: Run Speed Up, Ink Recovery, ISM
**Archetype**: Mid-anchor
**RSU**: Beneficial (strafe while charging)
**Death philosophy**: Positioning; avoid deaths

---

## Template Builds

### Standard Frontline (Japanese Meta)
```
Head:   LDE (Main) + SSU + SubRes + InkRes
Body:   Ninja Squid (Main) + SSU + SSU + SSU
Shoes:  Stealth Jump (Main) + QSJ + SpecialSaver + IA
```
**Total**: 57 AP
**Labels**: `stealth_slayer`, `japanese_meta_standard`, `omamori_package`

---

### Zombie Slayer (カムバゾンビステジャン)
```
Head:   Comeback (Main) + QR + QR + QR
Body:   Ninja Squid (Main) + SSU + SSU + SSU
Shoes:  Stealth Jump (Main) + QR + QR + QR
```
**QR Total**: 18 AP (1.8)
**Labels**: `zombie_archetype`, `aggressive_reentry`, `death_mitigation_hard`

---

### Anchor Charger
```
Head:   Tenacity/LDE (Main) + ISM + ISM + InkRec
Body:   Respawn Punisher (Main) + ISM + InkRec + InkRes
Shoes:  Object Shredder (Main) + QSJ + SpecialSaver + SubRes
```
**Labels**: `anchor_playstyle`, `punishment`, `backline`

---

### Splatling Strafe
```
Head:   LDE (Main) + RSU + RSU + RSU
Body:   RSU (Main) + RSU + RSU + RSU
Shoes:  Stealth Jump (Main) + RSU + RSU + InkRes
```
**RSU Total**: 27 AP (2.7)
**Labels**: `splatling_archetype`, `strafe_fighting`, `turret_mobility`

---

### Omamori Standard (Utility Focused)
```
Sub slots include:
- QSJ (0.1)
- SubRes (0.1)
- InkRes (0.1)
- SpecialSaver (0.1)
```
**Labels**: `omamori_package`, `optimal_utility`, `safety_net`

---

## Archetype Detection Pseudocode

```python
def detect_archetype(build):
    abilities = build.get_abilities()
    
    # Check for critical conflicts first
    if has(RP) and has(QR):
        return "CONFLICT: RP + QR anti-synergy"
    
    if has(NS) and SSU < 10:
        return "INCOMPLETE: NS without SSU offset"
    
    # Archetype detection
    if QR >= 12 and has(Comeback) and has(StealthJump):
        return "zombie_slayer"
    
    if has(NS) and SSU >= 10 and has(StealthJump):
        return "stealth_slayer"
    
    if has(RP) and (has(ObjectShredder) or ISM >= 10):
        return "anchor_backline"
    
    if RSU >= 20:
        return "splatling_turret"
    
    if has(SubPowerUp) and ISS >= 6 and has(Comeback):
        return "support_beacon"
    
    if SCU >= 10 and has(SpecialSaver):
        return "special_farmer"
    
    if ISS >= 10 and has(LDE):
        return "bomb_zoner"
    
    return "hybrid_undefined"
```

---

## Mode-Specific Archetype Shifts

### Tower Control
- Object Shredder value increases (tower interactions)
- Zombie builds more valuable (constant pressure needed)

### Rainmaker
- Object Shredder essential (shield break)
- Respawn Punisher high value (punish Rainmaker carrier deaths)
- Ninja Squid NOT recommended on carrier (visible anyway)

### Splat Zones
- LDE peaks in value (count mechanic triggers)
- Special economy builds stronger (zone defense)
- Object Shredder lower value (fewer objects)

### Clam Blitz
- QSJ especially valuable (jump plays)
- Stealth Jump near-mandatory
- Comeback strong (frequent returns to basket)
