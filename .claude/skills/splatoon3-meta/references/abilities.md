# Splatoon 3 Abilities: Complete Domain Reference

## Standard Abilities (Any-Slot)

---

### Ink Saver (Main) — メイン効率 / ISM

**Standard adjustment**: Weapon-dependent; 0.3-2.0 based on consumption

**Domain relationships**: `ink_economy`, `combo_thresholds`, `sustained_pressure`

**Used-as patterns**:
- Enable specific loops (main → main → sub without emptying)
- Maintain retreat/paint ink after committing
- Value is breakpoint-y, not "stack forever"

**Weapon context**:
- **Required**: Dynamo Roller, E-liter 4K, Camping Shelter, .96 Gal, Explosher
- **Optional**: Most other weapons (LDE often replaces need)

**Synergies**: Ink Recovery Up, Comeback, Last-Ditch Effort
**Redundancy with**: LDE (provides 18 AP equivalent late-game)

**Labels**: `ink_economy`, `weapon_specific`, `breakpoint_dependent`

---

### Ink Saver (Sub) — サブ効率 / ISS

**Standard adjustment**: 0.6-1.3 typical; 2.0+ for bomb-centric

**Domain relationships**: `ink_economy`, `sub_spam`, `zoning`, `double_bomb_threshold`

**Used-as patterns**:
- Double-bomb threshold (consecutive sub throws)
- "Bomb + still fight" ink management
- Primary tool for support kits

**Key threshold**: Splattershot Jr. with ~23 AP can throw two Splat Bombs consecutively

**Synergies**: Ink Recovery Up, Last-Ditch Effort, Sub Power Up
**Redundancy with**: LDE (provides equivalent late-game)

**Labels**: `ink_economy`, `sub_spam`, `zoning`, `breakpoint_dependent`

---

### Ink Recovery Up — インク回復

**Standard adjustment**: 0.1-0.3 (charm gear level)

**Domain relationships**: `ink_economy`, `reset_speed`, `sustained_fights`

**Used-as patterns**:
- Smooth "throw sub → need ink back" cycle
- Stacks with LDE/Comeback windows
- More swimming time with Ninja Squid = more recovery

**Synergies**: ISS, LDE, Ninja Squid, Comeback

**Labels**: `ink_economy`, `reset_speed`, `omamori_adjacent`

---

### Run Speed Up — 人速 / ヒト速 / RSU

**Standard adjustment**: 20-30 AP for Splatlings; minimal for others

**Domain relationships**: `mobility`, `strafe_fighting`, `weapon_specific_necessity`

**Used-as patterns**:
- Strafe during charge/fire (Splatlings)
- "Dazzle" effect on N-ZAP (outwalk aim tracking)
- NOT for travel, FOR combat strafing

**Weapon context**:
- **Essential**: Barrel Splatling, Heavy Splatling, Ballpoint, Hydra (20-30 AP)
- **Good**: N-ZAP series, some midlines
- **Useless**: Chargers (stationary), Rollers (locked animation), Blasters (jump-focused)

**Synergies**: Ink Resistance Up (strafe through enemy ink)

**Labels**: `mobility`, `strafe_fighting`, `weapon_specific`, `splatling_essential`

---

### Swim Speed Up — イカ速 / SSU

**Standard adjustment**: 0.6-1.3 typical; 1.0+ mandatory with Ninja Squid

**Domain relationships**: `mobility`, `universal_utility`, `micro_spacing`, `ninja_squid_offset`

**Used-as patterns**:
- Universal baseline mobility
- Combat strafing (submerge/reposition between shots)
- Offset Ninja Squid's 10% penalty (MANDATORY)

**Weight class tax**: Heavyweights need ~10 AP just to reach Middleweight baseline

**Synergies**: Ninja Squid (MANDATORY pairing)

**Labels**: `mobility`, `universal_utility`, `micro_spacing`, `ninja_squid_offset`

---

### Special Charge Up — スペ増 / SCU

**Standard adjustment**: 0.3-1.0 typical; higher for special-focused builds

**Domain relationships**: `special_economy`, `timing_attacks`, `first_fight_timing`

**Used-as patterns**:
- Special as win condition (farm → push cycle)
- First fight timing (reach mid with special ready)
- Tacticooler spam (~100% uptime)

**Efficiency note**: If dying often, Special Saver more efficient than SCU

**Synergies**: Tenacity, Special Power Up, ink economy abilities

**Labels**: `special_economy`, `timing_attacks`, `tempo`, `support_anchor`

---

### Special Saver — スペ減 / SS

**Standard adjustment**: 3 AP (0.1) — HIGHEST EFFICIENCY INVESTMENT IN GAME

**Domain relationships**: `omamori`, `death_insurance`, `momentum_stabilization`

**Used-as patterns**:
- First sub reduces death loss 50% → 41%
- Enables "Double Special Loop" (die → respawn → paint → special ready)
- Zone control win condition

**The math**: For 200p special, saves 18p (~6-7 shooter shots)

**Heavy investment exception**: Reef-Lux Tenta Missiles builds

**Synergies**: Quick Respawn, Comeback
**Anti-synergies**: Respawn Punisher (reduces effectiveness)

**Labels**: `omamori`, `insurance`, `death_mitigation_soft`, `momentum_stabilization`

---

### Special Power Up — スペ強 / SPU

**Standard adjustment**: 0.3-1.3 when used; highly weapon-specific

**Domain relationships**: `special_economy`, `duration_extension`, `knowledge_check`

**Good scaling**:
- Tacticooler: buff duration (very strong)
- Zipcaster: transformation duration (critical)
- Wave Breaker: radius and HP
- Inkjet: +1 shot at max fire rate

**Bad scaling**:
- Trizooka: minimal (better to get more via SCU)
- Crab Tank: longer = more vulnerable

**Labels**: `special_economy`, `weapon_specific`, `knowledge_check`

---

### Quick Respawn — 復短 / ゾンビ / QR

**Standard adjustment**: 19-26 AP (1.3-2.6) for Zombie builds

**Domain relationships**: `zombie_archetype`, `aggression_insurance`, `temporal_economy`

**CRITICAL CONDITION**: Only activates on consecutive deaths WITHOUT splatting someone

**Used-as patterns**:
- Weaponize failed aggression (dive 1v3, die without kill → 2.8s respawn)
- Risk inversion: take unfavorable fights because penalty is negligible
- Rewards "failing forward"

**Efficiency note**: GP38+ has bad efficiency; GP24-32 optimal range

**Synergies**: Comeback, Stealth Jump, Quick Super Jump (complete package)
**Anti-synergies**: **Respawn Punisher (reduces by 85%)**, getting kills (resets benefit)

**Labels**: `zombie_archetype`, `aggression_insurance`, `death_mitigation_hard`, `risk_budgeting`

**DEATH INCENTIVE**: YES — this ability specifically rewards dying without kills

---

### Quick Super Jump — スパ短 / QSJ

**Standard adjustment**: 3-6 AP (0.1-0.2)

**Domain relationships**: `omamori`, `escape_vector`, `gauge_preservation`

**Used-as patterns**:
- PRIMARY USE IS DEFENSIVE: "Jump Outs" to escape lost fights
- Trizooka interaction: 3 AP lets you enter Flight phase before impact
- Mathematically equivalent to infinite Special Saver if you jump out successfully

**The Trizooka math**: Without QSJ, react to sound → die during Squat. With 3 AP, enter invulnerable Flight milliseconds before impact.

**Synergies**: Stealth Jump (offsets added jump time), Drop Roller

**Labels**: `omamori`, `escape_vector`, `gauge_preservation`, `safe_evacuation`

---

### Sub Power Up — サブ性能 / SPU

**Standard adjustment**: 0.1-1.3 depending on sub weapon

**Domain relationships**: `kit_identity`, `team_utility`, `sub_dependent`

**Effect varies by sub**:
- **High value**: Squid Beakon, Splash Bomb, Burst Bomb, Torpedo
- **Medium**: Curling, Fizzy, Suction, Ink Mine
- **Low**: Sprinkler, Point Sensor

**Team utility**: Beakon SPU reduces jump time FOR TEAMMATES too

**Synergies**: ISS, Ink Recovery Up

**Labels**: `kit_identity`, `team_utility`, `sub_dependent`, `beakon_support`

---

### Ink Resistance Up — 安全靴 / InkRes

**Standard adjustment**: 3 AP (0.1) baseline; 6-10 AP for heavyweights

**Domain relationships**: `omamori`, `mobility_floor`, `dot_mitigation`

**Used-as patterns**:
- Creates ~10 frame delay before damage/slowdown apply
- Prevents "pixel of enemy ink breaks momentum"
- Critical for Squid Roll/Surge that clips enemy ink

**Heavyweight necessity**: Hydra/Dynamo have lower base speed; disproportionately punished

**Synergies**: Run Speed Up, Drop Roller, Ninja Squid approaches

**Labels**: `omamori`, `mobility_floor`, `dot_mitigation`, `tax_sub`

---

### Sub Resistance Up — 爆減 / SubRes

**Standard adjustment**: 3-6 AP (0.1-0.2)

**Domain relationships**: `omamori`, `combo_breaking`, `tracking_reduction`

**Used-as patterns**:
- Breaks lethal combos: 30 dmg splash → 28.4 dmg (survive with 1-2 HP)
- Reduces Point Sensor tracking duration (8s → ~6s)
- Affects SUBS only, NOT specials

**Combo examples broken**:
- Sloshing Machine (76) + Fizzy (35) = 111 → survives
- Splattershot Pro (70) + bomb splash (30) = 100 → survives

**Labels**: `omamori`, `combo_breaking`, `tracking_reduction`, `survivability`

---

### Intensify Action — アク強 / IA

**Standard adjustment**: 6 AP (0.2) for shooters; 10-13 AP for blasters

**Domain relationships**: `mobility`, `rng_mitigation`, `jump_shot_accuracy`

**Used-as patterns**:
- META-DEFINING: Shot spread reduction while jumping
- Turns "lucky" jump shots into skill-based mechanics
- Squid Roll/Surge benefits are secondary

**Weapon breakpoints**:
- **Shooters (Splattershot, .52)**: 6 AP (0.2) = "Golden Ratio"
- **Blasters**: 10-13 AP = ground accuracy while jumping
- **Not needed**: Sharp Marker, H3, Bottle Geyser (no jump spread); Chargers; Dualies (slide-focused)

**Labels**: `mobility`, `rng_mitigation`, `jump_shot_accuracy`, `duel_reliability`

---

## Head-Exclusive Abilities

---

### Opening Gambit — スタダ / スタートダッシュ

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `snowball_opening`, `high_stakes_gambling`, `rush_strategy`

**Used-as patterns**:
- First 30s: Run Speed + Swim Speed + Ink Res + Intensify Action
- Extends 15s per splat/assist (theoretically infinite)
- High risk: if buff expires without kills → dead slot rest of game

**Usage**: Rush strategies, Bamboozler fast picks; NOT preferred in organized play

**Synergies**: Respawn Punisher (coordinated opening pressure)

**Labels**: `snowball_opening`, `high_stakes_gambling`, `tempo`, `rush_strategy`

---

### Last-Ditch Effort — ラスパ / LDE

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `efficiency_dynamic`, `comeback_mechanic`, `snowball_closing`

**RANKED MECHANIC**: Activates at enemy count 50, scales to max at count 30 (not just last 30s)

**Used-as patterns**:
- "Lose-More-to-Win": When losing, weapon becomes hyper-efficient
- Provides ~18 AP of ISM + ISS + Ink Recovery at max
- REPLACEMENT THEORY: Replaces all static ink efficiency investments

**Post-v5.0.0**: Nerfed from 24 AP to 18 AP max

**Comparison**: LDE for low-death support; Comeback for death-prone frontline

**Labels**: `efficiency_dynamic`, `comeback_mechanic`, `snowball_closing`, `late_game_insurance`

---

### Tenacity — 逆境 / 逆境強化

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `special_economy`, `anchor_utility`, `passive`, `inconsistent`

**Used-as patterns**:
- Fills special ONLY when team has fewer alive players
- For anchors who stay alive while teammates die
- Buffed v5.0.0 but still unreliable

**Problem**: If team is even or winning, provides zero value

**Synergies**: SPU, SCU, backline anchors (E-liter, Hydra)

**Labels**: `special_economy`, `anchor_utility`, `passive`, `inconsistent_value`

---

### Comeback — カムバ / カムバック

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `zombie_archetype`, `reentry_tempo`, `stat_compression`

**Used-as patterns**:
- 20s post-respawn: ~10 AP each of ISM, ISS, Ink Recovery, RSU, SSU, SCU
- Called "破格" (exceptional value); ~60% frontline adoption
- "Infinite ink" for 20s retake window

**CRITICAL NUANCE**: Unlike QR, does NOT reward feeding. Rewards CAPITALIZING on 20s window.

**Usage signals**: Slosher 73.4%, Stamper 59.18%

**Synergies**: Quick Respawn, Stealth Jump, QSJ

**Labels**: `zombie_archetype`, `reentry_tempo`, `stat_compression`, `twenty_second_window`

**DEATH INCENTIVE**: NO — rewards surviving after death, not dying

---

## Clothing-Exclusive Abilities

---

### Ninja Squid — イカニン / イカニンジャ

**Standard adjustment**: 10 AP (Main slot only); MUST PAIR WITH SSU ≥1.0

**Domain relationships**: `information_warfare`, `stealth`, `approach_concealment`, `velocity_tax`

**Used-as patterns**:
- Removes swimming splashes/wake
- 10% swim speed penalty (MUST offset with SSU)
- Best on weapons winning by getting close unseen

**Post-v2.1.0**: Speed penalty offset easier; cemented as top-tier

**TACTICOOLER INTERACTION**: Cooler buff REVEALS Ninja Squid users

**Synergies**: Swim Speed Up (MANDATORY ≥1.0)
**Anti-synergies**: Tacticooler team, Rainmaker carrying

**Labels**: `information_warfare`, `stealth`, `approach_concealment`, `velocity_tax`

**BUILD CHECK**: Ninja Squid WITHOUT SSU ≥1.0 = incomplete/suboptimal build

---

### Haunt — リベンジ

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `information_warfare`, `counter_meta`, `anti_flank`, `reactive`

**Used-as patterns**:
- Tracks killer after respawn (thermal outline)
- If you splat tracked opponent → they get Respawn Punisher penalties
- Hybrid: Information + Punishment

**Meta positioning**: Counter-pick tier; devastating vs stealth compositions, useless otherwise

**Labels**: `information_warfare`, `counter_meta`, `revenge_punish`, `anti_flank`, `reactive`

---

### Thermal Ink — サーマル / サーマルインク

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `information_warfare`, `wall_hacks`, `chip_tracking`

**Used-as patterns**:
- Reveals enemies hit by MAIN weapon (not kills, not subs) for 16s
- Distance constraint: must be 3+ lines away to see
- Essentially "wall hacks" for artillery weapons

**Weapon synergy**: Bloblobber, Tri-Stringer (lob shots, tag enemies behind walls)
**NOT recommended**: Blasters (splash doesn't trigger), one-shot weapons (tracking useless)

**Labels**: `information_warfare`, `wall_hacks`, `chip_tracking`, `artillery_support`

---

### Respawn Punisher — ペナアップ / デスペナ / RP

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `punishment`, `asymmetric_warfare`, `anchor_privilege`, `anti_zombie`

**Used-as patterns**:
- Increases respawn time + special loss for you AND victims
- SELF PENALTY LARGER: +1.13s self vs +0.75s victim
- Reduces QR effectiveness by ~85%

**Usage domain**: ANCHORS ONLY (E-liter, Charger, Hydra)—players who rarely die

**Post-v2.1.0**: No longer blocks Tacticooler benefits

**Synergies**: Object Shredder (anchor control package)
**Anti-synergies**: **Quick Respawn (CRITICAL CONFLICT)**, Special Saver, frontline weapons

**Labels**: `punishment`, `asymmetric_warfare`, `anchor_privilege`, `anti_zombie`, `make_kills_hurt`

**BUILD CHECK**: Respawn Punisher + QR = critical conflict / build error

---

### Ability Doubler — 倍化

**Standard adjustment**: 10 AP (Splatfest Tee exclusive)

**Domain relationships**: `splatfest_specific`, `gear_farming`

**Used-as patterns**: Doubles sub ability effects on Splatfest Tee
**Note**: In S3, Festival T main CAN be changed—usually better to change main than use Doubler

**Labels**: `splatfest_specific`, `gear_farming`, `niche`

---

## Shoes-Exclusive Abilities

---

### Stealth Jump — ステジャン / ステルスジャンプ

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `zombie_archetype`, `aggressive_positioning`, `radius_logic`, `mandatory_frontline`

**Used-as patterns**:
- Hides landing marker from enemies outside ~1.5 lines
- SINGLE MOST IMPORTANT ability for team pressure
- Cuts travel time to engagement by ~50%

**Usage signals**: Slosher 93.55%, Stamper 86.46%, ~62% overall

**Misconception**: S3 Stealth Jump does NOT significantly extend jump time (negligible)

**Synergies**: QSJ, Comeback, QR (Zombie package)
**Slot competition**: Drop Roller, Object Shredder

**Labels**: `zombie_archetype`, `aggressive_positioning`, `safe_reentry`, `mandatory_frontline`

---

### Object Shredder — 対物 / 対物攻撃力アップ

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `counter_tech`, `objective_shredding`, `mode_dependent`

**Used-as patterns**:
- Damage bonus to: Crab Tank (~130%), Big Bubbler (~110%), Rainmaker Shield, Beakons, etc.
- Mode-dependent: High value in Tower Control, Rainmaker; Low in Splat Zones

**Example**: Barrel Splatling vs Sprinkler: 4 shots → 1 shot

**Synergies**: Respawn Punisher (anchor package)
**Slot competition**: Stealth Jump

**Labels**: `counter_tech`, `objective_shredding`, `mode_dependent`, `counterpick`

---

### Drop Roller — 受け身 / 受け身術

**Standard adjustment**: 10 AP (Main slot only)

**Domain relationships**: `mobility`, `anti_camp`, `recall_safety`, `checkmate_breaker`

**Used-as patterns**:
- Roll on Super Jump landing + Inkjet/Zipcaster return
- 3s buff: Run Speed + Swim Speed + Ink Resistance
- "Checkmate Breaker" for camped recalls

**Usage**: ~3.75% overall; much rarer than Stealth Jump

**Synergies**: Inkjet weapons, Zipcaster users
**Slot competition**: Stealth Jump (mutually exclusive)

**Labels**: `mobility`, `anti_camp`, `recall_safety`, `checkmate_breaker`
