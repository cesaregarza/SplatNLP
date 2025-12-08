# Splatoon 3 Competitive Meta: Consolidated Semantic Labeling Reference

**Version**: 1.0 (Consolidated)  
**Date**: December 7, 2025  
**Purpose**: SAE Feature Labeling System for Gear Build Pattern Recognition

---

## Document Overview

This consolidated reference synthesizes three independent research efforts into the Splatoon 3 competitive meta, prioritizing Japanese competitive sources (wikiwiki.jp/splatoon3mix, note.com pro blogs, Game8.jp, GameWith.jp) with verification from Sendou.ink statistics.

**Core Principle**: When the SAE discovers a feature pattern like "High ISS + Fizzy Bomb co-activation," this document provides the semantic reasoning (Fizzy is spam-viable; ISS enables double-throw) and appropriate labels (`sub_spam`, `fizzy_zoning`, `aggressive_efficiency`).

---

# SECTION 1: Special Weapon Philosophy Classification

Special weapons drive primary build philosophy. The **gauge lock mechanic** is the single most important factor—specials with gauge locks nullify Special Saver entirely, while high-commitment specials make Special Saver mandatory.

## 1.1 Category Definitions

| Category | Definition | Primary Build Signal | Secondary Signal |
|----------|------------|---------------------|------------------|
| **Spam** | Value from frequency; low individual impact | SCU 1.0+ | Skip Special Saver |
| **Conserve** | High impact, timing-critical; "win conditions" | Special Saver 0.1-0.6 | Moderate SCU |
| **Burst** | Aggressive immediate use; high-risk activation | QR + Mobility | Special Saver secondary |
| **Team** | Value scales with team uptime/coordination | SPU + SCU stack | Max uptime focus |
| **Reactive** | Defensive/counter tools | Special Saver + Survival | Timing-critical |

---

## 1.2 Complete Special Classification

### SPAM SPECIALS

#### Tenta Missiles (マルチミサイル)
**Category**: Spam / Global Displacement  
**Build Philosophy**: Map-wide pressure through frequency; value = activations per match. "回す (rotate) to keep pressure" is the JP framing—missiles as tempo tax, not held resource.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 1.0-1.3 | **CORE** | More missiles = more value |
| SPU | 0.1-0.6 | Medium | Lock-on coverage, paint parameters |
| Special Saver | 0.0-0.1 | LOW | Cycling constantly makes saver inefficient |

**Semantic Labels**: `missile_farmer`, `global_displacement`, `special_spam`  
**Anti-Synergies**: Heavy SS with low SCU (reduces cast frequency)  
**Gauge Lock**: 10 sec fixed—SPU does NOT add missiles  
**Pro Sentiment**: Effective only if producing 6+ activations per match

---

#### Ink Storm (アメフラシ)
**Category**: Spam / Area Control  
**Build Philosophy**: Throw-and-forget; storm persists after death. JP wiki calls it 出し得 (always worth casting). Use as "soft checkmate"—storm forces movement, then win the next duel.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 0.6-1.3 | **CORE** | More storms = more control |
| SPU | 0.1-0.6 | Medium | Coverage/duration |
| Special Saver | 0.0-0.1 | LOW | Storm remains after death |

**Semantic Labels**: `storm_spam`, `area_denial`, `throw_and_forget`  
**Gauge Lock**: 8 sec  
**Anti-Synergies**: Heavy SS over SCU

---

#### Triple Inkstrike (トリプルトルネード)
**Category**: Spam / Displacement  
**Build Philosophy**: Space denial + paint swing; "塗りと分断" (paint + split) on cooldown. Especially strong in Splat Zones.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 0.6-1.3 | **CORE** | Strike frequency |
| SPU | 0.1-0.6 | Medium | Throw distance/placement |
| Special Saver | 0.0-0.1 | LOW | Often cast then fight |

**Semantic Labels**: `siege_artillery`, `zone_denial`, `inkstrike_cycle`  
**Anti-Synergies**: Overbuilding SS instead of SCU

---

#### Super Chump (デコイチラシ)
**Category**: Spam / Attention Split  
**Build Philosophy**: Paint + decoy pressure; value from forcing enemies to choose between shooting decoys or losing space. SPU increases blast/paint ranges (~1.2×).

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 0.6-1.3 | **CORE** | Decoy frequency |
| SPU | 0.1-0.6 | Medium | Coverage improvement |
| Special Saver | 0.0-0.1 | LOW | Low commitment special |

**Semantic Labels**: `decoy_pressure`, `attention_split`, `paint_chaos`  
**Anti-Synergies**: Expecting chumps to get kills alone

---

#### Killer Wail 5.1 (メガホンレーザー5.1ch)
**Category**: Spam / Neutral Control  
**Build Philosophy**: Frequent pressure + pseudo-tracking to force dodges. "Neutral-win tool"—pop it to start advantage, not as miracle save.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 0.6-1.3 | **CORE** | Wail tempo |
| Special Saver | 0.1 | Medium | If dying right after firing |
| SPU | 0.0-0.3 | LOW | Most value is getting it again |

**Semantic Labels**: `wail_tempo`, `neutral_control`, `movement_forcer`  
**Anti-Synergies**: Heavy SPU at cost of SCU

---

#### Wave Breaker (ホップソナー)
**Category**: Spam → Team Hybrid  
**Build Philosophy**: Support special—chip damage + marking + movement constraints. "置き所とタイミングが命" (placement/timing is everything). Best as fight amplifier, not finisher.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SPU | 0.1-0.6 | **HIGH** | Max +35% wave radius |
| SCU | 0.3-0.8 | Medium | Constant fight-tax |
| Special Saver | 0.1 | Medium | Backline sonar protection |

**Semantic Labels**: `sonar_support`, `fight_amplifier`, `movement_tax`  
**Note**: Forces jumping; synergizes with IA awareness (enemies suffer jump-shot spread)  
**Anti-Synergies**: Treating as "guaranteed clearing"

---

### CONSERVE SPECIALS

#### Trizooka (ウルトラショット)
**Category**: Conserve / Pick Tool  
**Build Philosophy**: Pick tool + displacement; converts timing window into 1-2 kills. SPU buys better windows, not more shots. **SPU doesn't add shots**—only extends activation time + blast radius.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Special Saver | 0.1-0.3 | **CORE** | Dying with Zooka is catastrophic |
| SCU | 0.3-1.0 | HIGH | Kit-dependent |
| SPU | 0.1-0.6 | Situational | Time/radius, not DPS |

**Semantic Labels**: `pick_generator`, `sniper_counter`, `timing_conserve`  
**Dead Zone**: 0.1-1.0 SS often shows—either minimal omamori or committed conservation  
**Anti-Synergies**: Overstacking SPU on "shoot 3 fast and leave" playstyles  
**Synergy**: Often paired with Comeback on frontline weapons

---

#### Crab Tank (カニタンク)
**Category**: Conserve / Win Condition  
**Build Philosophy**: Premier pushing tool; commitment-heavy and punishable. Use for "structured advantage" (守る/抑える/打開に使う = defend/hold/breakthrough). JP explicitly warns: you're slow and can't cancel.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Special Saver | 0.2-0.6 | **CORE** | Dying with Crab is brutal |
| SCU | 0.3-0.8 | HIGH | Hit key timings |
| SPU | 0.1-0.6 | Purposeful | Duration with intent |

**Semantic Labels**: `win_condition`, `tank_pivot`, `structured_push`  
**Dead Zone**: SS between 0.1 and 1.0—competitive builds run either omamori or 1.0+ for 50%+ retention  
**Anti-Synergies**: Max SPU without positioning discipline  
**Counter**: Object Shredder deals 1.1× to Crab armor

---

#### Inkjet (ジェットパック)
**Category**: Conserve / Pick Potential  
**Build Philosophy**: High pick potential but punishable. "Angle economy"—best jets are ones that don't get traded. Mix B-button hover with ink descent.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Special Saver | 0.2-0.6 | **CORE** | Dying in jet is common |
| SCU | 0.3-0.8 | HIGH | Hit right windows |
| SPU | 0.1-0.6 | HIGH | Duration + fire rate if living |

**Semantic Labels**: `pick_potential`, `angle_economy`, `jet_slayer`  
**Anti-Synergies**: Overstack SPU without escape plan

---

#### Ink Vac (キューインキ)
**Category**: Reactive / Conserve Hybrid  
**Build Philosophy**: Defensive counter; "turn their push into yours." Timing-critical—use to delete enemy bombs/specials and swing a push.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Special Saver | 0.1-0.3 | **CORE** | Vac is specific answer; losing hurts |
| SCU | 0.2-0.6 | Medium | How often you need the answer |
| SPU | 0.1-0.6 | Medium | Vacuum parameters |

**Semantic Labels**: `projectile_eater`, `bodyguard`, `counter_pivot`  
**Anti-Synergies**: Full SCU "spam Vac" on kits that can't safely convert

---

### BURST SPECIALS

#### Zipcaster (ショクワンダー)
**Category**: Burst / Flank  
**Build Philosophy**: Mobility/assassination window. Pop → create chaos → force trades or picks. "Zip to create inevitable crossfire," not to hero. Recall mechanic often leads to death.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Special Saver | 0.3-0.6 | **CORE** | Zip plays = high mortality |
| QSJ | 0.1-0.3 | HIGH | Post-zip exit + bailout |
| SCU | 0.3-0.8 | Medium | Hit engage timings |
| ISM | Situational | Medium | Ink depletes during Zip |

**Semantic Labels**: `entry_fragger`, `backline_harasser`, `formation_breaker`  
**Anti-Synergies**: Over-greeding SCU while skipping survivability

---

#### Reefslider (サメライド)
**Category**: Burst / Commit  
**Build Philosophy**: Decisive entry, panic button, or trade-for-space. Known as "Suicide Button" due to predictable end-lag.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| QR | Essential | **CORE** | Slider deaths happen |
| Special Saver | 0.2-0.6 | HIGH | Backup insurance |
| SCU | 0.3-0.8 | Medium | More attempts |
| SPU | 0.1-0.3 | LOW | Radius/paint breakpoints |

**Semantic Labels**: `objective_diver`, `trade_forcer`, `commit_entry`  
**Anti-Synergies**: Greedy SPU expecting to fix bad positioning  
**Note**: SPU rarely used; extending slide duration increases vulnerability

---

#### Kraken Royale (テイオウイカ)
**Category**: Burst / Conserve Hybrid  
**Build Philosophy**: Entry + invuln pressure; "space first, kills second." Convert panic into objective gain. SPU extends duration up to 10s.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Special Saver | 0.2-0.6 | **CORE** | High-variance engages |
| SCU | 0.3-0.8 | HIGH | Key timings |
| SPU | 0.1-0.6 | Situational | Duration only |

**Semantic Labels**: `objective_cheese`, `panic_button`, `invuln_entry`  
**Anti-Synergies**: Ignoring SS on "kraken-in" kits  
**Note**: Tier X for Tower Control; charge attack uncounterable

---

#### Triple Splashdown (ウルトラチャクチ)
**Category**: Burst / Reactive Hybrid  
**Build Philosophy**: Panic/trade tool + punish over-extensions. "I refuse to lose this space" rather than "I will solo wipe." Usable during Super Jump for surprise plays.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Special Saver | 0.3-0.6 | **CORE** | Splashdown users die by design |
| QSJ | 0.1-0.3 | HIGH | Jump plays + bailout |
| SCU | 0.2-0.6 | Medium | Moderate investment |

**Semantic Labels**: `panic_counter`, `close_range_nuke`, `trade_tool`  
**Anti-Synergies**: Full SCU with no SS

---

### TEAM SPECIALS

#### Tacticooler (エナジースタンド)
**Category**: Team / Stat Buff  
**Build Philosophy**: THE meta-defining special. Teamwide buff uptime is the win condition. "置ければ価値" (placing it is already value)—optimize for getting it again. JP verification shows drink duration scales to **25s** with full SPU.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 0.6-1.3 | **CORE** | Cooler uptime |
| SPU | 0.3-1.0 | **CORE** | Buff duration for entire team |
| Special Saver | 0.0-0.1 | LOW | Value extracted even if you die after placing |

**Semantic Labels**: `cooler_bot`, `team_engine`, `cooler_loop`  
**"Cooler Loop"**: ~20 AP SPU + ~20 AP SCU = fresh Cooler when previous expires  
**Anti-Synergies**: Heavy SS at cost of SCU/SPU  
**Gauge Lock**: 10 sec

---

#### Big Bubbler (グレートバリア)
**Category**: Team / Area Denial  
**Build Philosophy**: Objective fort + push anchor. Includes jump beacon function. SPU increases durability but **gauge lock extends too** (up to 16.67s), so rotation slightly worsens.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 0.3-1.0 | **CORE** | Bubble timings |
| Special Saver | 0.1-0.3 | HIGH | Frontline bubble protection |
| SPU | 0.1-0.6 | Trade-off | Durability vs rotation |

**Semantic Labels**: `checkpoint_anchor`, `jump_beacon`, `objective_fort`  
**Critical Insight**: SPU is "good but not free"—optimize for either stronger bubble OR more bubbles, not both  
**Counter**: Object Shredder destroys Bubbler ~30% faster

---

#### Booyah Bomb (ナイスダマ)
**Category**: Team / Spam Hybrid  
**Build Philosophy**: Objective/control special scaling with team presence. "打開/抑え" (break/hold) timing—don't toss when it won't move count. SPU increases auto-charge speed + armor durability.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 0.3-1.0 | **CORE** | More bombs = more swings |
| Special Saver | 0.1-0.3 | HIGH | Dying with Booyah is painful |
| SPU | 0.1-0.6 | Medium | Armor + charge speed |

**Semantic Labels**: `zone_stall`, `armor_pivot`, `break_hold_timing`  
**Anti-Synergies**: All-in SCU on kits that can't paint enough to cycle

---

#### Splattercolor Screen (スミナガシート)
**Category**: Team / Vision Denial  
**Build Philosophy**: Vision denial + formation split. "Call-to-action" special—value decays if you don't act after placement.

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 0.3-1.0 | HIGH | Screen frequency |
| SPU | 0.1-0.6 | Medium | Deployment/movement |
| Special Saver | 0.1-0.3 | Medium | Midline user protection |

**Semantic Labels**: `visual_disruptor`, `formation_split`, `push_enabler`  
**Anti-Synergies**: Pure spam without follow-up push

---

## 1.3 Special Classification Quick Reference

| Special | Category | SCU Priority | SS Priority | SPU Priority | Gauge Lock |
|---------|----------|--------------|-------------|--------------|------------|
| Tenta Missiles | Spam | **1.0+** | 0.0-0.1 | 0.1-0.6 | 10s |
| Ink Storm | Spam | **0.6-1.3** | 0.0-0.1 | 0.1-0.6 | 8s |
| Triple Inkstrike | Spam | **0.6-1.3** | 0.0-0.1 | 0.1-0.6 | Variable |
| Super Chump | Spam | **0.6-1.3** | 0.0-0.1 | 0.1-0.6 | Variable |
| Killer Wail 5.1 | Spam | **0.6-1.3** | 0.1 | 0.0-0.3 | Variable |
| Wave Breaker | Spam/Team | 0.3-0.8 | 0.1 | **0.1-0.6** | 6s |
| Trizooka | Conserve | 0.3-1.0 | **0.1-0.3** | 0.1-0.6 | — |
| Crab Tank | Conserve | 0.3-0.8 | **0.2-0.6** | 0.1-0.6 | — |
| Inkjet | Conserve | 0.3-0.8 | **0.2-0.6** | 0.1-0.6 | — |
| Ink Vac | Reactive | 0.2-0.6 | **0.1-0.3** | 0.1-0.6 | — |
| Zipcaster | Burst | 0.3-0.8 | **0.3-0.6** | — | — |
| Reefslider | Burst | 0.3-0.8 | 0.2-0.6 | 0.1-0.3 | — |
| Kraken Royale | Burst | 0.3-0.8 | **0.2-0.6** | 0.1-0.6 | — |
| Triple Splashdown | Burst/Reactive | 0.2-0.6 | **0.3-0.6** | — | — |
| Tacticooler | Team | **0.6-1.3** | 0.0-0.1 | **0.3-1.0** | 10s |
| Big Bubbler | Team | **0.3-1.0** | 0.1-0.3 | 0.1-0.6 | Up to 16.67s |
| Booyah Bomb | Team | **0.3-1.0** | 0.1-0.3 | 0.1-0.6 | — |
| Splattercolor Screen | Team | 0.3-1.0 | 0.1-0.3 | 0.1-0.6 | — |

---

# SECTION 2: Sub Weapon Build Implications

Sub weapons generate the strongest build signals. A player's sub choice reveals intended playstyle more reliably than any other kit component.

## 2.1 Spam Viability Tiers

### HIGH SPAM VIABILITY
Low ink cost, enabling frequent usage without heavy ISS investment.

| Sub | Ink Cost | Native Doubles | ISS for 2× | Playstyle Signal |
|-----|----------|----------------|------------|------------------|
| **Burst Bomb** | 40% | YES (80%) | GP0 | `aggressive_combo`, chip + combo starter |
| **Angle Shooter** | 40% | YES (80%) | GP0* | `information_poke`, precision tag |
| **Point Sensor** | 45% | YES (90%) | GP0 | `team_support`, marking priority |
| **Fizzy Bomb** | 60% | NO | GP21 | `turf_pressure_hybrid`, variable bounce |

*ISS NOT recommended for Angle Shooter (max 20% reduction)

### MEDIUM SPAM VIABILITY
Require moderate ISS (GP12-25) for comfortable usage.

| Sub | Ink Cost | ISS for 2× | Key Abilities | Playstyle Signal |
|-----|----------|------------|---------------|------------------|
| **Autobomb** | 55% | **GP12** | ISS, IRU | `passive_disruption`, most ISS-efficient |
| **Ink Mine** | 60% | GP15 | SPU, IRU | `defensive_anchor`, flank protection |
| **Splash Wall** | 60% | GP21 | ISS, **SPU** (HP) | `positional_control`, shield-based |
| **Toxic Mist** | 60% | GP21 | ISS, IRU | `choke_control`, Rainmaker-relevant |
| **Torpedo** | 65% | ~GP29-35 | ISS, IRU | `passive_harassment`, autonomous tracking |

### LOW SPAM VIABILITY
Heavy ISS investment (GP29+) required; signals lethal bomb focus.

| Sub | Ink Cost | ISS for 2× | Key Abilities | Playstyle Signal |
|-----|----------|------------|---------------|------------------|
| **Splat Bomb** | 70% | GP35 | ISS, IRU, **LDE** | `versatile_aggression`, standard offensive |
| **Suction Bomb** | 70% | GP35 | ISS, IRU | `delayed_area_control`, defensive zoning |
| **Curling Bomb** | 70% | ~GP29-35 | SSU, NS | `flanker_mobility`, ink path creation |
| **Squid Beakon** | 75% | GP36 | **SPU**, QSJ | `macro_support`, team logistics |
| **Sprinkler** | — | — | IRU, SCU | `paint_support`, farming tool |

---

## 2.2 Detailed Sub Weapon Profiles

### Splat Bomb (スプラッシュボム)
**Spam Viability**: MEDIUM — strong but expensive  
**Ink Cost**: 70%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.1-0.3 | Comfort | Reduce "one bomb empties me" |
| IRU | 0.1-0.3 | Support | Recovery after throws |
| LDE | 0.1 | **HIGH** | Enables late-game double bomb |

**ISS Breakpoints**:
- Comfort spam: 0.1-0.3
- True double: ~GP35 (impractical without LDE)
- **Wakaba exception**: 110% tank enables double at only GP23

**Semantic Labels**: `generalist_pressure`, `corner_forcing`, `bomb_control`  
**Build Signal**: Generalist pressure; bomb control + corner forcing

---

### Suction Bomb (キューバンボム)
**Spam Viability**: LOW-MEDIUM — high ink, strong stall  
**Ink Cost**: 70%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.1-0.3 | Comfort | Basic efficiency |
| IRU | 0.1-0.3 | Support | Recovery |
| SS | Situational | Medium | If dying while setting up suction + special |

**Semantic Labels**: `set_play_control`, `denial_zoning`, `delayed_pressure`  
**Build Signal**: Plant bombs for denial more than chip spam

---

### Burst Bomb (クイックボム)
**Spam Viability**: HIGH — low cost, fast, combo enabler  
**Ink Cost**: 40%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.2-0.6 | **CORE** | Combo/spam frequency |
| SPU | Situational | Medium | Throw velocity/range |
| IRU | 0.1-0.3 | Support | Sustain pressure |

**ISS Breakpoints**:
- Baseline: 2 throws possible natively (40% × 2 = 80%)
- **GP39**: Enables 3 throws (true burst spam)
- **GP41**: Comfortable triple burst

**Semantic Labels**: `aggressive_combo`, `chip_starter`, `finisher_range`  
**Build Signal**: Aggressive chip/combo; correlates with Comeback and trade-friendly tempo  
**Note**: On Carbon Roller, SPU signals "Finisher" archetype—securing kills beyond flick range

---

### Curling Bomb (カーリングボム)
**Spam Viability**: LOW-MEDIUM — expensive, mobility-focused  
**Ink Cost**: 70%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SSU | 0.3-1.0 | **CORE** | Follow the path |
| Ninja Squid | 1.0 | **CORE** | If flank kit |
| ISS | 0.0-0.2 | Comfort | Not spam identity |

**Semantic Labels**: `entry_routing`, `flank_intent`, `mobility_tool`  
**Build Signal**: "I want angles" more than poke  
**Anti-Pattern**: High ISS (curling isn't for spam)

---

### Autobomb (ロボットボム)
**Spam Viability**: MEDIUM — mid cost, "fire-and-forget" chase pressure  
**Ink Cost**: 55%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.2-0.6 | **CORE** | **Most efficient ISS breakpoint in game** |
| IRU | 0.1-0.3 | Support | Sustain loop |
| SPU | Situational | Medium | Autobomb behavior |

**ISS Breakpoints**:
- **GP12**: Enables double-throw—most efficient breakpoint in game
- Look for ISS 0.2-0.6 on "autobomb loop" kits

**Semantic Labels**: `search_and_destroy`, `passive_tracking`, `scout_pressure`  
**Build Signal**: Passive tracking/chase; supports safer midline play

---

### Ink Mine (トラップ)
**Spam Viability**: LOW — setup tool, value in placement  
**Ink Cost**: 60%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SPU | 0.1-0.3 | HIGH | Trigger/mark parameters |
| IRU | 0.1-0.2 | Support | Multiple placements |
| SS/QSJ | Situational | Medium | Anchor kit protection |

**Semantic Labels**: `defensive_info`, `anti_flank`, `trap_control`  
**Build Signal**: Defensive info + anti-flank; anchors and trap-control styles

---

### Toxic Mist (ポイズンミスト)
**Spam Viability**: MEDIUM — can cycle but ink-hungry  
**Ink Cost**: 60%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.2-0.6 | HIGH | Mist uptime window |
| IRU | 0.1-0.3 | Support | Recovery |
| SPU | Situational | Medium | Radius/duration effects |

**Semantic Labels**: `choke_zoning`, `movement_tax`, `duel_enabler`  
**Build Signal**: Zoning/control; enabling teammates to win duels  
**Note**: "Mist Locking" = multiple mists to trap opponents

---

### Point Sensor (ポイントセンサー)
**Spam Viability**: HIGH — cheap, fast, constant info  
**Ink Cost**: 45%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.3-0.6 | HIGH | Constant sensor pressure |
| SPU | 0.1-0.3 | HIGH | Mark duration/range |
| IRU | 0.1-0.2 | Support | Sustain |

**Semantic Labels**: `information_broker`, `scout_support`, `safe_pressure`  
**Build Signal**: "We win by information + safe pressure"

---

### Splash Wall (スプラッシュシールド)
**Spam Viability**: MEDIUM — very strong, spacing/ink gated  
**Ink Cost**: 60% (reduces to 39% at GP57)

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.6-1.6 | **CORE** | Wall frequency |
| SPU | 0.1-0.6 | **HIGH** | Wall durability scales significantly |
| IRU | 0.1-0.3 | Support | Wall → Shoot loop is ink-intensive |

**ISS Breakpoints**:
- Practical "wall spam" starts around ISS 1.0+
- JP table: cost reduces from 60% to 39% at GP57

**Semantic Labels**: `wall_stall`, `turret_playstyle`, `safe_space_take`  
**Build Signal**: "I take space safely" — aggressive midline control

---

### Sprinkler (スプリンクラー)
**Spam Viability**: LOW — paint/attention tool, not spam  
**Ink Cost**: Low

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| IRU | 0.1-0.2 | Medium | If placing frequently |
| SCU | 0.3-1.0 | HIGH | Paint → Special farming |
| SPU | Situational | LOW | Sprinkler behavior |

**Semantic Labels**: `paint_support`, `special_farming`, `soft_distraction`  
**Build Signal**: Paint support, farming, soft distraction

---

### Squid Beakon (ジャンプビーコン)
**Spam Viability**: LOW — placement-limited, map control focus  
**Ink Cost**: 75%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SPU | 0.1-0.6 | **MANDATORY** | Grants QSJ speed to teammates |
| QSJ | 0.1 | HIGH | Jump plays |
| Stealth Jump | 1.0 | HIGH | Carrier role protection |

**Semantic Labels**: `logistics_support`, `team_infrastructure`, `rotation_enabler`  
**Build Signal**: Support/anchor infrastructure; "I'm enabling rotations and jumps"

---

### Fizzy Bomb (タンサンボム)
**Spam Viability**: HIGH — one of the biggest ISS-feature discoverers  
**Ink Cost**: 60% (recovers ink during charge)

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.6-1.3 | **CORE** | Fizzy loop identity |
| IRU | 0.3-0.6 | **CORE** | Sustain pressure |
| RSU/SSU | 0.2-0.6 | Medium | Reposition while charging |

**ISS Breakpoints**:
- GP8-9: Double throw threshold
- GP23: Comfortable two bombs per tank
- Look for ISS 0.6-1.3 on fizzy-centric kits

**Semantic Labels**: `fizzy_zoning`, `constant_chip`, `mobility_combo`  
**Build Signal**: Zoning + chip; "I create constant safe damage/paint"  
**Counter**: Sub Resistance Up breaks the 35+35+35 (105 damage) combo

---

### Torpedo (トーピード)
**Spam Viability**: MEDIUM-HIGH — strong info + poke, ink gated  
**Ink Cost**: 65% (recovers if shot down)

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.3-0.8 | HIGH | Torpedo frequency |
| SPU | 0.1-0.3 | HIGH | Throw velocity, forces faster reaction |
| IRU | 0.1-0.3 | Support | Sustain |

**Semantic Labels**: `distraction_initiator`, `scout_poke_hybrid`, `autonomous_pressure`  
**Build Signal**: Scout/poke hybrid; "I want safe pressure and information"  
**Note**: On Splatana Wiper, forces movement before dash-in

---

### Line Marker (ラインマーカー) / Angle Shooter
**Spam Viability**: MEDIUM — lower commitment utility  
**Ink Cost**: 40%

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.2-0.6 | Medium | Frequent usage |
| SPU | 0.1-0.6 | HIGH | Velocity significantly increases |

**Semantic Labels**: `precision_poke`, `tag_pressure`, `snipe_marker`  
**Build Signal**: Precision poke / tag-based pressure  
**Note**: High SPU on Line Marker = pseudo-charger shot; signals "Snipe_Marker" build

---

## 2.3 ISS Investment Efficiency Curve

```
MOST EFFICIENT                                    LEAST EFFICIENT
├─────────────────────────────────────────────────────────────────┤
GP12 (Autobomb 2×) ★ BEST BREAKPOINT
     GP15 (Ink Mine 2×)
          GP21 (Fizzy/Wall/Mist 2×)
               GP29 (Curling 2×)
                    GP35 (Splat/Suction 2×)
                         GP36 (Beakon 2×)
                              GP39 (Burst 3×)
                                   GP41 (Burst comfortable 3×)
```

**Key Insight**: Autobomb at GP12 is the most efficient ISS investment in the game. High ISS on 70% ink cost bombs (Splat/Suction) without LDE signals `inefficient_build` or `niche_spam`.

---

## 2.4 Sub Weapon Semantic Label Quick Reference

| Sub | Primary Label | Build Signal | LDE Synergy |
|-----|---------------|--------------|-------------|
| Splat Bomb | `bomb_control` | Generalist pressure | **HIGH** |
| Suction Bomb | `set_play_control` | Denial zoning | HIGH |
| Burst Bomb | `aggressive_combo` | Chip/combo | LOW |
| Curling Bomb | `flanker_mobility` | Entry routing | LOW |
| Autobomb | `passive_tracking` | Scout pressure | LOW |
| Ink Mine | `defensive_info` | Anti-flank | LOW |
| Toxic Mist | `choke_zoning` | Movement tax | MEDIUM |
| Point Sensor | `information_broker` | Scout support | LOW |
| Splash Wall | `wall_stall` | Safe space take | LOW |
| Sprinkler | `paint_support` | Special farming | LOW |
| Squid Beakon | `logistics_support` | Team infrastructure | LOW |
| Fizzy Bomb | `fizzy_zoning` | Constant chip | HIGH |
| Torpedo | `distraction_initiator` | Scout/poke | MEDIUM |
| Line Marker | `precision_poke` | Tag pressure | LOW |

---

# SECTION 3: Kit-Level Build Directions

Build direction emerges from sub+special synergy, not main weapon alone. The same main weapon with different kits produces entirely different optimal builds.

## 3.1 Shooters

### Splattershot (Suction Bomb + Trizooka)
**Build Direction**: Midline skirmisher winning fights via bomb checks → Zooka picks  
**Archetype**: `frontline_slayer`, `flex_entry`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Comeback | 1.0 | HIGH | Aggression recovery |
| Stealth Jump | 1.0 | HIGH | Entry tool |
| Action Intensify | 0.2-1.0 | HIGH | Jump accuracy (squid rolls) |
| SCU | 0.3-1.0 | Medium | Zooka timings |
| Special Saver | 0.1-0.3 | Medium | Zooka protection |

**Distinguishing Factor**: Suction stalls space *into* Zooka timing  
**Anti-Patterns**: Over-SPU Zooka; heavy ISS (suction isn't spam identity)

---

### Splash-o-matic (Burst Bomb + Crab Tank)
**Build Direction**: Aggressive duelist using burst chip to win midrange, Crab for structured holds  
**Archetype**: `perfect_accuracy_slayer`, `crab_specialist`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **Ninja Squid** | 1.0 | **MANDATORY** | Ambush essential |
| Special Saver | 0.2-0.6 | HIGH | Crab is win condition |
| SPU | 0.2-0.3 | HIGH | Crab duration |
| SSU | 0.6-1.0 | HIGH | Counteract NS penalty |
| Object Shredder | 0.1 | Medium | Crab counter utility |

**Distinguishing Factor**: Zero jump spread (unique among shooters); Burst combo = 35 splash + 3 shots = kill  
**Anti-Patterns**: Full burst-spam ISS (unless explicitly bomb-loop style)

---

### .52 Gal (Splash Wall + Killer Wail 5.1)
**Build Direction**: Aggressive zone control through wall-tempo + wail pressure  
**Archetype**: `wall_abuser`, `rng_mitigation`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISS | 0.6-1.6 | **CORE** | Wall uptime |
| SCU | 0.6-1.3 | HIGH | Wail tempo |
| **Intensify Action** | 0.6-1.0 | **MANDATORY** | 25% max spread vs 2% initial |
| Stealth Jump | 1.0 | HIGH | Entry |
| Special Saver | 0.1 | Medium | Dying after wail |

**Distinguishing Factor**: Wall lets .52 take "illegal" angles safely  
**Anti-Patterns**: Pure QR stacking without map-control follow-through; **DON'T skip IA**; RSU is useless (SSU better for repositioning)  
**JP Note**: "Starting Dash" technique is "super expert only"

---

### .52 Gal Deco (Curling Bomb + Triple Splashdown)
**Build Direction**: Flank/entry kit using curling route → burst duel → splashdown bailout  
**Archetype**: `stealth_slayer`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **Ninja Squid** | 1.0+ | **CORE** | Flank requirement |
| SSU | 0.3-1.0 | HIGH | Speed on paths |
| Special Saver | 0.3-0.6 | HIGH | Splashdown is bailout |
| QSJ | 0.1-0.3 | Medium | Exit plays |

**Distinguishing Factor**: Curling is for routing, not poke; Splashdown = "I refuse to lose this trade"  
**Anti-Patterns**: High ISS (curling isn't spam lever)

---

### .96 Gal (Sprinkler + Ink Vac)
**Build Direction**: Area control with passive paint, requiring extreme ink management  
**Archetype**: `paint_control`, `2hko_anchor`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **ISM** | 1.0+ | **MANDATORY** | Only 40 shots per tank |
| SSU | 0.6-1.0 | HIGH | Repositioning |
| IRU | 0.2-0.3 | HIGH | Recovery |
| Special Saver | 0.1-0.3 | Medium | Vac is specific answer |

**Distinguishing Factor**: Cannot function without ink management  
**Anti-Patterns**: Spraying continuously—accuracy drops from 96% to 70% after 9 shots

---

### .96 Gal Deco (Splash Wall + Kraken Royale)
**Build Direction**: Invincible entry/pivot using Kraken as panic button  
**Archetype**: `kraken_cheese`, `wall_turret`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **Special Saver** | 0.2-0.6 | **CORE** | Kraken is defensive pivot |
| ISM/ISS | 0.3-0.6 | HIGH | Wall consumption |
| SSU | 0.3-0.6 | HIGH | Repositioning |

**Distinguishing Factor**: Kraken dictates entire build; weapon is vessel for special  
**Anti-Patterns**: Dying without Kraken loses defensive pivot

---

### N-ZAP '85 (Suction Bomb + Tacticooler)
**Build Direction**: Team tempo through paint → cooler uptime → fight with buff advantage  
**Archetype**: `cooler_bot`, `paint_support`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 0.6-1.3 | **CORE** | Cooler frequency |
| **SPU** | 0.3-1.0 | **CORE** | Cooler duration |
| QSJ | 0.1 | HIGH | Standard mobility |
| Special Saver | 0.0-0.1 | LOW | Cooler value on placement |
| LDE | 0.1 | Situational | Late-game bomb spam |

**Distinguishing Factor**: Cooler makes entire comp faster/safer; job is "keep it online"  
**Anti-Patterns**: Heavy SS over SCU/SPU; any deviation (e.g., Ninja Squid) signals off-meta "Slayer-Hybrid"

---

### Splattershot Nova (Point Sensor + Killer Wail 5.1)
**Build Direction**: Pure support through information + wail pressure  
**Archetype**: `information_broker`, `special_spammer`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SPU | 0.3-0.6 | HIGH | Sensor range/duration |
| ISS | 0.3-0.6 | HIGH | Constant tagging |
| SCU | 0.6-1.0 | HIGH | Wail spam |

**Distinguishing Factor**: Low damage forces reliance on team coordination via Sensors and Wails

---

### Squeezer (Splash Wall + Trizooka)
**Build Direction**: Midline slayer/turret with extreme ink hunger  
**Archetype**: `mechanical_slayer`, `ink_hungry`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **ISM** | 0.6-1.0 | **MANDATORY** | Very ink hungry |
| RSU | 0.3-0.6 | HIGH | Strafe while firing |
| SCU | 0.3-0.6 | HIGH | Zooka timings |
| IA | 0.2-0.6 | HIGH | Jump shots |

**Distinguishing Factor**: Wall allows forward positions but depletes ink rapidly

---

### Aerospray RG (Sprinkler + Booyah Bomb)
**Build Direction**: Turf support through special output  
**Archetype**: `turf_bot`, `booyah_spammer`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SCU | 1.0+ | **CORE** | Pure special output |
| SPU | 0.3-0.6 | HIGH | Booyah durability |

**Note**: Often viewed as "low skill" but build intent is pure special output

---

## 3.2 Rollers

### Splat Roller (Curling Bomb + Big Bubbler)
**Build Direction**: Objective entry using curling route → bubble claim → close-range threat  
**Archetype**: `sharking_slayer`, `objective_entry`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **Ninja Squid** | 1.0+ | **ESSENTIAL** | "Hiding and ambush is basic playstyle" |
| SSU | 0.6-1.0 | HIGH | Counteract NS + follow curling |
| SCU | 0.3-0.8 | HIGH | Bubble timings |
| Stealth Jump | 1.0 | HIGH | Entry safety |
| ISS | 0.2-0.3 | Medium | Curling comfort |

**Distinguishing Factor**: Bubble turns "one pick" into "we own this area"  
**Anti-Patterns**: SPU overstack if need bubble frequency (SPU worsens rotation)

---

### Carbon Roller (Burst Bomb + Zipcaster)
**Build Direction**: Hyper-aggressive pick kit using burst combo → zip chaos  
**Archetype**: `burst_combo_assassin`, `zip_skirmisher`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Ninja Squid | 1.0 | HIGH | Ambush |
| SSU | 0.3-0.6 | HIGH | Speed |
| **Special Saver** | 0.3-0.6 | **CORE** | Zip = high mortality |
| ISS | 0.3-0.8 | HIGH | Burst combos |
| QSJ | 0.1-0.3 | HIGH | Zip exit |
| **Drop Roller** | 1.0 | HIGH | Zipcaster landing safety |

**Distinguishing Factor**: Burst makes Carbon's lethal windows larger; Zip turns picks into collapse pressure; Zipcaster enables normally impossible flank angles  
**Anti-Patterns**: Pure SCU without SS

---

### Carbon Roller Deco (Autobomb + Trizooka)
**Build Direction**: Combo slayer using autobomb scouting into kills  
**Archetype**: `search_and_destroy_assassin`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SPU | 0.3-0.6 | HIGH | Autobomb for locating |
| ISS | 0.2-0.6 | HIGH | Autobomb loop |
| **Ninja Squid** | 1.0 | **CORE** | Sharking requirement |
| Special Saver | 0.1-0.3 | Medium | Zooka protection |

---

### Dynamo Roller (Sprinkler + Tacticooler)
**Build Direction**: Heavy support through area denial + team buffs  
**Archetype**: `heavy_support`, `ink_manager`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **ISM** | 1.0+ | **MANDATORY** | Slowest ink recovery, highest consumption |
| SPU | 0.3-0.6 | HIGH | Cooler duration |
| Thermal Ink | 0.1 | Situational | Long-range flick synergy |

**Distinguishing Factor**: Cannot function without efficiency perks

---

## 3.3 Chargers

### E-liter 4K / Scope (Ink Mine + Wave Breaker)
**Build Direction**: Ultimate backline anchor with defensive self-protection  
**Archetype**: `hard_anchor`, `punisher`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **Respawn Punisher** | 1.0 | **CORE META** | Rarely dies; maximize kill penalty |
| ISM | 0.6-1.0 | HIGH | More shots per tank |
| QSJ | 0.1 | HIGH | Escape option |
| SCU | 0.3-0.6 | Medium | Sonar frequency |
| SPU | 0.1-0.6 | Medium | Sonar radius |
| Sub Res | 0.1 | Standard | Omamori |

**Distinguishing Factor**: "Position is everything—your role is to kill enemy long-range and support teammates"; Sonar turns "I see you" into "you can't move cleanly"  
**Anti-Patterns**: Heavy QR (you're stabilizing, not trading)  
**Counter Note**: Object Shredder lets E-liter one-shot Crab Tank armor

---

### Splat Charger / Splatterscope (Splat Bomb + Triple Inkstrike)
**Build Direction**: Standard anchor with mobility options  
**Archetype**: `precision_anchor`, `mobile_anchor`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Respawn Punisher | 1.0 | HIGH | Standard anchor tech |
| ISM | 0.6-1.0 | HIGH | Shot efficiency |
| IRU | 0.2-0.3 | Medium | Recovery |
| SSU | 0.3-0.6 | Medium | Repositioning |
| SCU | 0.3-0.6 | Medium | Inkstrike frequency |

**Distinguishing Factor**: Fastest charge-to-range ratio (0.056 vs Liter's 0.046 test lines/frame); Bomb for close-range self-defense

---

### Bamboozler 14 Mk I (Autobomb + Killer Wail 5.1)
**Build Direction**: Mobile skirmisher playing closer to shooter than charger  
**Archetype**: `mobile_sniper`, `tap_shot_king`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| RSU | 0.6-1.0 | HIGH | Mobility requirement |
| ISM | 0.3-0.6 | HIGH | Tap efficiency |
| SSU | 0.3-0.6 | HIGH | Repositioning |
| SCU | 0.3-0.6 | Medium | Wail tempo |

---

### Snipewriter 5H (Sprinkler + Tacticooler)
**Build Direction**: Mobile support anchor prioritizing team buffs  
**Archetype**: `cooler_anchor`, `support_sniper`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SPU | 0.6-1.0 | **CORE** | Cooler duration |
| RSU | 0.3-0.6 | HIGH | Strafe mobility |
| Thermal Ink | 0.1 | Situational | Tracking |

**Note**: Unlike E-liter, prioritizes SPU for Cooler over Respawn Punisher

---

## 3.4 Sloshers

### Slosher / Slosher Deco (Splat Bomb + Trizooka / Angle Shooter + Zipcaster)
**Build Direction**: Mid-range curved-shot specialist excelling over walls  
**Archetype**: `ledge_slayer`, `midline_slayer`

**Vanilla (Trizooka)**:
| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SSU | 0.6-1.0 | HIGH | Repositioning |
| ISS | 0.2-0.3 | HIGH | Bomb comfort |
| SRU | 0.1 | Standard | Omamori |
| Comeback | 1.0 | HIGH | Aggression recovery |
| Special Saver | 0.1-0.3 | Medium | Zooka protection |

**Deco (Zipcaster)**:
| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **QR** | 0.6-1.0 | **CORE** | Zip = high mortality |
| Stealth Jump | 1.0 | HIGH | Entry |
| SSU | 0.6-1.0 | HIGH | Speed |

**Distinguishing Factor**: 70 damage main + 30 bomb splash = instant kill  
**Anti-Patterns**: Ignoring ink management—extremely ink-hungry

---

### Tri-Slosher / Nouveau (Toxic Mist + Inkjet / Fizzy + Tacticooler)
**Build Direction**: Close-range ambush shredder  
**Archetype**: `close_range_shredder`, `ambush_slayer`

**Vanilla (Inkjet)**:
| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SSU | 1.0+ | HIGH | Get in close |
| SPU | 0.1+ | HIGH | Inkjet explosion radius |
| Special Saver | 0.1-0.2 | HIGH | Jet protection |
| Drop Roller | 1.0 | HIGH | Landing safety |
| Ninja Squid | 1.0 | HIGH | Ambush requirement |
| Comeback | 1.0 | HIGH | Aggression recovery |

**Nouveau (Tacticooler)**:
| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| SPU | 0.3-0.6 | HIGH | Cooler duration |
| SSU/ISS | 0.3-0.6 | HIGH | Fizzy loop |

---

### Sloshing Machine / Neo (Fizzy Bomb + Booyah Bomb / Splat Bomb + Trizooka)
**Build Direction**: Versatile skirmisher abusing massive hitbox  
**Archetype**: `hitbox_abuser`, `physics_poker`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Stealth Jump | 1.0 | HIGH | Entry |
| SSU | 0.6-1.0 | HIGH | Repositioning |
| ISS | 0.3-0.6 | HIGH | Fizzy spam |
| **LDE** | 0.1 | **HIGH** | Late-game Fizzy spam |

**Distinguishing Factor**: Massive hitbox + Fizzy mobility allows awkward angle fights

---

## 3.5 Splatlings

### Heavy Splatling (Sprinkler + Wave Breaker)
**Build Direction**: Defensive anchor requiring extreme RSU investment  
**Archetype**: `run_gun_anchor`, `turret_control`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **RSU** | 2.0+ (27.6 AP avg) | **MANDATORY** | Strafe while firing |
| IA | 0.3-0.6 | HIGH | Jump stability |
| IR | 0.1-0.2 | HIGH | Ink resistance |
| SPU | 0.1-0.6 | Medium | Sonar radius |
| SCU | 0.3-0.8 | Medium | Sonar frequency |
| Special Saver | 0.1 | Standard | Protection |

**Distinguishing Factor**: Weapon requires repositioning while firing; sonar punishes jump approaches  
**Anti-Patterns**: **DON'T run without Run Speed**—too little RSU wastes the kit; ignoring strafe thresholds

---

### Hydra Splatling (Autobomb + Booyah Bomb)
**Build Direction**: Ultimate turret anchor with extreme immobility  
**Archetype**: `immobile_turret`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **RSU** | 2.0+ (GP30+ essential) | **MANDATORY** | Basic functioning |
| IR | 0.1-0.2 | HIGH | Ink resistance |
| Respawn Punisher | 1.0 | HIGH | Extreme range/safety |

**Distinguishing Factor**: Most immobile weapon; RSU is necessity, not luxury

---

### Mini Splatling (Burst Bomb + Ultra Stamp)
**Build Direction**: Aggressive mobile splatling  
**Archetype**: `mobile_aggro_splatling`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| RSU | GP10-20 | HIGH | Sufficient for mobility |
| ISS | 0.2-0.6 | HIGH | Burst combos |
| SSU | 0.3-0.6 | HIGH | Repositioning |

---

### Nautilus 47 / 79 (Point Sensor + Ink Storm / Suction Bomb + Triple Inkstrike)
**Build Direction**: Aggressive midline that swims to hold charge  
**Archetype**: `swim_charger`, `midline_aggro`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **SSU** | 0.6-1.0 | **CORE** | Swim to hold charge |
| IA | 0.3-0.6 | HIGH | Jump shots |
| ISM | 0.3-0.6 | HIGH | Efficiency |

**Distinguishing Factor**: Only splatling that holds charge while swimming; SSU more valuable than RSU

---

### Ballpoint Splatling (Fizzy Bomb + Inkjet)
**Build Direction**: Hybrid anchor/slayer with mechanical depth  
**Archetype**: `mechanics_hybrid`, `jet_slayer`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| RSU | 0.6-1.0 | HIGH | Strafe short-range mode |
| SPU | 0.3-0.6 | HIGH | Inkjet radius/duration |
| Special Saver | 0.2-0.6 | HIGH | Jet protection |
| IR | 0.1-0.2 | HIGH | Standard |

**Distinguishing Factor**: Inkjet turns it into a slayer

---

## 3.6 Dualies

### Splat Dualies / Enperry (Suction Bomb + Crab Tank / Curling + Triple Inkstrike)
**Build Direction**: Skirmisher slayer using dodge rolls to win 1v1s  
**Archetype**: `mobile_slayer`, `crab_entry`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **ISM** | 0.3-0.6 | HIGH | Roll efficiency |
| QR | 0.3-0.6 | HIGH | Trading pattern |
| Stealth Jump | 1.0 | HIGH | Entry |
| Comeback | 1.0 | HIGH | Aggression recovery |
| Special Saver | 0.2-0.6 | Medium | Crab protection |

**Anti-Patterns**: Builds without ISM struggle to maintain roll aggression

---

### Tetra Dualies (Autobomb + Reefslider)
**Build Direction**: Pure zombie slayer designed to slide in, distract, die, repeat  
**Archetype**: `zombie_aggro`, `distraction_tank`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **QR** | 1.0+ (20+ AP) | **MANDATORY** | Core identity; weapon's K/D naturally volatile |
| Stealth Jump | 1.0 | **CORE** | Jump-in playstyle |
| QSJ | 0.1-0.3 | HIGH | Fast return |

**Distinguishing Factor**: Any build without QR is off-meta/suboptimal  
**Note**: Sendou.ink builds consistently show >20 AP QR

---

### Dualie Squelchers / Custom (Splat Bomb + Wave Breaker / Point Sensor + Kraken)
**Build Direction**: Mobile midline with unique jump-after-roll tech  
**Archetype**: `evasive_midline`, `bomb_poker`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **ISM** | 0.6-1.0 | HIGH | Poor efficiency |
| **IA** | 0.3-0.6 | **CRITICAL** | Only dualie that can jump after roll |
| LDE | 0.1 | HIGH | Bomb spam |

**Distinguishing Factor**: IA is critical to maintain accuracy during jump-after-roll tech

---

### Dapple Dualies / Nouveau (Squid Beakon + Tacticooler / Torpedo + Reefslider)
**Build Direction**: Zombie frontline support with beacon network  
**Archetype**: `zombie_support`, `frontline_maintainer` / `torpedo_slayer`, `shark_aggro`

**Vanilla (Tacticooler)**:
| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **QR** | 1.0+ | **CORE** | Zombie build |
| Comeback | 1.0 | HIGH | Death recovery |
| Stealth Jump | 1.0 | HIGH | Entry |
| SPU | 0.3-0.6 | HIGH | Beakon QSJ for teammates |

**Nouveau (Reefslider)**:
| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| QR | 0.6-1.0 | HIGH | Slider deaths |
| SPU | 0.2-0.3 | HIGH | Torpedo speed |
| **Ninja Squid** | 1.0 | HIGH | Extremely short range |

**Distinguishing Factor**: "Kit lacks breakout power—rely on teammate specials for openings"  
**Anti-Patterns**: Playing lone wolf—weapon needs Beacon network

---

## 3.7 Blasters

### Luna Blaster / Neo (Splat Bomb + Zipcaster / Fizzy + Ultra Stamp)
**Build Direction**: Close-range ambush using Zipcaster for gap-closing  
**Archetype**: `ambush_blaster`, `close_quarter_assassin`

**Vanilla (Zipcaster)**:
| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **Ninja Squid** | Highly rec | HIGH | Ambush requirement |
| SSU | 0.6-1.0 | HIGH | Approach speed |
| Stealth Jump | 1.0 | HIGH | Entry |
| QR | 0.3-0.6 | HIGH | Close-range = deaths |
| Special Saver | 0.3-0.6 | HIGH | Zip protection |

**Anti-Patterns**: Don't spam Splat Bomb—reveals position for ambush weapon

---

### Range Blaster (Suction Bomb + Wave Breaker)
**Build Direction**: Space control slayer walling out enemies with indirect hits  
**Archetype**: `space_control`, `jump_shooter`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **ISM** | 0.6-1.0 | **MANDATORY** | Very low shots/tank |
| **IA** | 0.6-2.0 | **MANDATORY** | Jump accuracy required; terrible RNG without |
| QR | 0.3-0.6 | HIGH | Trading pattern |
| Stealth Jump | 1.0 | HIGH | Entry |

**Distinguishing Factor**: Relies on indirect hits to wall out enemies; runs out of ink instantly without ISM

---

### Rapid Blaster / Pro (Ink Mine + Triple Inkstrike / Toxic Mist + Ink Vac)
**Build Direction**: Midline area denial  
**Archetype**: `area_denial_blaster`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| ISM | 0.3-0.6 | HIGH | Efficiency |
| IA | 0.3-0.6 | HIGH | Jump shots |
| SCU | 0.3-0.6 | Medium | Inkstrike frequency |

---

## 3.8 Stringers & Splatanas

### Tri-Stringer / Inkline (Toxic Mist + Killer Wail / Sprinkler + Super Chump)
**Build Direction**: Zoning anchor using explosive arrows for area denial  
**Archetype**: `explosive_zoner`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| RSU | 0.6-1.0 | HIGH | Mobility while aiming |
| SCU | 0.3-0.6 | HIGH | Wail frequency |
| **Object Shredder** | 0.1 | **HIGH (RM)** | Premier shield popper |

**Distinguishing Factor**: Wail forces movement; Object Shredder makes it premier Rainmaker pick

---

### REEF-LUX 450 (Curling Bomb + Tenta Missiles)
**Build Direction**: Paint support with Missile spam for global pressure  
**Archetype**: `missile_farmer`, `paint_support`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **SCU** | 1.0+ | **CORE IDENTITY** | Pure special spam |
| SPU | 0.3-0.6 | HIGH | More targets, paint parameters |
| RSU | 0.3-0.6 | Medium | Mobility |

**Distinguishing Factor**: Fastest painter in game; used almost exclusively for Missile spam  
**Anti-Patterns**: Building for kills—strength is special spam

---

### REEF-LUX 450 Deco (Splash Wall + Reefslider)
**Build Direction**: Frontline shield positioning with Reefslider counter-plays  
**Archetype**: `shield_slayer`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| QR | 0.6-1.0 | HIGH | Slider deaths |
| Stealth Jump | 1.0 | HIGH | Entry |
| **SPU** | 0.3-0.6 | HIGH | Reefslider destroys Barriers, Crab Tanks |
| ISS | 0.3-0.6 | HIGH | Wall frequency |

---

### Splatana Stamper (Burst Bomb + Zipcaster)
**Build Direction**: Combo skirmisher using burst → zip chaos  
**Archetype**: `zip_skirmisher`, `combo_heavy`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **QR** | 0.6-1.0 | **CORE** | Zipcaster = high mortality |
| Stealth Jump | 1.0 | HIGH | Entry |
| ISS | 0.3-0.6 | HIGH | Burst combos |
| Special Saver | 0.2-0.6 | HIGH | Zip risk mitigation |

**Distinguishing Factor**: Burst combo essential for consistent kills; Zipcaster enables backline challenges  
**Note**: Builds lacking QR are suboptimal in competitive

---

### Splatana Wiper / Deco (Torpedo + Ultra Stamp / Toxic Mist + Triple Inkstrike)
**Build Direction**: Fast slayer entry fragger  
**Archetype**: `speed_slayer`, `entry_fragger`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **QR** | 0.6-1.0 | **CORE** | Zombie style mandatory |
| SSU | 0.6-1.0 | HIGH | Speed |
| Stealth Jump | 1.0 | HIGH | Entry |
| Comeback | 1.0 | HIGH | Death recovery |
| SPU | 0.1-0.3 | Medium | Torpedo speed for movement forcing |

**Distinguishing Factor**: Plays very similarly to Tetra Dualies (zombie style); Torpedo forces movement before dash-in

---

## 3.9 Brushes

### Painbrush (Curling Bomb + Wave Breaker)
**Build Direction**: Aggressive flanker with extreme ink hunger  
**Archetype**: `ink_hungry_flanker`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| **ISM** | 0.6-1.0 | **MANDATORY** | Extremely hungry per flick |
| Ninja Squid | 1.0 | HIGH | Flank requirement |
| SSU | 0.6-1.0 | HIGH | Speed |

---

### Octobrush / Nouveau (Suction Bomb + Zipcaster / Fizzy + Inkjet)
**Build Direction**: Aggressive ambush brush  
**Archetype**: `ambush_brush`

| Ability | Investment | Priority | Reasoning |
|---------|------------|----------|-----------|
| Ninja Squid | 1.0 | HIGH | Ambush |
| SSU | 0.6-1.0 | HIGH | Approach |
| QR | 0.3-0.6 | HIGH | Close-range = deaths |
| Special Saver | 0.3-0.6 | HIGH | Zip/Jet protection |

---

# SECTION 4: Mode-Specific Ability Value Shifts

Ranked mode mechanics create dramatic ability value shifts. The same ability can go from LOW to ESSENTIAL depending on mode context.

## 4.1 Master Value Table

| Ability | Splat Zones | Tower Control | Rainmaker | Clam Blitz |
|---------|-------------|---------------|-----------|------------|
| **Object Shredder** | LOW | HIGH | **ESSENTIAL** | Standard |
| **Quick Super Jump** | Standard | Standard | Standard | **HIGH** |
| **Special Saver** | **HIGH** | HIGH | Standard | LOW-MED |
| **Quick Respawn** | **HIGH** | **HIGH** | Standard-HIGH | HIGH |
| **Respawn Punisher** | **HIGH** | LOW | HIGH | LOW |
| **Stealth Jump** | Standard-HIGH | **ESSENTIAL** | Standard-HIGH | **ESSENTIAL** |
| **Tenacity** | HIGH | Standard | LOW-Standard | LOW |
| **Last-Ditch Effort** | **ESSENTIAL** | HIGH | HIGH | Medium |
| **Sub Resistance Up** | Standard | Standard | Standard | HIGH |
| **Ink Resistance Up** | Standard | Standard | Standard | HIGH |
| **Ninja Squid** | Standard | Standard | Standard | **HIGH** |
| **Opening Gambit** | LOW | LOW | Medium | LOW |

---

## 4.2 Key Mode-Ability Interactions Explained

### Object Shredder
**Rainmaker → ESSENTIAL**: 10% faster shield break wins barrier races. Being first to break = map control + potential splats from explosion. Single ability determines many barrier contests.

**Tower Control → HIGH**: Big Bubblers placed on tower become nearly indestructible without Object Shredder. Destroys Bubblers ~30% faster. Also deals 1.1× to Crab Tank armor.

**Splat Zones → LOW**: No breakable objects beyond occasional Bubbler.

---

### Quick Respawn
**Splat Zones → HIGH/ESSENTIAL**: JP wiki explicitly states "compatibility is optimal" (最適). Zones require constant presence; early opponent lead activates ability for entire match.

**Tower Control → HIGH**: Tower fights repeatedly force close-quarters contests; trading into tower progress is often correct strategy.

**Clam Blitz → HIGH**: Fast-paced mode with frequent deaths during clam delivery attempts.

---

### Respawn Punisher
**Splat Zones → HIGH**: Counter-pick in QR-heavy metas. Reduces Quick Respawn effectiveness by **85%**. Best on Chargers/Splatlings that consistently splat without dying.

**Rainmaker → HIGH**: Killing carrier with RP forces massive delay, enabling knockout potential.

**Tower Control → LOW**: Frequent trading makes dying more acceptable.

**Clam Blitz → LOW**: Deaths while holding clams already punishing.

---

### Stealth Jump
**Tower Control → ESSENTIAL**: Jumping to tower-riding teammates puts you directly in enemy sightlines. Without Stealth Jump, enemies camp landing points; with it, surprise arrivals enable tower support.

**Clam Blitz → ESSENTIAL**: Jump plays with clams are core strategy. Last-second jump scores require unpredictable landings.

---

### Tenacity
**Splat Zones → HIGH**: Count mechanic creates long "we're behind" stretches even in even-skill games, so passive special value is more frequent.

**Other Modes → LOW-Standard**: Dependent on teammates dying faster than you while you survive—unreliable in fast-paced modes.

**Note**: Generally C-tier ability due to inconsistent activation.

---

### Last-Ditch Effort
**Splat Zones → ESSENTIAL**: Triggers based on enemy count. Zones provides the most consistent "losing" state (counting down from 100) to trigger early and reliably.

**All Modes → HIGH for Lethal Bombs**: Enables "double bomb" plays otherwise impossible. High ISS without LDE on 70% bombs = `inefficient_build` signal.

---

## 4.3 Mode-Context Semantic Rules

**If Object Shredder + Rainmaker context** → Label as `shield_popper`  
**If Object Shredder + Tower Control** → Label as `bubble_popper`, `wall_breaker`  
**If Object Shredder + Charger/Blaster** → Label as `beacon_clearer`, `armor_counter`

**If Stealth Jump + Frontline weapon** → Label as `aggressive_entry`  
**If Stealth Jump + Backline weapon** → Label as `unusual_aggression` or `pivot_anchor` (signal player intends forward play)

**If QR + Splat Zones** → Label as `zones_zombie`  
**If RP + Splat Zones** → Label as `punisher_counter` (counter-meta to QR builds)

---

# SECTION 5: Ability Investment Curve Refinements

Non-linear value curves create "dead zones" of wasted AP and "sweet spots" of exceptional efficiency. The Japanese concept of **Chousei (調整)** dictates investing just enough AP to hit specific frame data thresholds.

## 5.1 The "Omamori" Philosophy

**Omamori (お守り)** = "Charm gear" — 0.1 (GP3) investments in protective abilities that provide outsized value for minimal cost.

**Standard Omamori Suite**:
- QSJ 0.1 (GP3)
- Sub Res 0.1 (GP3)
- Ink Res 0.1 (GP3)
- Special Saver 0.1 (GP3)

These abilities are "one-point wonders" where first sub provides ~30-40% of maximum effect.

---

## 5.2 Quick Super Jump (QSJ)

**Curve Shape**: Rapid diminishing returns (Front-loaded)

| Investment | Effect | Frame Reduction | Efficiency |
|------------|--------|-----------------|------------|
| GP0 | Base | — | — |
| **GP3 (0.1)** | ~37% of max | ~22 frames | **★ BEST EFFICIENCY** |
| GP6 (0.2) | ~50% of max | ~30 frames | Good |
| GP10+ | ~65%+ | Diminishing | Inefficient |

**Sweet Spot**: **0.1 (GP3)** — single most efficient investment in the game  
**Dead Zone**: GP6-10 often inefficient unless stacking for Clam Blitz strategies  
**Semantic Label**: `omamori_utility`

---

## 5.3 Quick Respawn (QR)

**Curve Shape**: Threshold + Diminishing returns

| Investment | Reduction | Respawn Time | Efficiency |
|------------|-----------|--------------|------------|
| GP0 | Base | 8.5 sec | — |
| GP18 (0.6) | ~2 sec | ~6.5 sec | **Very efficient** |
| **GP24-32** | ~2.5-3 sec | 5.5-6 sec | **★ SWEET SPOT** |
| GP38+ | 3.5+ sec | ~5 sec | ~2 F/GP (severe diminishing) |

**Sweet Spot**: **GP24-32 (1.6-2.0)** — "Zombie standard"  
**Dead Zone**: GP0-15 (too little effect) and GP38+ (efficiency drops severely)  
**Activation Requirement**: Must die without getting a kill  
**Standard Zombie Build**: GP24-32 + Stealth Jump + QSJ 0.1

**Semantic Labels**:
- GP18-32: `zombie_commitment`
- GP0-15: `fake_zombie` (inefficient)
- GP38+: `excessive_zombie`

---

## 5.4 Special Saver (SS)

**Curve Shape**: Extremely front-loaded ("One-point wonder")

| Investment | Gauge Retained | Loss | Efficiency |
|------------|----------------|------|------------|
| GP0 | 50% | 50% | Base |
| **GP3 (0.1)** | ~59% | 41% | **★ ~3% per GP — Extremely efficient** |
| GP6 (0.2) | ~64% | 36% | 1.67% per GP |
| GP10 (1.0) | ~70% | 30% | 1.5% per GP |

**Sweet Spot**: **0.1 (GP3)** — universally recommended ("one-point wonder")  
**Dead Zone**: GP10-20 offers poor efficiency compared to minimal investment  
**Semantic Labels**: `gauge_insurance`, `omamori_special`

**Key Insight**: Special Saver 0.1 outperforms Special Charge Up at equal investment for death-heavy playstyles.

---

## 5.5 Sub Resistance Up (SRU)

**Curve Shape**: Threshold-based step function with combo-breaking gates

| Investment | Reduction | Combo Broken |
|------------|-----------|--------------|
| **GP3 (0.1)** | ~15% | Fizzy 35+35+35 (105), Burst splash combos |
| GP6 (0.2) | ~22% | More combo denial |
| GP10+ | ~28%+ | Diminishing returns |

**Sweet Spot**: **0.1 (GP3)** — standard competitive inclusion  
**Dead Zone**: GP10-20 (poor efficiency; better abilities available)  
**Semantic Label**: `combo_breaker`

**Critical Breakpoint**: GP3 reduces Fizzy Bomb direct damage from 50 to 45.9, splash from 35 to 32.1—prevents 3-hit kill.

---

## 5.6 Intensify Action (IA)

**Curve Shape**: Logarithmic decay (Bimodal value)

| Investment | Spread Reduction | Use Case |
|------------|-----------------|----------|
| GP3 (0.1) | ~33% | Dramatic baseline improvement |
| **GP6 (0.2)** | ~50% | **Half base spread** |
| GP10 (1.0) | ~65% | Committed accuracy |
| GP20 (2.0) | ~80% | Maximum accuracy |

**Sweet Spots** (Bimodal):
- **0.2-0.3**: "Small correction" for Shooters, Splatlings, ground-focused play
- **1.0+**: "Build identity" for high-spread weapons, jump-heavy playstyles

**When 0.1-0.2 sufficient**: Splatlings, short-range shooters, ground-focused players  
**When 1.0+ needed**: .52 Gal (25% max spread), .96 Gal, Blasters (single-shot accuracy critical)

**Dead Zone**: 0.1-0.2 on kits that don't jump-shoot under pressure  
**Zero Value**: Weapons without jump spread (Sharp Marker, H3)

**Semantic Labels**: `accuracy_fix` (Shooters at low IA), `jump_tech_enabler` (Blasters at high IA)

---

## 5.7 Run Speed Up on Splatlings

**Curve Shape**: Threshold-based (Weapon-class dependent)

| Splatling | Base Speed | Competitive Standard | Threshold |
|-----------|------------|---------------------|-----------|
| Heavy Splatling | 0.60 DU/F | GP23-30 (2.3-3.0) | ~GP23 |
| Hydra Splatling | 0.48 DU/F | **GP30+ essential** | ~GP30 |
| Mini Splatling | 0.84 DU/F | GP10-20 sufficient | ~GP16 |
| Nautilus | 0.72 DU/F | GP15-25 typical | ~GP20 |

**Sweet Spots**: Heavy at GP23, Mini at GP16, Hydra at GP30+  
**Dead Zone**: Small amounts that don't change strafe viability (you still get clipped)

**Key Insight**: RSU becomes quasi-Special Charge for Splatlings—better strafe coverage means more efficient special building through paint overlap reduction.

---

## 5.8 Ink Saver Sub (ISS) — Double-Bomb Thresholds

**Curve Shape**: Hard threshold gates (Binary value jumps)

| Sub | Ink Cost | ISS for 2× | Note |
|-----|----------|------------|------|
| Autobomb | 55% | **GP12** | ★ Most efficient |
| Torpedo | 65% | GP10-11 | Recovers if shot |
| Fizzy Bomb | 60% | GP8-9, GP23 comfortable | Variable charge |
| Splash Wall | 60% | GP21 | Critical for wall spam |
| Toxic Mist | 60% | GP21 | Mist locking |
| Splat/Suction/Curling | 70% | GP29-35 | Heavy investment |
| Burst Bomb | 40% | GP0 native, **GP39** for triple | |
| Squid Beakon | 75% | GP36 | Not spam pattern |

**Dead Zones**:
- GP1-11 for 70% bombs (below threshold = wasted)
- GP14-15 (just over threshold, no practical benefit over GP13)

**Sweet Spots**: GP12-13 (minimum viable double bomb), GP16 (comfortable), GP26-28 (Wakaba standard)

**Semantic Labels**:
- At threshold: `bomb_spam_threshold`
- Below threshold on 70% bomb: `inefficient_build`
- High ISS without LDE on lethal bombs: `niche_spam` or `inefficient_build`

---

## 5.9 Investment Curve Quick Reference

| Ability | Curve | Sweet Spot | Dead Zone | Semantic Label |
|---------|-------|------------|-----------|----------------|
| QSJ | Front-loaded | **0.1 (GP3)** | GP6-10 | `omamori_utility` |
| QR | Threshold + Dim | **GP24-32** | GP0-15, GP38+ | `zombie_commitment` |
| SS | Front-loaded | **0.1 (GP3)** | GP10-20 | `gauge_insurance` |
| SRU | Threshold | **0.1 (GP3)** | GP10+ | `combo_breaker` |
| IA | Bimodal | 0.2 or 1.0+ | 0.1-0.2 if no jump | `accuracy_fix` / `jump_tech` |
| RSU (Splatling) | Threshold | Weapon-dependent | Below threshold | `strafe_threshold` |
| ISS | Hard threshold | At breakpoint | Below breakpoint | `bomb_spam_threshold` |

---

# SECTION 6: Semantic Label Master Reference

## 6.1 Build Philosophy Labels

| Label | Definition | Signals |
|-------|------------|---------|
| `sub_spam` | ISS investment + high-spam-viability sub | ISS 0.6+, Fizzy/Burst/Autobomb |
| `special_farm` | SCU 1.0+ + spam special | Missile/Storm/Inkstrike focus |
| `conserve_special` | Special Saver 0.1+ + high-impact special | Crab/Trizooka/Inkjet |
| `zombie_aggression` | QR 1.0+ + Stealth Jump + Comeback | カムバゾンビステジャン pattern |
| `team_support` | Tacticooler/Beacon + SPU investment | Cooler loop, logistics |
| `ambush_slayer` | Ninja Squid + SSU + short-range | Rollers, close blasters |
| `backline_anchor` | ISM/IRU focus + charger/splatling | Respawn Punisher viable |

## 6.2 Ability Efficiency Labels

| Label | Definition | Signals |
|-------|------------|---------|
| `omamori_build` | 0.1 investment abilities stacked | QSJ/SRU/IR/SS at GP3 each |
| `one_point_wonder` | Single ability at 0.1 providing max efficiency | SS 0.1, QSJ 0.1 |
| `threshold_investment` | ISS at exact double-throw breakpoint | GP12 Autobomb, GP21 Fizzy |
| `dead_zone_waste` | GP in inefficient ranges | QR GP10, ISS GP14 |
| `mobility_stack` | SSU/RSU above 2.0 investment | Splatling strafe, NS compensation |

## 6.3 Mode-Context Labels

| Label | Definition | Signals |
|-------|------------|---------|
| `rainmaker_shredder` | Object Shredder priority | RM mode + OS ability |
| `tower_stealth` | Stealth Jump mandatory | TC mode context |
| `zones_zombie` | Quick Respawn optimal | SZ mode + QR build |
| `punisher_counter` | Respawn Punisher in QR meta | Charger/Splatling + RP |

## 6.4 Kit-Archetype Labels

| Label | Definition | Example Kits |
|-------|------------|--------------|
| `shield_slayer` | Splash Wall + aggressive main | .52 Gal |
| `special_farmer` | Spam special + SCU stack | REEF-LUX, N-ZAP '85 |
| `combo_assassin` | Burst/Quick Bomb + instant-kill main | Carbon Roller, Splash-o-matic |
| `mobile_anchor` | Charger + mobility focus | Splat Charger, Bamboozler |
| `cooler_loop` | Tacticooler + SPU/SCU focus | N-ZAP '85, Dapple Dualies |
| `wall_turret` | Wall + stationary playstyle | .52 Gal, .96 Gal Deco |
| `zip_skirmisher` | Zipcaster + aggressive main | Carbon Roller, Splatana Stamper |

## 6.5 Anti-Pattern Labels

| Label | Definition | Detection |
|-------|------------|-----------|
| `inefficient_build` | AP in dead zones | ISS below threshold, excessive QR |
| `conflicting_abilities` | RP + QR | Both present in build |
| `missing_core_ability` | Critical ability absent | .52 Gal without IA, Heavy without RSU |
| `wrong_philosophy` | Special investment mismatched | SS on spam special, SPU overstack |

---

# APPENDIX A: Japanese Terminology Reference

| English | Japanese | Romanization | Notes |
|---------|----------|--------------|-------|
| Quick Respawn | 復活時間短縮 | Fukkatsu jikan tanshuku | 復短 abbreviated |
| Special Saver | スペシャル減少量ダウン | Supesharu genshou-ryou daun | スペ減 abbreviated |
| Special Charge Up | スペシャル増加量アップ | Supesharu zouka-ryou appu | |
| Ninja Squid | イカニンジャ | Ika Ninja | |
| Stealth Jump | ステルスジャンプ | Suterusu Janpu | ステジャン abbreviated |
| Comeback | カムバック | Kamubakku | |
| Respawn Punisher | 復活ペナルティアップ | Fukkatsu Penarutii Appu | |
| Omamori | お守り | Omamori | "Charm gear" = 0.1 investments |
| Chousei | 調整 | Chousei | "Adjustment" = precise breakpoints |
| Zombie Build | ゾンビ | Zonbi | カムバゾンビステジャン full name |

---

# APPENDIX B: Source Tier Reference

### Tier 1 (Most Valuable)
- **wikiwiki.jp/splatoon3mix** — Japanese wiki with detailed mechanics, breakpoints, per-weapon/ability pages
- **Sendou.ink** — Build statistics, weapon data, competitive community
- **note.com** (Japanese blog platform) — Pro player reasoning and build philosophy

### Tier 2 (Good Supporting Data)
- **Game8.jp / GameWith.jp** — Per-ability guides, recommended weapons
- **Famitsu** — Beginner-friendly but sometimes shallow
- **Altema.jp** — Similar to Game8

### Tier 3 (Western Sources)
- **Squidboards** — Competitive discussion, sometimes outdated
- **Inkipedia** — Factual wiki, less competitive insight
- **Reddit/Discord** — Variable quality, occasional pro insight

### Tier 4 (Verification Only)
- **YouTube guides** — Often outdated or casual-focused
- **English gaming sites** — Usually thin content

---

**Document End**

*This consolidated reference synthesizes three independent research efforts into a unified semantic labeling framework for SAE feature pattern recognition in Splatoon 3 gear builds.*