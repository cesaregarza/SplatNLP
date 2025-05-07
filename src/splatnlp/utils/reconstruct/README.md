# Splatoon Gear Build Reconstruction Module

This module reconstructs optimal and legal Splatoon-style gear builds based on a predictive model and defined game mechanics. The primary function, `reconstruct_build`, employs a beam search algorithm to explore combinations of "capstone" abilities. An "Allocator" class then attempts to materialize these capstone sets into concrete, valid gear configurations.

## Core Requirements and Logic

### 1. Goal
To determine the single best, legal gear build for a given weapon, potentially influenced by an initial set of abilities (context). A "build" consists of three gear pieces (head, clothes, shoes), each with one main ability and up to three sub-ability slots (total 9 sub-slots).

### 2. Input Components
- `predict` function: A model that, given a current sequence of abilities (context) and a weapon ID, predicts scores or probabilities for potential next abilities.
- `weapon_id`: Identifier for the weapon for which the build is generated.
- `initial_context`: A sequence of ability names to start the build process.
- `vocab_tokens`: A comprehensive list of all possible ability tokens recognized by the system (e.g., "swim_speed_up_3", "stealth_jump").
- Game Constants: Imported definitions for main-only abilities, abilities exclusive to certain gear types, maximum sub-slots, etc.

### 3. Beam Search (`reconstruct_build`)
- The system iteratively builds up a set of "capstone" abilities using beam search.
- A capstone `AbilityToken` represents a target ability. For standard abilities (e.g., Ink Saver Main, Swim Speed Up), it specifies a `family` (e.g., "ink_saver_main") and a `min_ap` (minimum Ability Points to achieve). For main-only abilities (e.g., Stealth Jump), it represents the specific ability.
- The search expands states in the beam by adding new capstone abilities suggested by the `predict` function.
- The search avoids adding duplicate standard ability families or duplicate main-only ability names but allows a standard ability family's `min_ap` to be updated if a higher-AP token for that family is chosen.

### 4. Build Allocation and Legality (`Allocator` class)
- For each set of capstones explored by the beam search, the `Allocator` attempts to assign them to a legal gear configuration.

#### Main-Only Abilities
- Must occupy main ability slots.
- All main-only abilities have canonical gear piece assignments (e.g., "Stealth Jump" on "shoes", "Comeback" on "head"). These preferences are honored.
- Conflicts (e.g., trying to place two head-only abilities, or no free main slots for a required main-only ability) render the capstone set unbuildable by the allocator.

#### Standard Abilities
- Can be satisfied using main slots (contributing 10 AP) or sub-slots (each contributing 3 AP).
- The `Allocator` must ensure that the `achieved_ap` for each standard ability family meets or exceeds its `min_ap` from the capstone token.

#### Allocator's Optimal Assignment Strategy
- After placing main-only abilities, the `Allocator` uses a combinatorial approach to assign standard capstone abilities to the remaining main slots.
- It iterates through the number of standard abilities (`k`) to place on main slots, and for each `k`, through all combinations of `k` standard ability families.
- For each valid combination (where all `min_ap` are met using the chosen mains and necessary subs within the 9-sub-slot limit):
  - A score is calculated. This score prioritizes:
    1. Filling more main slots with *any* capstone abilities (main-only or standard).
    2. Achieving a higher total sum of AP for all *requested capstone abilities*.
- The `Allocator` selects the configuration (assignment of abilities to mains and subs) that achieves the highest score.

#### Constraint Adherence
- The total number of sub-slots used must not exceed `MAX_SUB_SLOTS` (9).
- The `Allocator` only uses abilities present in the input `capstones`. It does not introduce new abilities to simply fill empty gear slots if those abilities were not part of the requested capstone set.

### 5. Output
- The `reconstruct_build` function returns a single `Build` object representing the best legal gear configuration found.
- "Best" is primarily determined by the cumulative `log_score` of the capstone sequence in the beam search that led to a valid build.
- The returned `Build` contains the assignment of ability families to main gear slots, the count of sub-slots per ability family, and the total AP achieved for each family.

### 6. Data Representation
- `AbilityToken`: Stores `name`, `family`, `min_ap`, and `main_only` status. Main-only tokens have `min_ap=0` as their value is fixed at 10 AP when placed, requiring no further AP via subs from their own token.
- `Build`: Encapsulates the `mains` (slot -> family), `subs` (family -> count), and `achieved_ap` (family -> AP) of a concrete gear setup.
- `BeamState`: Tracks a hypothesis in the beam search, including the current set of `capstones`, its `log_score`, and any materialized `Build`.