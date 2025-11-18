# Build Reconstruction

The model predicts probabilities for individual ability tokens, but Splatoon 3 builds have physical constraints. You can't just pick the top-N predicted abilities. This module converts model predictions into legal, equippable gear configurations.

## The Problem

**Model output:**
- Probabilities for each ability token
- Example: `swim_speed_12: 0.85, ninja_squid: 0.78, swim_speed_6: 0.72`

**What we need:**
- 3 gear pieces (head, clothes, shoes)
- Each with 1 main ability + 3 sub-slots
- Respects game constraints (main-only abilities, AP requirements, etc.)

**Why it's hard:**
- Ability tokens aren't independent. Picking `swim_speed_12` means you're committing to 12 AP of Swim Speed.
- Some tokens overlap. `swim_speed_6` is redundant if you already have `swim_speed_12`.
- Main-only abilities have fixed gear slots. Ninja Squid must go on clothes.
- You have limited slots. Can't fit everything the model wants.

## The Solution: Beam Search + Constraint Satisfaction

Two-phase approach:

1. **Beam search** finds the best set of "capstone" abilities (the abilities you want, ignoring physical constraints)
2. **Allocator** maps capstones to actual gear slots, respecting game mechanics

## Phase 1: Beam Search

Explores different combinations of ability tokens to maximize model probability while respecting ability semantics.

### Starting Point: Greedy Closure

Before beam search, we do a greedy pass:

```python
while True:
    predictions = model(current_context, weapon)
    candidates = [token for token in predictions if prob >= 0.5]
    if no new candidates:
        break
    add_all_candidates_to_context()
```

**Why greedy first?**
- The model threshold is 0.5 (trained with BCEWithLogitsLoss)
- Abilities with >50% probability are clearly wanted
- Grab the "obvious" choices before exploring alternatives

**Why threshold 0.5?**
- Matches the training decision boundary
- Anything below 0.5 the model thinks is more likely absent than present
- Conservative but reliable starting point

### Beam Search Refinement

Starting from the greedy closure, beam search explores variations:

```python
beam = [greedy_result]
for step in range(max_steps):
    for state in beam:
        predictions = model(state.context, weapon)
        # Consider adding/replacing abilities
        new_states = generate_successors(state, predictions)
    beam = keep_top_k(new_states, k=beam_width)
```

**What gets explored:**
- Adding abilities below the 0.5 threshold
- Replacing lower-AP tokens with higher-AP versions (e.g., swap `swim_speed_6` for `swim_speed_12`)
- Different ability combinations

**Scoring:**
```python
score = sum(log(prob) for each ability)
      + TOKEN_BONUS * num_abilities
      - ALPHA * penalty
```

- **Log probabilities:** Favors high-confidence predictions
- **Token bonus:** `log(2.0)` per ability. Encourages adding more abilities (exploration).
- **Penalty:** Weighted term (alpha=0.3) discourages unbuildable sets

**Constraints during search:**
- Can't have duplicate ability families (one Swim Speed level only)
- Can't have duplicate main-only abilities
- Can upgrade within a family (`swim_speed_6` → `swim_speed_12`)

### Example Beam Search

```
Initial (greedy): [swim_speed_12, ninja_squid, quick_respawn_6]
Score: log(0.85) + log(0.78) + log(0.72) + 3*log(2.0)

Successor 1: Add ink_saver_main_6 (prob 0.48)
Score: previous + log(0.48) + log(2.0)

Successor 2: Upgrade quick_respawn_6 to quick_respawn_12 (prob 0.55)
Score: log(0.85) + log(0.78) + log(0.55) + log(0.48) + 4*log(2.0)

Keep top 10 successors, repeat
```

The beam explores different ability combinations, but doesn't worry about physical constraints yet.

## Phase 2: Build Allocation

The allocator takes a set of capstone abilities and tries to fit them into actual gear slots.

### Splatoon 3 Constraints

**Gear structure:**
- 3 pieces: head, clothes, shoes
- Each piece: 1 main slot + 3 sub-slots
- Total: 3 mains + 9 subs

**Main-only abilities:**
- Must go in main slots
- Fixed gear assignments:
  - Ninja Squid → clothes
  - Stealth Jump → shoes
  - Comeback → head
  - etc.

**Standard abilities:**
- Can use mains (10 AP) or subs (3 AP each)
- Need to reach minimum AP threshold from the capstone token
- Example: `swim_speed_12` needs at least 12 AP of Swim Speed

### Allocator Strategy

**Step 1: Place main-only abilities**
```python
for ability in capstones:
    if ability.main_only:
        slot = ability.canonical_slot  # head/clothes/shoes
        if slot already occupied:
            return UNBUILDABLE
        mains[slot] = ability
```

If two main-only abilities want the same slot (e.g., Ninja Squid and Haunt both want clothes), the build fails.

**Step 2: Assign standard abilities**

This is an optimization problem:
- Decision: which standard abilities go in main slots vs subs?
- Constraint: total subs ≤ 9
- Objective: maximize filled mains + total AP

The allocator tries all combinations:
```python
for num_on_mains in range(num_free_mains + 1):
    for combination in combinations(standard_abilities, num_on_mains):
        # Check if this combination is feasible
        subs_needed = calculate_sub_requirements(combination)
        if subs_needed > 9:
            continue
        score = num_on_mains + sum(achieved_ap)
        if score > best_score:
            best = this_combination
```

**Scoring:**
```python
score = mains_filled_with_capstones + sum(achieved_ap for all capstones)
```

- Prioritize filling mains (wasted slots are bad)
- Maximize total AP (over-delivering on thresholds is good)

**Example:**

Capstones: `[swim_speed_12, ninja_squid, ink_saver_6]`

Main-only: ninja_squid → clothes

Remaining mains: head, shoes (2 slots)
Standard capstones: swim_speed_12, ink_saver_6

Try assigning 0 to mains:
- swim_speed_12: 4 subs (12 AP)
- ink_saver_6: 2 subs (6 AP)
- Total subs: 6 ≤ 9 ✓
- Score: 0 (no mains filled) + 18 AP = 18

Try assigning 1 to mains:
- swim_speed → main (10 AP) + 1 sub (3 AP) = 13 AP
- ink_saver_6: 2 subs (6 AP)
- Total subs: 3 ≤ 9 ✓
- Score: 1 + 19 AP = 20 ✓ Better

Try assigning 2 to mains:
- swim_speed → main (10 AP) + 1 sub (3 AP) = 13 AP
- ink_saver → main (10 AP), already exceeds 6 AP target
- Total subs: 1 ≤ 9 ✓
- Score: 2 + 23 AP = 25 ✓ Best

Final build:
- Head: Swim Speed Up (main) + Swim Speed Up (1 sub)
- Clothes: Ninja Squid (main)
- Shoes: Ink Saver Main (main)

### Unbuildable Sets

Some capstone combinations can't be allocated:

**Too many main-only abilities:**
- 4 main-only abilities, only 3 gear pieces
- No valid assignment

**Conflicting main-only abilities:**
- Ninja Squid (clothes) + Haunt (clothes)
- Both want the same slot

**Too many standard abilities:**
- Needs more than 9 subs to satisfy all AP requirements
- Example: 6 abilities each needing 15 AP

When a set is unbuildable, the beam search gets a penalty. This guides the search away from infeasible combinations.

## Why This Design?

**Why not greedily pick top N abilities?**
- Might violate constraints
- Doesn't account for ability interactions
- Misses better combinations with slightly lower individual probabilities

**Why beam search instead of exact optimization?**
- Exact optimization is NP-hard (combinatorial explosion)
- Beam search finds good solutions in reasonable time
- Beam width = 10 is a good trade-off

**Why separate search and allocation?**
- Search handles model probabilities (soft constraints)
- Allocation handles game mechanics (hard constraints)
- Cleaner separation of concerns

**Why the token bonus?**
- Without it, the search prefers fewer high-probability abilities
- We want to explore adding more abilities
- `log(2.0)` means adding an ability at 50% probability is neutral
- Encourages exploration while still respecting model confidence

## Tuning Parameters

**Beam width:**
- Default: 10
- Higher: better solutions, slower
- Lower: faster, might miss optimal builds

**Max steps:**
- Default: 5
- Controls search depth
- Typical builds converge in 3-4 steps

**Token bonus:**
- Default: `log(2.0)` ≈ 0.693
- Higher: encourages more abilities
- Lower: favors fewer high-confidence abilities

**Alpha (penalty weight):**
- Default: 0.3
- Higher: avoid unbuildable sets more aggressively
- Lower: explore more freely, allocator does the filtering

## Output

The final `Build` object contains:
```python
Build(
    mains={
        "head": "swim_speed_up",
        "clothes": "ninja_squid",
        "shoes": "ink_saver_main",
    },
    subs={
        "swim_speed_up": 1,  # 1 sub of Swim Speed
    },
    achieved_ap={
        "swim_speed_up": 13,
        "ninja_squid": 10,
        "ink_saver_main": 10,
    },
)
```

This is a legal, equippable gear set that maximizes the model's predicted ability preferences.

## Code Location

Main algorithm: `src/splatnlp/utils/reconstruct/beam_search.py`
- `reconstruct_build()`: Entry point
- `greedy_closure()`: Phase 1 greedy pass
- `beam_search()`: Phase 2 exploration

Allocator: `src/splatnlp/utils/reconstruct/allocator.py`
- `Allocator.allocate()`: Constraint satisfaction
- `_score_allocation()`: Scoring function

Data structures: `src/splatnlp/utils/reconstruct/classes.py`
- `AbilityToken`: Represents an ability with metadata
- `Build`: Represents a complete gear configuration
- `BeamState`: Tracks a hypothesis in beam search
