Use the mechinterp-investigator skill to: $ARGUMENTS

## Investigation Standards

**DO NOT do shallow sweeps.** Every investigation must include:

1. **Enrichment Analysis** - Calculate presence rate in top 10% activations vs baseline
   - Use binary presence for non-scaling abilities (CB, SJ, Haunt, etc.)
   - A 2x+ enrichment is meaningful; <1.5x is weak signal

2. **Sample Size Checks** - Before claiming suppression:
   - Verify ≥20 examples exist in the "suppressed" condition
   - Zero examples = data sparsity, NOT suppression
   - Report sample sizes with all claims

3. **Semantic Consistency** - Labels must not contradict:
   - "Zombie" (embraces death mechanics) conflicts with high SSU (avoids death)
   - "Death-Averse" conflicts with high CB/SJ enrichment
   - Check that claimed suppressors are actually suppressed, not just sparse

4. **Decoder Weight Check** - Assess feature strength:
   - >30th percentile = meaningful feature
   - <10th percentile = likely noise, consider NULL label

5. **Weapon Distribution** - Don't trust top-100 sample alone:
   - Check full activation distribution across weapons
   - A weapon at 14% in top-100 may be dominant or may be noise

## Label Updates

If issues are found, update `/mnt/e/mechinterp_runs/labels/consolidated_ultra.json` with corrected labels.

## Examples of valid commands:
- `/investigate feature 1819`
- `/investigate features 1819, 552, and 14964`
- `/investigate read docs/stamper_null_feature_hits.md and validate the labels`
- `/investigate the features related to Stealth Jump that we discussed earlier`
