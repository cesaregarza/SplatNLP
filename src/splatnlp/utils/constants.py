import math

MAIN_ONLY_ABILITIES = [
    "comeback",
    "last_ditch_effort",
    "opening_gambit",
    "tenacity",
    "ability_doubler",
    "haunt",
    "ninja_squid",
    "respawn_punisher",
    "thermal_ink",
    "drop_roller",
    "object_shredder",
    "stealth_jump",
]
STANDARD_ABILITIES = [
    "ink_recovery_up",
    "ink_resistance_up",
    "ink_saver_main",
    "ink_saver_sub",
    "intensify_action",
    "quick_respawn",
    "quick_super_jump",
    "run_speed_up",
    "special_charge_up",
    "special_power_up",
    "special_saver",
    "sub_power_up",
    "sub_resistance_up",
    "swim_speed_up",
]
BUCKET_THRESHOLDS = [3, 6, 12, 15, 21, 29, 38, 51, 57]
REMOVE_COLUMNS = [
    "kill-assist",
    "kill",
    "assist",
    "death",
    "special",
    "inked",
    "abilities",
    "player_no",
    "lobby",
    "mode",
    "win",
    "weapon",
    "ability_hash",
    "team",
]
MASK = "<MASK>"
PAD = "<PAD>"
NULL = "<NULL>"
SPECIAL_TOKENS = [MASK, PAD, NULL]
SEASONS_WITHOUT_NEW_WEAPONS = [8, 9]
BUFFER_DAYS_FOR_MAJOR_PATCH = 14
BUFFER_DAYS_FOR_MINOR_PATCH = 7
TARGET_WEAPON_WINRATE = 0.6

HEADGEAR_ABILITIES = [
    "comeback",
    "last_ditch_effort",
    "opening_gambit",
    "tenacity",
]
CLOTHING_ABILITIES = [
    "ability_doubler",
    "haunt",
    "ninja_squid",
    "respawn_punisher",
    "thermal_ink",
]
SHOES_ABILITIES = [
    "drop_roller",
    "object_shredder",
    "stealth_jump",
]
CANONICAL_MAIN_ONLY_ABILITIES = {
    **{name: "head" for name in HEADGEAR_ABILITIES},
    **{name: "clothes" for name in CLOTHING_ABILITIES},
    **{name: "shoes" for name in SHOES_ABILITIES},
}
TOKEN_BONUS = math.log(2.0)
