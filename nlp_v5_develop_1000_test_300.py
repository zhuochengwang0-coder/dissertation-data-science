"""
Steam Action RPG six-factor NLP — Version 5

PyCharm setup:
    pip install pandas numpy scipy scikit-learn

Place this script beside the labelled 1,000-row development CSV and the
labelled 300-row fixed test CSV, then click Run. Version 5 keeps the frozen
Version 4 TF-IDF classifiers and adds development-derived, auditable
sentence rules for indirect immersion language.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# =========================================================
# 1. File names, frozen model settings, and validation settings
# =========================================================

# Put this Python file and the two labelled CSV files in the same folder.
BASE_DIR = Path(__file__).resolve().parent

DEVELOPMENT_CANDIDATES = [
    "nlp_validation_sample_20260722_ai_labeled.csv",
    "nlp_validation_sample_20260722_ai_labeled(1).csv",
]

TEST_CANDIDATES = [
    "nlp_v3_independent_test_20260723_manual_labeled.csv",
    "nlp_v3_independent_test_20260723_manual_labeled(1).csv",
    "nlp_v3_independent_test_20260723_manual_labeled(2).csv",
]

OUTPUT_DIR = BASE_DIR / "nlp_v5_results"

# The 1,000-row development target is deliberately stricter. The fixed
# 300-row test uses the exploratory-study minimum selected for Version 5.
DEVELOPMENT_F1_TARGET = 0.80
TEST_F1_TARGET = 0.70
RANDOM_SEED = 20260723
CV_FOLDS = 5

FACTORS = [
    "combat",
    "challenge",
    "progression",
    "exploration",
    "narrative",
    "immersion",
]

# These settings were selected using only out-of-fold predictions from the
# 1,000-row development set. They are frozen before the 300-row test is read.
# The final prediction is:
#     V5 = (V3 + V5) rule match OR high-confidence TF-IDF LinearSVC match.
MODEL_CONFIGS = {
    "combat": {
        "features": "char",
        "c": 2.0,
        "threshold": 0.26168627393741006,
    },
    "challenge": {
        "features": "char",
        "c": 2.0,
        "threshold": 0.12492285681498969,
    },
    "progression": {
        "features": "word",
        "c": 0.25,
        "threshold": 0.3029445220112233,
    },
    "exploration": {
        "features": "both",
        "c": 0.25,
        "threshold": 0.6961137993719319,
    },
    "narrative": {
        "features": "char",
        "c": 1.0,
        "threshold": 0.2249272283651293,
    },
    "immersion": {
        "features": "char",
        "c": 2.0,
        "threshold": 0.33202377681200534,
    },
}

EXPECTED_GAMES = [
    "Cyberpunk 2077",
    "Elden Ring",
    "Hogwarts Legacy",
    "Monster Hunter Wilds",
    "The Witcher 3: Wild Hunt",
]

EXPECTED_DEVELOPMENT_COUNTS = {
    game: 200 for game in EXPECTED_GAMES
}

EXPECTED_VALIDATION_COUNTS = {
    game: 60 for game in EXPECTED_GAMES
}


# =========================================================
# 2. Version 1 base rules retained in Version 3
# =========================================================

BASE_RULES = {
    "combat": [
        ("combat", r"\bcombat\b"),
        ("fight", r"\bfights?\b|\bfighting\b"),
        ("battle", r"\bbattles?\b"),
        ("boss fight", r"\bboss\s+fights?\b"),
        ("weapon", r"\bweapons?\b"),
        ("melee", r"\bmelee\b"),
        ("ranged combat", r"\branged\s+combat\b"),
        ("gunplay", r"\bgunplay\b"),
        ("swordplay", r"\bswordplay\b"),
        ("attack", r"\battacks?\b|\battacking\b"),
        ("dodge", r"\bdodges?\b|\bdodging\b"),
        ("parry", r"\bparry\b|\bparries\b|\bparrying\b"),
        ("block attacks", r"\bblocking\b|\bblock\s+attacks?\b"),
        ("hitbox", r"\bhitboxes?\b"),
        ("moveset", r"\bmovesets?\b"),
    ],
    "challenge": [
        ("difficulty", r"\bdifficulty\b|\bdifficult\b"),
        ("challenging", r"\bchallenging\b"),
        ("challenge", r"\bchallenges?\b"),
        ("punishing", r"\bpunishing\b"),
        ("unforgiving", r"\bunforgiving\b"),
        ("difficulty spike", r"\bdifficulty\s+spikes?\b"),
        ("too hard", r"\btoo\s+hard\b"),
        (
            "very hard",
            r"\b(?:very|really|extremely|brutally)\s+hard\b",
        ),
        (
            "hard boss",
            r"\bhard(?:er|est)?\s+(?:boss|fight|enemy|mode)\b",
        ),
        ("too easy", r"\btoo\s+easy\b"),
        ("easy mode", r"\beasy\s+mode\b"),
        (
            "unfair difficulty",
            r"\bunfair\s+(?:difficulty|boss|fight)\b",
        ),
        ("balanced difficulty", r"\bbalanced\s+difficulty\b"),
    ],
    "progression": [
        ("progression", r"\bprogression\b"),
        ("level up", r"\blevel(?:s|ed|ing)?\s+up\b"),
        ("leveling", r"\blevelling\b|\bleveling\b"),
        ("upgrade", r"\bupgrades?\b|\bupgrading\b"),
        ("skill tree", r"\bskill\s+trees?\b"),
        ("character build", r"\bcharacter\s+builds?\b"),
        ("weapon build", r"\bweapon\s+builds?\b"),
        ("build variety", r"\bbuild\s+(?:variety|diversity)\b"),
        ("different builds", r"\bdifferent\s+builds\b"),
        ("equipment", r"\bequipment\b"),
        ("gear", r"\bgear\b"),
        ("loot", r"\bloot\b|\blooting\b"),
        ("character progression", r"\bcharacter\s+progression\b"),
        ("crafting", r"\bcrafting\b"),
    ],
    "exploration": [
        (
            "explore",
            r"\bexplore\b|\bexplores\b|\bexplored\b|\bexploring\b",
        ),
        ("exploration", r"\bexploration\b"),
        ("open world", r"\bopen\s+world\b"),
        (
            "discover",
            r"\bdiscover\b|\bdiscovers\b|\bdiscovered\b|"
            r"\bdiscovering\b",
        ),
        ("discovery", r"\bdiscovery\b|\bdiscoveries\b"),
        (
            "hidden area",
            r"\bhidden\s+(?:area|areas|location|locations)\b",
        ),
        (
            "secret area",
            r"\bsecret\s+(?:area|areas|location|locations)\b",
        ),
        ("world map", r"\bworld\s+map\b"),
        ("map design", r"\bmap\s+design\b"),
        (
            "environmental exploration",
            r"\benvironmental\s+exploration\b",
        ),
    ],
    "narrative": [
        ("story", r"\bstory\b|\bstories\b"),
        ("storyline", r"\bstoryline\b"),
        ("plot", r"\bplot\b"),
        ("narrative", r"\bnarrative\b"),
        ("dialogue", r"\bdialogue\b|\bdialogues\b"),
        ("quest", r"\bquests?\b"),
        ("side quest", r"\bside\s+quests?\b"),
        ("ending", r"\bendings?\b"),
        ("cutscene", r"\bcutscenes?\b"),
        ("lore", r"\blore\b"),
        ("character writing", r"\bcharacter\s+writing\b"),
        (
            "character development",
            r"\bcharacter\s+development\b",
        ),
    ],
    "immersion": [
        ("immersive", r"\bimmersive\b"),
        ("immersion", r"\bimmersion\b"),
        ("atmosphere", r"\batmosphere\b"),
        ("atmospheric", r"\batmospheric\b"),
        ("world building", r"\bworld\s+building\b"),
        ("worldbuilding", r"\bworldbuilding\b"),
        ("believable world", r"\bbelievable\s+world\b"),
        ("living world", r"\bliving\s+world\b"),
        ("world feels alive", r"\bworld\s+feels?\s+alive\b"),
        ("sense of presence", r"\bsense\s+of\s+presence\b"),
    ],
}


# =========================================================
# 3. Version 2 rules retained in Version 5
# =========================================================

V2_RULES = {
    "combat": [],
    "challenge": [
        (
            "hard or easy after game context",
            r"\b(?:game|gameplay|combat|boss(?:es)?|enem(?:y|ies)|"
            r"fight(?:s)?|monster(?:s)?|puzzle(?:s)?)\b.{0,25}"
            r"\b(?:hard|harder|hardest|easy|easier|easiest)\b",
        ),
        (
            "hard or easy before game context",
            r"\b(?:hard|harder|hardest|easy|easier|easiest)\b.{0,25}"
            r"\b(?:game|gameplay|combat|boss(?:es)?|enem(?:y|ies)|"
            r"fight(?:s)?|monster(?:s)?|puzzle(?:s)?|beginner(?:s)?|"
            r"veteran(?:s)?)\b",
        ),
        (
            "hard or easy to play",
            r"\b(?:hard|easy)\s+to\s+(?:play|learn|master|get\s+into|"
            r"beat|finish)\b",
        ),
        (
            "game is hard or easy",
            r"\bgame\s+(?:is|was|feels?|gets?)\s+(?:not\s+)?"
            r"(?:that\s+|so\s+)?(?:hard|easy)\b",
        ),
        (
            "hardest or easiest RPG",
            r"\b(?:hardest|easiest)\s+(?:souls|soulslike|rpg)\b",
        ),
        ("learning curve", r"\blearning\s+curve\b"),
        ("get gud", r"\b(?:git|get)\s+gud\b"),
        (
            "struggle with boss",
            r"\bstruggl(?:e|es|ed|ing)\b.{0,30}"
            r"\b(?:boss|fight|enemy|game|combat|win|beat)\b",
        ),
        (
            "boss struggle",
            r"\b(?:boss|fight|enemy|game|combat)\b.{0,30}"
            r"\bstruggl(?:e|es|ed|ing)\b",
        ),
        (
            "death in challenge context",
            r"\b(?:die|dies|died|dying|deaths?)\b.{0,30}"
            r"\b(?:boss|fight|enemy|game|combat|tries|attempts)\b",
        ),
        (
            "challenge context and death",
            r"\b(?:boss|fight|enemy|game|combat|tries|attempts)\b"
            r".{0,30}\b(?:die|dies|died|dying|deaths?)\b",
        ),
        ("rage or ragequit", r"\b(?:rage|raging|raged|ragequit)\b"),
        (
            "frustration in challenge context",
            r"\bfrustrat(?:e|es|ed|ing|ion)\b.{0,25}"
            r"\b(?:boss|fight|enemy|combat|difficulty)\b|"
            r"\b(?:boss|fight|enemy|combat|difficulty)\b.{0,25}"
            r"\bfrustrat(?:e|es|ed|ing|ion)\b",
        ),
        (
            "challenge pain",
            r"\b(?:game|boss|fight|enemy|combat)\b.{0,30}"
            r"\b(?:pain|painful|suffer|suffering|suffered)\b",
        ),
        (
            "pain in challenge context",
            r"\b(?:pain|painful|suffer|suffering|suffered)\b.{0,30}"
            r"\b(?:game|boss|fight|enemy|combat)\b",
        ),
        (
            "tough or brutal challenge",
            r"\b(?:game|boss|fight|enemy|combat)\b.{0,30}"
            r"\b(?:tough|brutal|relentless|demanding)\b",
        ),
        (
            "tough before challenge context",
            r"\b(?:tough|brutal|relentless|demanding)\b.{0,30}"
            r"\b(?:game|boss|fight|enemy|combat)\b",
        ),
        (
            "multiple attempts",
            r"\b(?:tries|attempts?)\b.{0,35}"
            r"\b(?:boss|beat|defeat|win|fight|enemy)\b",
        ),
        (
            "boss attempts",
            r"\b(?:boss|beat|defeat|win|fight|enemy)\b.{0,35}"
            r"\b(?:tries|attempts?)\b",
        ),
        (
            "overcome boss",
            r"\b(?:beat|defeat|overcome)\b.{0,35}"
            r"\b(?:boss|enemy|after|tries|attempts|hours)\b",
        ),
    ],
    "progression": [
        ("extended upgrade forms", r"\bupgrad(?:e|es|ed|ing|able)\b"),
        ("lvl up", r"\blvl\s*up\b"),
        ("stats", r"\bstats?\b"),
        ("grind", r"\bgrind(?:ing|y|ed)?\b"),
        ("farming", r"\bfarm(?:ing|ed|s)?\b"),
        ("armor or armour", r"\barmor\b|\barmour\b"),
        ("cyberware or implant", r"\bcyberware\b|\bimplants?\b"),
        ("loadout", r"\bloadouts?\b"),
        ("craft forms", r"\bcraft(?:s|ed|ing)?\b"),
        ("materials", r"\bmaterials?\b"),
        (
            "skill perk or stat points",
            r"\bskill\s+points?\b|\bperk\s+points?\b|"
            r"\bstat\s+points?\b",
        ),
        (
            "specific build",
            r"\b(?:character|weapon|skill|perk|equipment|gear)"
            r"\s+builds?\b",
        ),
        (
            "build before progression term",
            r"\bbuilds?\b.{0,30}\b(?:skill|perk|weapon|gear|stat|"
            r"class|character)\b",
        ),
        (
            "progression term before build",
            r"\b(?:skill|perk|weapon|gear|stat|class|character)\b"
            r".{0,30}\bbuilds?\b",
        ),
        ("weapon variety", r"\bweapon\s+variety\b"),
        (
            "rune or experience progression",
            r"\b(?:runes?|experience|xp)\b.{0,30}"
            r"\b(?:level|farm|spend|gain|earn)\b",
        ),
    ],
    "exploration": [],
    "narrative": [],
    "immersion": [
        (
            "extended immersion forms",
            r"\bimmers(?:e|es|ed|ing|ive|ion)\b",
        ),
        ("ambience", r"\bambien(?:ce|t)\b"),
        ("roleplay", r"\brole\s*play(?:ing)?\b"),
        (
            "environment feels alive",
            r"\b(?:city|town|village|environment)\s+"
            r"(?:feels|felt)\s+alive\b",
        ),
        ("feels alive", r"\bfeels?\s+alive\b"),
        ("living breathing world", r"\bliving\s+breathing\s+world\b"),
        (
            "pulled or sucked in",
            r"\b(?:suck|sucks|sucked|drag|drags|dragged|draw|draws|"
            r"drew|pull|pulls|pulled)\s+(?:me|you)\s+(?:in|into)\b",
        ),
        (
            "consumed by game",
            r"\b(?:consume|consumes|consumed)\s+(?:me|you)\b",
        ),
        ("lost myself", r"\b(?:lost|lose)\s+(?:myself|yourself)\b"),
        ("static world", r"\bstatic\s+world\b"),
        (
            "world feels real",
            r"\b(?:world|city)\s+(?:feels|felt)\s+"
            r"(?:real|realistic|authentic)\b",
        ),
    ],
}


# =========================================================
# 4. Version 3 additions
# =========================================================

# Version 3 uses sentence-level context. Weak words such as hard, easy,
# beautiful, vibe, and frustrating do not trigger a factor across sentences.
V3_RULES = {
    "combat": [],
    "challenge": [
        (
            "hard moments or sections",
            r"\bhard\s+(?:moment|moments|section|sections|part|parts)\b",
        ),
        (
            "finally beat or overcome",
            r"\bfinally\s+(?:beat|beaten|defeat(?:ed)?|won|win|"
            r"overcame|overcome|got\s+past)\b",
        ),
        (
            "keep trying or persevere",
            r"\b(?:keep|kept)\s+try(?:ing)?\b|"
            r"\btry\s+again\b|\bnever\s+give\s+up\b|"
            r"\bpersever(?:e|es|ed|ing)\b",
        ),
        (
            "stuck on challenge",
            r"\bstuck\b.{0,25}\b(?:boss|enemy|fight|level|area|game)\b|"
            r"\b(?:boss|enemy|fight|level|area|game)\b.{0,25}\bstuck\b",
        ),
        (
            "cannot get past challenge",
            r"\b(?:can't|cannot|couldn't|could\s+not|haven't|"
            r"have\s+not)\b.{0,30}\b(?:beat|defeat|get|got|gotten|"
            r"make|progress)\s+(?:past|through|beyond)\b",
        ),
        (
            "wipe against enemy",
            r"\b(?:wipe|wipes|wiped|wiping)\b.{0,25}"
            r"\b(?:against|on|to|by|boss|enemy|monster|fight|wyvern)\b|"
            r"\b(?:boss|enemy|monster|fight|wyvern)\b.{0,25}"
            r"\b(?:wipe|wipes|wiped|wiping)\b",
        ),
        (
            "stomped or whooped",
            r"\b(?:curb\s+)?stomp(?:ed|ing)?\b|"
            r"\b(?:butt|ass)\s+whooped\b|"
            r"\b(?:butt|ass)\s+kicked\b|"
            r"\b(?:bullied|destroyed)\s+by\s+(?:a\s+)?"
            r"(?:boss|enemy|monster)\b",
        ),
        (
            "slapped or thrown around by enemy",
            r"\b(?:slap|slapped|smacked|thrown\s+around)\s+by\s+"
            r"(?:a\s+)?(?:boss|enemy|monster|monsters)\b",
        ),
        (
            "break things after challenge",
            r"\b(?:make|makes|made)\s+(?:me|you)\s+"
            r"(?:break|smash|throw)\s+(?:things|controllers?|"
            r"keyboards?|screens?)\b",
        ),
        (
            "rage physical expression",
            r"\brip\s+(?:my|your)\s+(?:hair|face)\s+out\b|"
            r"\b(?:smash|bang)\s+(?:my|your)\s+head\s+against\b",
        ),
        (
            "repeated deaths",
            r"\bdie\s+(?:a\s+lot|all\s+the\s+time|repeatedly|"
            r"over\s+and\s+over)\b|"
            r"\b(?:keep|kept)\s+dying\b",
        ),
        (
            "takes time to learn",
            r"\btakes?\s+(?:a\s+while|some\s+time|time)\s+to\s+"
            r"(?:learn|master|understand|get\s+used\s+to)\b",
        ),
        (
            "getting used to mechanic",
            r"\b(?:take|takes|took)\s+(?:some\s+)?getting\s+used\s+to\b|"
            r"\b(?:get|gets|got|getting)\s+used\s+to\b.{0,25}"
            r"\b(?:combat|controls?|hitbox|mechanics?|systems?)\b",
        ),
        (
            "not intuitive mechanic",
            r"\bnot\s+(?:immediately\s+)?intuitive\b",
        ),
        (
            "overwhelming challenge context",
            r"\b(?:combat|gameplay|mechanics?|systems?|boss|fight|enemy)"
            r"\b.{0,30}\boverwhelming\b|"
            r"\boverwhelming\b.{0,30}\b(?:combat|gameplay|mechanics?|"
            r"systems?|boss|fight|enemy)\b",
        ),
        (
            "tough battles or missions",
            r"\b(?:tough|brutal|demanding)\s+"
            r"(?:battles?|missions?|encounters?|hunts?)\b",
        ),
        (
            "mistake is punished",
            r"\b(?:mistake|mistakes)\b.{0,25}"
            r"\b(?:pay|punish(?:ed)?|death|die|cost)\b",
        ),
        (
            "skill test",
            r"\b(?:test|tests)\s+(?:of\s+)?(?:your\s+)?skill\b|"
            r"\bskill\s+check\b|\bskill\s+issue\b",
        ),
        (
            "manageable with help",
            r"\bmanag(?:e)?able\b.{0,25}"
            r"\b(?:coop|co\s+op|friends?|people|players?|help)\b",
        ),
    ],
    "progression": [],
    "exploration": [],
    "narrative": [],
    "immersion": [
        (
            "theming",
            r"\btheming\b",
        ),
        (
            "vibe with environment",
            r"\b(?:vibe|vibes)\b.{0,25}"
            r"\b(?:environment|surroundings?|setting)\b|"
            r"\b(?:environment|surroundings?|setting)\b.{0,25}"
            r"\b(?:vibe|vibes)\b",
        ),
        (
            "captured the essence",
            r"\b(?:catch|catches|catched|caught|capture|captures|"
            r"captured)\s+the\s+essence\b",
        ),
        (
            "lively world",
            r"\blively\s+(?:open\s+)?world\b",
        ),
        (
            "people actually live there",
            r"\b(?:world|city|town|village|place)\b.{0,40}"
            r"\b(?:people|ppl)\s+(?:actually\s+)?live\b|"
            r"\b(?:people|ppl)\s+(?:actually\s+)?live\b.{0,40}"
            r"\b(?:world|city|town|village|place)\b",
        ),
        (
            "insanely detailed world",
            r"\binsane(?:ly)?\s+detailed\s+(?:open\s+)?world\b",
        ),
        (
            "feel like being a role",
            r"\bfeel(?:s|ing|t)?\s+like\s+(?:i(?:\s+am|'m)\s+)?"
            r"being\s+(?:a|an)\s+"
            r"(?:wizard|witch|hunter|mercenary|samurai|warrior|"
            r"student|professional)\b|"
            r"\bfeel(?:s|ing|t)?\s+like\s+(?:i(?:\s+am|'m)|"
            r"you(?:\s+are|'re))\s+(?:a|an)\s+"
            r"(?:wizard|witch|hunter|mercenary|samurai|warrior|"
            r"student|professional)\b",
        ),
        (
            "feel inside Hogwarts movie",
            r"\bfeel(?:s|ing|t)?\s+like\s+(?:i(?:\s+am|'m)|"
            r"you(?:\s+are|'re))\s+(?:really\s+)?(?:in|inside)\s+"
            r"(?:the\s+)?(?:hogwarts|harry\s+potter)\s+movie\b",
        ),
        (
            "feels like franchise",
            r"\bfeels?\s+(?:very|really|exactly)?\s*"
            r"(?:harry\s+potter|hogwarts|witcher|cyberpunk)\b",
        ),
    ],
}


# =========================================================
# 5. Version 5 development-derived immersion additions
# =========================================================

# These rules capture indirect but explicit presence/absorption language.
# They were selected from the 1,000-row development errors before the fixed
# 300-row test was evaluated. Generic praise or graphics alone do not match.
V5_RULES = {
    "combat": [],
    "challenge": [],
    "progression": [],
    "exploration": [],
    "narrative": [],
    "immersion": [
        (
            "broader feels alive",
            r"\b(?:world|city|town|village|place|environment)\b"
            r".{0,50}\balive\b",
        ),
        (
            "lost in the world",
            r"\b(?:get|gets|got|getting|be|being|become|became|"
            r"completely)?\s*lost\s+in\s+(?:the|this|its|a)?\s*"
            r"(?:world|game|city|lands?|setting|story)\b",
        ),
        (
            "escape into a world",
            r"\b(?:escape|escapes|escaped|escaping)\s+into\s+"
            r"(?:the|this|a)?\s*(?:world|game|city|story)\b",
        ),
        (
            "feel like inside a place",
            r"\bfeel(?:s|ing|t)?\s+like\b.{0,45}"
            r"\b(?:in|inside)\s+(?:an?\s+|the\s+|this\s+)?"
            r"(?:actual\s+|real\s+)?"
            r"(?:world|city|place|movie|game|hogwarts|school|battle)\b",
        ),
        (
            "like being a role or place",
            r"\blike\s+being\s+(?:a|an|at|in|inside|part\s+of)\b"
            r".{0,45}\b(?:wizard|witch|hunter|merc|mercenary|student|"
            r"warrior|samurai|hogwarts|school|world|city|story|game)\b",
        ),
        (
            "inhabit the world",
            r"\b(?:inhabit|inhabits|inhabited|inhabiting)\b",
        ),
        (
            "common immersive misspellings",
            r"\b(?:emmersive|imersive|immerssive|immursive)\b",
        ),
        (
            "feels like home",
            r"\bfeel(?:s|ing|t)?\s+like\b.{0,20}\bhome\b|"
            r"\b(?:came|come|coming)\s+(?:back\s+)?home\b",
        ),
        (
            "once in a lifetime experience",
            r"\bonce\s+in\s+a\s+lifetime\s+experience\b",
        ),
    ],
}


# =========================================================
# 6. Phrase-level exclusions
# =========================================================

# Excluded phrases are masked inside each sentence. They no longer suppress
# valid evidence elsewhere in the same review.
EXCLUDE_RULES = {
    "combat": [],
    "challenge": [
        ("hard to recommend", r"\bhard\s+to\s+recommend\b"),
        ("hard to fault", r"\bhard\s+to\s+fault\b"),
        ("hard to put down", r"\bhard\s+to\s+put\s+down\b"),
        ("hard to say", r"\bhard\s+to\s+say\b"),
        ("long and hard", r"\blong\s+and\s+hard\b"),
        ("difficulty choosing", r"\bdifficulty\s+choos(?:e|ing)\b"),
        ("hard drive", r"\bhard\s+(?:drive|disk|ware)\b"),
        ("easy to lose track", r"\beasy\s+to\s+lo+se\s+track\b"),
        (
            "frustrating non-challenge issue",
            r"\bfrustrat(?:e|es|ed|ing|ion)\b.{0,20}"
            r"\b(?:camera|ui|menus?|performance|technical|bugs?|"
            r"optimization|optimisation|seikret|navigation)\b|"
            r"\b(?:camera|ui|menus?|performance|technical|bugs?|"
            r"optimization|optimisation|seikret|navigation)\b.{0,20}"
            r"\bfrustrat(?:e|es|ed|ing|ion)\b",
        ),
    ],
    "progression": [
        (
            "natural progression from",
            r"\bnatural\s+progression\s+from\b",
        ),
        ("PC build", r"\bpc\s+builds?\b|\bcomputer\s+builds?\b"),
    ],
    "exploration": [],
    "narrative": [],
    "immersion": [
        (
            "feel like tutorial",
            r"\bfeel(?:s|ing|t)?\s+like\s+(?:i(?:\s+am|'m)|"
            r"you(?:\s+are|'re))\s+(?:still\s+)?doing\s+"
            r"(?:the\s+)?tutorial\b",
        ),
    ],
}


ALL_RULES = {
    factor: (
        BASE_RULES[factor]
        + V2_RULES[factor]
        + V3_RULES[factor]
        + V5_RULES[factor]
    )
    for factor in FACTORS
}


# =========================================================
# 7. Text normalisation and sentence-level matching
# =========================================================

def normalise_characters(text):
    """Normalise Unicode, apostrophes, case, and hyphens."""
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )
    return re.sub(r"[\u2010-\u2015-]", " ", text)


def clean_segment(text):
    """Keep English letters, numbers, apostrophes, and spaces."""
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text):
    """
    Split before removing punctuation.

    This prevents words in different sentences from satisfying one weak
    context rule, e.g. "beautiful game. easy to lose track of time".
    """
    text = normalise_characters(text)
    raw_segments = re.split(r"[.!?;]+|[\r\n]+", text)
    return [
        cleaned
        for cleaned in (clean_segment(item) for item in raw_segments)
        if cleaned
    ]


def mask_exclusions(sentence, factor):
    """Mask excluded phrases but preserve other evidence in the sentence."""
    masked = sentence
    labels = []

    for label, pattern in EXCLUDE_RULES[factor]:
        if re.search(pattern, masked):
            labels.append(label)
            masked = re.sub(pattern, " ", masked)

    masked = re.sub(r"\s+", " ", masked).strip()
    return masked, list(dict.fromkeys(labels))


def match_factor(sentences, factor):
    """Return binary prediction, matched rule labels, and exclusions."""
    matched_labels = []
    excluded_labels = []

    for sentence in sentences:
        masked_sentence, sentence_exclusions = mask_exclusions(
            sentence, factor
        )
        excluded_labels.extend(sentence_exclusions)

        if not masked_sentence:
            continue

        for label, pattern in ALL_RULES[factor]:
            if re.search(pattern, masked_sentence):
                matched_labels.append(label)

    matched_labels = list(dict.fromkeys(matched_labels))
    excluded_labels = list(dict.fromkeys(excluded_labels))

    return (
        int(bool(matched_labels)),
        ", ".join(matched_labels),
        ", ".join(excluded_labels),
    )


# =========================================================
# 8. CSV reading and structural validation
# =========================================================

def resolve_input_file(candidates, glob_pattern, dataset_name):
    """Find an attached CSV beside this script without editing its name."""
    for file_name in candidates:
        candidate = BASE_DIR / file_name
        if candidate.exists():
            return candidate

    matches = sorted(BASE_DIR.glob(glob_pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = "\n".join(f"  - {path.name}" for path in matches)
        raise FileNotFoundError(
            f"\nMore than one possible {dataset_name} CSV was found:\n"
            f"{names}\n\nKeep only the intended file beside this script, "
            "or add its exact name to the candidate list at the top."
        )

    expected = "\n".join(f"  - {name}" for name in candidates)
    raise FileNotFoundError(
        f"\nCannot find the {dataset_name} CSV beside this script.\n"
        f"Expected one of:\n{expected}"
    )


def read_csv_safely(file_path):
    """Read UTF-8 CSV, replacing only damaged punctuation if necessary."""
    try:
        return pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            encoding_errors="replace",
        )
    except TypeError:
        with open(
            file_path,
            mode="r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as csv_file:
            return pd.read_csv(csv_file)


def validate_dataset(
    df,
    dataset_name,
    expected_rows,
    expected_game_counts,
):
    """Validate structure without modifying the original labels."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.lstrip("\ufeff")
    )

    required_columns = [
        "recommendation_id",
        "game",
        "review",
        "high_playtime",
        *[f"{factor}_manual" for factor in FACTORS],
    ]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            f"{missing_columns}"
        )

    if len(df) != expected_rows:
        raise ValueError(
            f"{dataset_name}: expected {expected_rows} rows, "
            f"found {len(df)}."
        )

    if df["recommendation_id"].isna().any():
        raise ValueError(
            f"{dataset_name}: recommendation_id contains blanks."
        )

    duplicate_mask = df["recommendation_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = df.loc[
            duplicate_mask, "recommendation_id"
        ].tolist()
        raise ValueError(
            f"{dataset_name}: duplicate IDs found:\n{duplicate_ids}"
        )

    empty_review_mask = (
        df["review"].isna()
        | df["review"].fillna("").astype(str).str.strip().eq("")
    )
    if empty_review_mask.any():
        empty_ids = df.loc[
            empty_review_mask, "recommendation_id"
        ].tolist()
        raise ValueError(
            f"{dataset_name}: blank reviews found for IDs:\n"
            f"{empty_ids}"
        )

    actual_game_counts = (
        df["game"].value_counts().sort_index().to_dict()
    )
    if actual_game_counts != dict(
        sorted(expected_game_counts.items())
    ):
        raise ValueError(
            f"{dataset_name}: game distribution is incorrect."
            f"\nActual: {actual_game_counts}"
            f"\nExpected: {expected_game_counts}"
        )

    df["high_playtime"] = pd.to_numeric(
        df["high_playtime"], errors="coerce"
    )
    invalid_playtime_mask = (
        df["high_playtime"].isna()
        | ~df["high_playtime"].isin([0, 1])
    )
    if invalid_playtime_mask.any():
        invalid_ids = df.loc[
            invalid_playtime_mask, "recommendation_id"
        ].tolist()
        raise ValueError(
            f"{dataset_name}: invalid high_playtime values for IDs:\n"
            f"{invalid_ids}"
        )
    df["high_playtime"] = df["high_playtime"].astype(int)

    for factor in FACTORS:
        manual_column = f"{factor}_manual"
        df[manual_column] = pd.to_numeric(
            df[manual_column], errors="coerce"
        )
        invalid_label_mask = (
            df[manual_column].isna()
            | ~df[manual_column].isin([0, 1])
        )
        if invalid_label_mask.any():
            invalid_ids = df.loc[
                invalid_label_mask, "recommendation_id"
            ].tolist()
            raise ValueError(
                f"{dataset_name}: {manual_column} contains blank or "
                f"non-binary values for IDs:\n{invalid_ids}"
            )

        df[manual_column] = df[manual_column].astype(int)
        if set(df[manual_column].unique()) != {0, 1}:
            raise ValueError(
                f"{dataset_name}: {manual_column} does not contain "
                "both 0 and 1."
            )

    print(f"\n{dataset_name} structure check passed:")
    print(f"  Total rows: {len(df)}")
    print("  Duplicate IDs: 0")
    for game, count in sorted(actual_game_counts.items()):
        print(f"  {game}: {count}")

    return df


# =========================================================
# 9. Prediction and metric calculation
# =========================================================

def add_rule_predictions(df):
    """Add frozen Version 3 + Version 5 rules and matched phrases."""
    result = df.copy()
    result["review_sentences"] = result["review"].apply(
        split_into_sentences
    )

    for factor in FACTORS:
        matches = result["review_sentences"].apply(
            lambda sentences, current_factor=factor: match_factor(
                sentences, current_factor
            )
        )
        result[f"{factor}_rule_mention"] = matches.str[0]
        result[f"{factor}_matched_terms"] = matches.str[1]
        result[f"{factor}_excluded_terms"] = matches.str[2]

    result["review_clean"] = result["review_sentences"].apply(
        lambda sentences: " | ".join(sentences)
    )
    result = result.drop(columns=["review_sentences"])
    return result


def vectorize_train_and_eval(train_text, eval_text, feature_kind):
    """Fit TF-IDF only on the training text and transform both partitions."""
    train_parts = []
    eval_parts = []

    if feature_kind in {"word", "both"}:
        word_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            max_features=40000,
        )
        train_parts.append(word_vectorizer.fit_transform(train_text))
        eval_parts.append(word_vectorizer.transform(eval_text))

    if feature_kind in {"char", "both"}:
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=80000,
        )
        train_parts.append(char_vectorizer.fit_transform(train_text))
        eval_parts.append(char_vectorizer.transform(eval_text))

    if not train_parts:
        raise ValueError(f"Unknown feature type: {feature_kind}")
    if len(train_parts) == 1:
        return train_parts[0], eval_parts[0]

    return (
        hstack(train_parts, format="csr"),
        hstack(eval_parts, format="csr"),
    )


def build_oof_development_scores(development_df):
    """
    Build five-fold out-of-fold scores for the 1,000-row development set.

    Each row is scored by a model that was not trained on that row. Frozen
    thresholds are applied later; the 300-row test is not used here.
    """
    texts = development_df["review"].fillna("").astype(str)
    scores_by_factor = {}

    for factor in FACTORS:
        labels = development_df[f"{factor}_manual"].to_numpy(dtype=int)
        positive_count = int(labels.sum())
        negative_count = int(len(labels) - positive_count)
        if min(positive_count, negative_count) < CV_FOLDS:
            raise ValueError(
                f"{factor}: too few examples for {CV_FOLDS}-fold "
                "stratified cross-validation."
            )

        config = MODEL_CONFIGS[factor]
        oof_scores = np.zeros(len(development_df), dtype=float)
        splitter = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )

        for train_index, valid_index in splitter.split(texts, labels):
            x_train, x_valid = vectorize_train_and_eval(
                texts.iloc[train_index],
                texts.iloc[valid_index],
                config["features"],
            )
            model = LinearSVC(
                C=config["c"],
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )
            model.fit(x_train, labels[train_index])
            oof_scores[valid_index] = model.decision_function(x_valid)

        scores_by_factor[factor] = oof_scores

    return scores_by_factor


def fit_development_and_score_test(development_df, test_df):
    """
    Fit each frozen classifier on all 1,000 development rows, then score
    the untouched 300-row test set.
    """
    development_text = (
        development_df["review"].fillna("").astype(str)
    )
    test_text = test_df["review"].fillna("").astype(str)
    scores_by_factor = {}

    for factor in FACTORS:
        config = MODEL_CONFIGS[factor]
        labels = development_df[f"{factor}_manual"].to_numpy(dtype=int)
        x_train, x_test = vectorize_train_and_eval(
            development_text,
            test_text,
            config["features"],
        )
        model = LinearSVC(
            C=config["c"],
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        model.fit(x_train, labels)
        scores_by_factor[factor] = model.decision_function(x_test)

    return scores_by_factor


def apply_hybrid_predictions(rule_df, scores_by_factor):
    """Combine frozen V3 rules with high-confidence TF-IDF predictions."""
    result = rule_df.copy()

    for factor in FACTORS:
        threshold = MODEL_CONFIGS[factor]["threshold"]
        scores = np.asarray(scores_by_factor[factor], dtype=float)
        ml_prediction = (scores >= threshold).astype(int)
        rule_prediction = (
            result[f"{factor}_rule_mention"].to_numpy(dtype=int)
        )
        final_prediction = np.maximum(
            rule_prediction, ml_prediction
        ).astype(int)

        result[f"{factor}_ml_score"] = np.round(scores, 8)
        result[f"{factor}_ml_threshold"] = threshold
        result[f"{factor}_ml_mention"] = ml_prediction
        result[f"{factor}_mention"] = final_prediction
        result[f"{factor}_decision_source"] = np.select(
            [
                (rule_prediction == 1) & (ml_prediction == 1),
                rule_prediction == 1,
                ml_prediction == 1,
            ],
            ["rule_and_tfidf", "rule_only", "tfidf_only"],
            default="neither",
        )

    return result


def calculate_metrics(df, dataset_name, f1_target):
    """Calculate category-level binary Precision, Recall, and F1."""
    rows = []

    for factor in FACTORS:
        y_true = df[f"{factor}_manual"]
        y_pred = df[f"{factor}_mention"]

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="binary",
                zero_division=0,
            )
        )
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        rows.append(
            {
                "dataset": dataset_name,
                "factor": factor,
                "sample_size": len(df),
                "manual_positive": int(y_true.sum()),
                "predicted_positive": int(y_pred.sum()),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "accuracy": round(
                    float((y_true == y_pred).mean()), 4
                ),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
                "f1_target": f1_target,
                "f1_target_met": (
                    "Yes" if f1 >= f1_target else "No"
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_per_game_metrics(df, dataset_name):
    """Calculate the same metrics separately for each game."""
    rows = []

    for game in EXPECTED_GAMES:
        game_df = df[df["game"] == game]

        for factor in FACTORS:
            y_true = game_df[f"{factor}_manual"]
            y_pred = game_df[f"{factor}_mention"]

            precision, recall, f1, _ = (
                precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    average="binary",
                    zero_division=0,
                )
            )
            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).ravel()

            rows.append(
                {
                    "dataset": dataset_name,
                    "game": game,
                    "factor": factor,
                    "sample_size": len(game_df),
                    "manual_positive": int(y_true.sum()),
                    "predicted_positive": int(y_pred.sum()),
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1_score": round(float(f1), 4),
                    "true_positive": int(tp),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_negative": int(tn),
                }
            )

    return pd.DataFrame(rows)


def build_error_cases(df, dataset_name):
    """Collect false positives and false negatives for every factor."""
    error_frames = []

    for factor in FACTORS:
        manual_column = f"{factor}_manual"
        prediction_column = f"{factor}_mention"
        error_mask = df[manual_column] != df[prediction_column]

        factor_errors = df.loc[
            error_mask,
            [
                "recommendation_id",
                "game",
                "review",
                manual_column,
                f"{factor}_rule_mention",
                f"{factor}_ml_score",
                f"{factor}_ml_threshold",
                f"{factor}_ml_mention",
                prediction_column,
                f"{factor}_decision_source",
                f"{factor}_matched_terms",
                f"{factor}_excluded_terms",
            ],
        ].copy()

        if factor_errors.empty:
            continue

        factor_errors.insert(0, "dataset", dataset_name)
        factor_errors.insert(3, "factor", factor)
        factor_errors["error_type"] = factor_errors.apply(
            lambda row: (
                "False Negative"
                if row[manual_column] == 1
                and row[prediction_column] == 0
                else "False Positive"
            ),
            axis=1,
        )
        factor_errors = factor_errors.rename(
            columns={
                manual_column: "manual_label",
                f"{factor}_rule_mention": "rule_prediction",
                f"{factor}_ml_score": "ml_score",
                f"{factor}_ml_threshold": "ml_threshold",
                f"{factor}_ml_mention": "ml_prediction",
                prediction_column: "nlp_prediction",
                f"{factor}_decision_source": "decision_source",
                f"{factor}_matched_terms": "matched_terms",
                f"{factor}_excluded_terms": "excluded_terms",
            }
        )
        error_frames.append(
            factor_errors[
                [
                    "dataset",
                    "recommendation_id",
                    "game",
                    "factor",
                    "error_type",
                    "manual_label",
                    "rule_prediction",
                    "ml_score",
                    "ml_threshold",
                    "ml_prediction",
                    "nlp_prediction",
                    "decision_source",
                    "matched_terms",
                    "excluded_terms",
                    "review",
                ]
            ]
        )

    if error_frames:
        return pd.concat(error_frames, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "dataset",
            "recommendation_id",
            "game",
            "factor",
            "error_type",
            "manual_label",
            "rule_prediction",
            "ml_score",
            "ml_threshold",
            "ml_prediction",
            "nlp_prediction",
            "decision_source",
            "matched_terms",
            "excluded_terms",
            "review",
        ]
    )


# =========================================================
# 10. Evaluation, output, and reporting
# =========================================================

def evaluate_and_save(
    predictions_df,
    dataset_name,
    output_stem,
    f1_target,
):
    """Evaluate one already-predicted dataset and save auditable outputs."""
    metrics_df = calculate_metrics(
        predictions_df, dataset_name, f1_target
    )
    per_game_df = calculate_per_game_metrics(
        predictions_df, dataset_name
    )
    errors_df = build_error_cases(predictions_df, dataset_name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(
        OUTPUT_DIR / f"{output_stem}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_df.to_csv(
        OUTPUT_DIR / f"{output_stem}_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_game_df.to_csv(
        OUTPUT_DIR / f"{output_stem}_metrics_by_game.csv",
        index=False,
        encoding="utf-8-sig",
    )
    errors_df.to_csv(
        OUTPUT_DIR / f"{output_stem}_error_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics_df, per_game_df, errors_df


def save_frozen_configuration():
    """Save the exact settings used by Version 5."""
    rows = []
    for factor in FACTORS:
        config = MODEL_CONFIGS[factor]
        rows.append(
            {
                "factor": factor,
                "rule_base": "frozen_v3_plus_v5_sentence_rules",
                "tfidf_features": config["features"],
                "linear_svc_c": config["c"],
                "decision_threshold": config["threshold"],
                "combination": "rule OR high-confidence TF-IDF",
                "development_target": DEVELOPMENT_F1_TARGET,
                "external_test_target": TEST_F1_TARGET,
                "cv_folds": CV_FOLDS,
                "random_seed": RANDOM_SEED,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "v5_frozen_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )


def print_metrics(metrics_df, heading):
    print("\n" + "=" * 78)
    print(heading)
    print("=" * 78)

    display_columns = [
        "factor",
        "manual_positive",
        "predicted_positive",
        "precision",
        "recall",
        "f1_score",
        "false_positive",
        "false_negative",
        "f1_target_met",
    ]
    print(metrics_df[display_columns].to_string(index=False))

    macro_f1 = metrics_df["f1_score"].mean()
    f1_target = float(metrics_df["f1_target"].iloc[0])
    passed = metrics_df.loc[
        metrics_df["f1_target_met"] == "Yes", "factor"
    ].tolist()
    failed = metrics_df.loc[
        metrics_df["f1_target_met"] == "No", "factor"
    ].tolist()

    print(f"\nMacro mean F1: {macro_f1:.4f}")
    print(f"Pass threshold for every factor: F1 >= {f1_target:.2f}")
    print(f"Factors passed: {len(passed)}/{len(FACTORS)}")
    if passed:
        print("Passed: " + ", ".join(passed))
    if failed:
        print("Not passed: " + ", ".join(failed))

    return len(failed) == 0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Develop Version 5 on 1,000 labelled reviews with "
            "out-of-fold predictions, then test the frozen model on "
            "a fixed 300-row labelled sample."
        )
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help=(
            "Run only the 1,000-row development stage. This is useful "
            "for verifying the frozen development result before the "
            "300-row test is opened."
        ),
    )
    parser.add_argument(
        "--development-file",
        type=Path,
        default=None,
        help="Optional explicit path to the 1,000-row labelled CSV.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Optional explicit path to the 300-row labelled CSV.",
    )
    return parser.parse_args()


# =========================================================
# 11. Main program: develop on 1,000, freeze, then test on 300
# =========================================================

def main():
    args = parse_arguments()
    development_path = (
        args.development_file.resolve()
        if args.development_file is not None
        else resolve_input_file(
            DEVELOPMENT_CANDIDATES,
            "nlp_validation_sample_20260722_ai_labeled*.csv",
            "1,000-row development",
        )
    )

    print(f"\nDevelopment file:\n{development_path}")
    development_df = validate_dataset(
        read_csv_safely(development_path),
        dataset_name="development_1000",
        expected_rows=1000,
        expected_game_counts=EXPECTED_DEVELOPMENT_COUNTS,
    )

    development_rules = add_rule_predictions(development_df)
    development_scores = build_oof_development_scores(
        development_df
    )
    development_predictions = apply_hybrid_predictions(
        development_rules, development_scores
    )
    (
        development_metrics,
        _development_per_game,
        _development_errors,
    ) = evaluate_and_save(
        development_predictions,
        dataset_name="development_1000_oof",
        output_stem="development_1000_v5_oof",
        f1_target=DEVELOPMENT_F1_TARGET,
    )
    save_frozen_configuration()

    development_passed = print_metrics(
        development_metrics,
        "VERSION 5: 1,000-ROW DEVELOPMENT RESULTS "
        "(5-FOLD OUT-OF-FOLD)",
    )

    if not development_passed:
        raise RuntimeError(
            "\nVersion 5 development failed: at least one factor did "
            f"not reach F1 >= {DEVELOPMENT_F1_TARGET:.2f}. The "
            "300-row test was not opened."
        )

    if args.development_only:
        print(
            "\nDEVELOPMENT-ONLY COMPLETE: all six factors reached "
            f"F1 >= {DEVELOPMENT_F1_TARGET:.2f}. The 300-row test "
            "was not read."
        )
        print(f"\nGenerated result folder:\n{OUTPUT_DIR}")
        return

    test_path = (
        args.test_file.resolve()
        if args.test_file is not None
        else resolve_input_file(
            TEST_CANDIDATES,
            "nlp_v3_independent_test_20260723_manual_labeled*.csv",
            "300-row external test",
        )
    )
    print(f"\nExternal test file:\n{test_path}")
    test_df = validate_dataset(
        read_csv_safely(test_path),
        dataset_name="external_test_300",
        expected_rows=300,
        expected_game_counts=EXPECTED_VALIDATION_COUNTS,
    )

    development_ids = set(
        development_df["recommendation_id"].astype(str).str.strip()
    )
    test_ids = set(
        test_df["recommendation_id"].astype(str).str.strip()
    )
    overlap = sorted(development_ids.intersection(test_ids))
    if overlap:
        raise ValueError(
            "\nThe development and test sets are not independent. "
            f"Found {len(overlap)} overlapping recommendation IDs."
        )
    print("  Development/test recommendation-ID overlap: 0")

    test_rules = add_rule_predictions(test_df)
    test_scores = fit_development_and_score_test(
        development_df, test_df
    )
    test_predictions = apply_hybrid_predictions(
        test_rules, test_scores
    )
    test_metrics, _test_per_game, _test_errors = evaluate_and_save(
        test_predictions,
        dataset_name="external_test_300",
        output_stem="external_test_300_v5",
        f1_target=TEST_F1_TARGET,
    )
    test_passed = print_metrics(
        test_metrics,
        "VERSION 5: 300-ROW EXTERNAL TEST RESULTS",
    )

    comparison_df = pd.concat(
        [development_metrics, test_metrics],
        ignore_index=True,
    )
    comparison_df.to_csv(
        OUTPUT_DIR / "v5_metrics_comparison_1000_vs_300.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("VERSION 5 OVERALL DECISION")
    print("=" * 78)
    if development_passed and test_passed:
        print(
            "PASS: all six factors reached F1 >= 0.80 in the "
            "1,000-row out-of-fold development evaluation and "
            "F1 >= 0.70 on the 300-row fixed test."
        )
    else:
        print(
            "EXTERNAL TEST NOT FULLY PASSED: development reached "
            "F1 >= 0.80 for all six factors, but at least one factor "
            "was below F1 = 0.70 on the 300-row fixed test."
        )

    print(
        "\nMethod note: the TF-IDF settings and decision thresholds "
        "and the Version 5 rule additions were selected from the "
        "1,000-row development set. The 300 rows have distinct IDs, "
        "but this fixed benchmark has already been used for earlier "
        "version comparisons and is not a never-seen final test."
    )
    print(f"\nGenerated result folder:\n{OUTPUT_DIR}")
    print("\nFiles include:")
    print("  1. Row-level rule, TF-IDF, and final V5 predictions")
    print("  2. Overall metrics for both datasets")
    print("  3. Per-game metrics for both datasets")
    print("  4. False-positive and false-negative cases")
    print("  5. Frozen configuration and 1,000-vs-300 comparison")


if __name__ == "__main__":
    main()
