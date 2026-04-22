BASE_PRICES = {
    "graphic_snake":      150,
    "graphic_skull":      110,
    "graphic_devil_babe": 120,
    "graphic_pin_up":     100,
    "graphic_marilyn":     85,
    "graphic_cross":       70,
    "graphic_other":       65,
    "graphic_logo_only":   45,
    "graphic_unknown":     55,
}

# Applied multiplicatively in the order defined — supreme first intentionally
MULTIPLIERS = [
    ("is_supreme_collab",  3.5),
    ("is_90s",             1.5),
    ("is_80s",             1.4),
    ("is_y2k",             1.1),
    ("is_japan_domestic",  1.25),
    ("has_back_graphic",   1.15),
    ("size_2xl",           1.15),
    ("size_xl",            1.10),
    ("is_other_collab",    1.20),
    ("is_guess_collab",    1.15),
    ("has_single_stitch",  1.10),
    ("is_reprint",         0.50),
    ("is_current",         0.60),
]

CONDITION_MULTIPLIERS = {
    5: 1.20,
    4: 1.10,
    3: 1.00,
    2: 0.85,
    1: 0.65,
}


def predict_price(features: dict) -> tuple:
    graphic_key = next(
        (k for k in BASE_PRICES if k != "graphic_unknown" and features.get(k)),
        "graphic_unknown",
    )
    base = float(BASE_PRICES[graphic_key])

    for key, mult in MULTIPLIERS:
        if features.get(key):
            base *= mult

    condition = max(1, min(5, int(features.get("condition", 3) or 3)))
    base *= CONDITION_MULTIPLIERS[condition]

    return round(base * 0.85, 2), round(base * 1.15, 2)
