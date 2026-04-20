BRANDS = {
    "harley": {
        "label": "Harley Davidson",
        "short": "HD",
        "csv": "data/raw/harley_features.csv",
        "signals": [
            {
                "id": "era", "label": "Era", "type": "select", "ai_key": "era",
                "options": [["unknown","Unknown"],["80s","80s"],["90s","90s"],["y2k","Y2K / 2000s"]],
                "tooltip": "80s = bold/simple graphics. 90s = detailed/airbrushed. Y2K = 2000–2006. Check neck tag copyright year.",
            },
            {
                "id": "size", "label": "Size", "type": "select", "ai_key": "size",
                "options": [["unknown","Unknown"],["S","S"],["M","M"],["L","L"],["XL","XL"],["2XL","2XL"]],
                "tooltip": "Check neck or side seam tag. Vintage tees often run 1–2 sizes small vs modern sizing.",
            },
            {
                "id": "emblem", "label": "3D Emblem", "type": "bool", "ai_key": "has_3d_emblem",
                "tooltip": "Harley shield logo is raised/embossed — not flat-printed. Common on 90s dealer tees, adds significant value.",
            },
            {
                "id": "single_stitch", "label": "Single Stitch", "type": "bool", "ai_key": "has_single_stitch",
                "tooltip": "Single row of stitching along sleeve and bottom hems. Vintage construction pre-1996. Flip the sleeve to check.",
            },
            {
                "id": "location", "label": "Location Name", "type": "bool", "ai_key": "has_location_name",
                "tooltip": "Specific city, state, or dealership name printed on the shirt (e.g. \"Daytona Beach\", \"Las Vegas H-D\").",
            },
            {
                "id": "event", "label": "Event Tee", "type": "bool", "ai_key": "is_event_tee",
                "tooltip": "References a specific rally, run, or event — e.g. \"Sturgis 1994\", \"HOG Rally\", \"Bike Week\".",
            },
        ],
        "vision_prompt": """You are an expert vintage Harley Davidson t-shirt appraiser.
Analyze this image and extract the following signals. Return ONLY valid JSON, no other text:
{
    "era": "80s or 90s or y2k or unknown",
    "size": "S or M or L or XL or 2XL or unknown",
    "has_3d_emblem": true or false,
    "has_single_stitch": true or false,
    "has_location_name": true or false,
    "is_event_tee": true or false
}
Rules:
- 3D emblem = Harley shield is raised/embossed, not flat-printed
- Single stitch = single row sleeve hem stitching (vintage pre-1996 indicator)
- Location name = specific city, state, or dealership on the shirt
- Event tee = references a rally, run, or specific event
- 80s graphics tend to be bolder/simpler, 90s more detailed/airbrushed
- Look at tags for size and era clues""",
    },

    "ed_hardy": {
        "label": "Ed Hardy",
        "short": "EH",
        "csv": "data/raw/ed_hardy_features.csv",
        "signals": [
            {
                "id": "era", "label": "Era", "type": "select", "ai_key": "era",
                "options": [["unknown","Unknown"],["y2k","Y2K / 2000s"],["10s","2010s"]],
                "tooltip": "Y2K (2000–2006) is peak Don Ed Hardy / Christian Audigier era and most valuable. 2010s pieces are lower value.",
            },
            {
                "id": "size", "label": "Size", "type": "select", "ai_key": "size",
                "options": [["unknown","Unknown"],["S","S"],["M","M"],["L","L"],["XL","XL"],["2XL","2XL"]],
                "tooltip": "Standard US sizing. Check neck tag.",
            },
            {
                "id": "rhinestones", "label": "Rhinestones", "type": "bool", "ai_key": "has_rhinestones",
                "tooltip": "Rhinestone or crystal embellishments on the graphic — not just print. Inspect closely, major value premium.",
            },
            {
                "id": "collab", "label": "Collab / Celeb", "type": "bool", "ai_key": "is_collab",
                "tooltip": "Celebrity collab or Christian Audigier era piece. Check all tags for CA branding or collab markings.",
            },
            {
                "id": "designer_series", "label": "Designer Series", "type": "bool", "ai_key": "is_designer_series",
                "tooltip": "Numbered or limited designer series — usually has a special hang tag or label indicating the series.",
            },
        ],
        "vision_prompt": """You are an expert Ed Hardy vintage t-shirt appraiser.
Analyze this image and extract the following signals. Return ONLY valid JSON, no other text:
{
    "era": "y2k or 10s or unknown",
    "size": "S or M or L or XL or 2XL or unknown",
    "has_rhinestones": true or false,
    "is_collab": true or false,
    "is_designer_series": true or false
}
Rules:
- Y2K (2000-2006) = peak Don Ed Hardy / Christian Audigier era
- Rhinestones = crystal/rhinestone embellishments on the graphic (not just ink)
- Collab = celebrity collaboration, CA-era tag, or special partnership branding
- Designer series = numbered/limited collection, special hang tag or label
- Look at neck tags for era clues and series markings""",
    },

    "hysteric": {
        "label": "Hysteric Glamour",
        "short": "HG",
        "csv": "data/raw/hysteric_features.csv",
        "signals": [
            {
                "id": "era", "label": "Era", "type": "select", "ai_key": "era",
                "options": [["unknown","Unknown"],["80s","80s"],["90s","90s"],["y2k","Y2K / 2000s"]],
                "tooltip": "80s–90s HG is most collectible. Check inside collar tag for Japanese text and copyright year.",
            },
            {
                "id": "size", "label": "Size", "type": "select", "ai_key": "size",
                "options": [["unknown","Unknown"],["S","S"],["M","M"],["L","L"],["XL","XL"]],
                "tooltip": "Japanese domestic sizing — runs 1–2 sizes smaller than US. Check tag carefully.",
            },
            {
                "id": "graphic_type", "label": "Graphic Type", "type": "select", "ai_key": "graphic_type",
                "options": [["unknown","Unknown"],["devil","Devil / Evil Babe"],["pin_up","Pin-up"],["logo","Logo Only"],["other","Other"]],
                "tooltip": "Devil and pin-up graphics command the highest premiums. Logo-only pieces are less sought after.",
            },
            {
                "id": "japan_market", "label": "Japan Market Tag", "type": "bool", "ai_key": "is_japan_market",
                "tooltip": "Japanese domestic market tag — Japanese text on inside collar. More collectible than export pieces.",
            },
            {
                "id": "collab", "label": "Collab", "type": "bool", "ai_key": "is_collab",
                "tooltip": "Brand collaboration — e.g. Hysteric Glamour x Playboy, or artist collab. Check all tags.",
            },
        ],
        "vision_prompt": """You are an expert Hysteric Glamour vintage t-shirt appraiser.
Analyze this image and extract the following signals. Return ONLY valid JSON, no other text:
{
    "era": "80s or 90s or y2k or unknown",
    "size": "S or M or L or XL or unknown",
    "graphic_type": "devil or pin_up or logo or other or unknown",
    "is_japan_market": true or false,
    "is_collab": true or false
}
Rules:
- Devil/evil babe and pin-up graphics are the most valuable graphic types
- Japan market tag = Japanese text on inside collar tag
- Collab = collaboration with another brand or artist referenced on tags/graphics
- 80s-90s pieces are most sought after
- Look at all visible tags for era, size, and market clues""",
    },
}
