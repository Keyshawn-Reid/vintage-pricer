BRANDS = {
    "harley": {
        "label": "Harley Davidson",
        "short": "HD",
        "csv": "data/raw/harley_features.csv",
        "multi_image": True,
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
- Always return all 6 fields. Use "unknown" for era/size you cannot determine; use false for features not clearly visible.
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
        "multi_image": True,
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
        "multi_image": True,
        "signals": [
            {
                "id": "era", "label": "Era", "type": "select", "ai_key": "era",
                "options": [["unknown","Unknown"],["80s","80s"],["90s","90s"],["y2k","Y2K / 2000s"],["current","Current Production"]],
                "tooltip": "80s–90s HG is peak collectibility. Current production (2010s–present) carries no vintage premium. Tag generation is the most reliable era signal.",
            },
            {
                "id": "size", "label": "Size", "type": "select", "ai_key": "size",
                "options": [["unknown","Unknown"],["S","S"],["M","M"],["L","L"],["XL","XL"],["2XL","2XL"]],
                "tooltip": "Japanese domestic sizing runs 1–2 sizes smaller than US. Larger sizes (XL/2XL) carry a premium for Hysteric.",
            },
            {
                "id": "graphic_type", "label": "Graphic Type", "type": "select", "ai_key": "graphic_type",
                "options": [
                    ["unknown","Unknown"],
                    ["snake","Snake Print"],
                    ["skull","Skull"],
                    ["devil_babe","Devil Babe / Evil Girl"],
                    ["pin_up","Pin-up"],
                    ["marilyn","Marilyn / Celebrity"],
                    ["cross","Cross / Religious"],
                    ["logo_only","Logo Only"],
                    ["other","Other Graphic"],
                ],
                "tooltip": "Graphic type is the single largest price driver. Snake, skull, and devil babe are highest value. Logo-only is lowest. When in doubt, choose Other.",
            },
            {
                "id": "collab_tier", "label": "Collab", "type": "select", "ai_key": "collab_tier",
                "options": [["none","No Collab"],["supreme","Supreme"],["guess","Guess"],["other","Other Collab"]],
                "tooltip": "Supreme x Hysteric (2017 and 2024) carries the highest collab premium — 3× or more. Guess collab adds a smaller premium. Check all tags and graphic text.",
            },
            {
                "id": "is_japan_domestic", "label": "Japan Domestic Tag", "type": "bool", "ai_key": "is_japan_domestic",
                "tooltip": "Full Japanese katakana/kanji text on the collar tag — not just 'Made in Japan.' Domestic market pieces are more collectible than export.",
            },
            {
                "id": "is_reprint", "label": "Reprint / Reproduction", "type": "bool", "ai_key": "is_reprint",
                "tooltip": "Modern reproduction of a vintage graphic. Key signals: current-gen tag with retro graphic, unnaturally bright print, modern blank construction. Reprints trade at 50% of original.",
            },
            {
                "id": "has_back_graphic", "label": "Back Graphic", "type": "bool", "ai_key": "has_back_graphic",
                "tooltip": "A printed graphic on the back of the shirt — not blank. Back prints add meaningful value, especially large or detailed ones.",
            },
            {
                "id": "condition", "label": "Condition", "type": "select", "ai_key": "condition",
                "options": [["3","Pre-Owned / Good"],["5","NWT / Deadstock"],["4","Excellent / Near Mint"],["2","Light Wear / Fading"],["1","Heavy Wear / Flaws"]],
                "tooltip": "5 = New with tags or deadstock. 4 = Excellent, no visible wear. 3 = Normal pre-owned. 2 = Light fade or small flaws. 1 = Heavy wear, staining, or damage.",
            },
        ],
        "vision_prompt": """You are an expert Hysteric Glamour vintage appraiser. Extract pricing signals from the provided image(s) and return ONLY valid JSON:
{
    "era": "80s or 90s or y2k or current or unknown",
    "size": "S or M or L or XL or 2XL or unknown",
    "graphic_type": "snake or skull or devil_babe or pin_up or marilyn or cross or logo_only or other or unknown",
    "collab_tier": "none or supreme or guess or other",
    "is_japan_domestic": true or false,
    "is_reprint": true or false,
    "has_back_graphic": true or false,
    "has_single_stitch": true or false,
    "condition": 1 or 2 or 3 or 4 or 5
}
Rules:
- Era: 80s = pre-1990, 90s = 1990–1999 (most collectible), y2k = 2000–2009, current = 2010–present
- is_japan_domestic = Japanese katakana or kanji text on the collar tag (more than just "Made in Japan")
- is_reprint = modern reproduction of vintage graphic — mismatched tag era vs graphic style, unnaturally bright print on modern blank, barcode/QR on tag with retro graphic are strong signals
- supreme collab = Supreme box logo present or "Supreme" text on any tag
- guess collab = Guess? branding alongside Hysteric Glamour text
- Condition: 5=NWT/deadstock, 4=excellent/near-mint, 3=pre-owned good, 2=light wear or fading, 1=heavy wear or visible flaws
- has_back_graphic = back image is provided and shows a printed graphic (not blank)
- has_single_stitch = single row of stitching visible on sleeve hem (vintage construction)
- Tag generation clues: early (80s–90s) tags are woven, Japanese text, no barcode. Modern tags have QR codes, barcodes, clean minimalist style.""",
    },
}
