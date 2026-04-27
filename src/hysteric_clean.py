import pandas as pd

INPUT  = "data/raw/hysteric_ebay.csv"
OUTPUT = "data/processed/hysteric_clean.csv"

df = pd.read_csv(INPUT)

# Drop duplicate header rows that got pasted in mid-file
df = df[df["title"] != "title"].copy()
df["sold_price"] = pd.to_numeric(df["sold_price"], errors="coerce")
df = df.dropna(subset=["sold_price"])

original_count = len(df)
print(f"Loaded {original_count} rows\n")

removals = {}

# ── Lots / bundles ────────────────────────────────────────────────────────────
LOT_PATTERN = r'\b(?:lot|bundle|set|pack|wholesale)\b'
mask = df["title"].str.lower().str.contains(LOT_PATTERN, regex=True, na=False)
removals["lot/bundle"] = df[mask][["title", "sold_price"]]
df = df[~mask].copy()

# ── Wrong brand — non-HG items that slipped in via Grailed searches ───────────
WRONG_BRAND = [
    "hydrogen",
    "milkboy",
    "murder license",
]
mask = df["title"].str.lower().str.contains('|'.join(WRONG_BRAND), na=False)
removals["wrong brand"] = df[mask][["title", "sold_price"]]
df = df[~mask].copy()

# ── Non-apparel / accessories that distort the apparel model ─────────────────
NON_APPAREL = [
    r'\bjibbitz\b',
    r'\bclog\b',
    r'\bkey holder\b',
]
mask = df["title"].str.lower().str.contains('|'.join(NON_APPAREL), regex=True, na=False)
removals["non-apparel"] = df[mask][["title", "sold_price"]]
df = df[~mask].copy()

# ── Damaged items ─────────────────────────────────────────────────────────────
DAMAGE_PATTERNS = [
    r'\bholes?\b',
    r'\bstains?\b',
    r'\brips?\b',
    r'\btorn\b',
    r'\btear\b',
    r'\bdamages?\b',
]
mask = df["title"].str.lower().str.contains(
    '|'.join(DAMAGE_PATTERNS), regex=True, na=False
)
removals["damaged"] = df[mask][["title", "sold_price"]]
df = df[~mask].copy()

# ── Price outliers ────────────────────────────────────────────────────────────
# HG has a wide legitimate range — only drop implausible lows
mask = df["sold_price"] < 20
removals["price outlier (low)"] = df[mask][["title", "sold_price"]]
df = df[~mask].copy()

# ── Audit report ─────────────────────────────────────────────────────────────
total_removed = 0
for reason, removed_df in removals.items():
    if len(removed_df) > 0:
        print(f"── {reason.upper()} ({len(removed_df)} removed) ──")
        for _, row in removed_df.iterrows():
            print(f"  ${row['sold_price']:<7.2f}  {row['title']}")
        print()
        total_removed += len(removed_df)

print(f"{'─' * 60}")
print(f"Removed:   {total_removed} rows")
print(f"Remaining: {len(df)} rows")
print(f"Price range: ${df['sold_price'].min():.2f} – ${df['sold_price'].max():.2f}")
print(f"Mean: ${df['sold_price'].mean():.2f}  |  Median: ${df['sold_price'].median():.2f}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
