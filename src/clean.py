import pandas as pd

INPUT  = "data/raw/harley_ebay.csv"
OUTPUT = "data/processed/harley_clean.csv"

df = pd.read_csv(INPUT)
original_count = len(df)
print(f"Loaded {original_count} rows\n")

removals = {}

# ── Multi-item lots ───────────────────────────────────────────────────────────
LOT_PATTERN = r'\b(?:lot|bundle|set|pack|wholesale)\b'
mask = df["title"].str.lower().str.contains(LOT_PATTERN, regex=True, na=False)
removals["lot/bundle"] = df[mask][["title", "sold_price"]]
df = df[~mask].copy()

# ── Reprints (not authentic vintage — different pricing regime) ───────────────
REPRINT_PATTERN = r'\breprint\b'
mask = df["title"].str.lower().str.contains(REPRINT_PATTERN, regex=True, na=False)
removals["reprint"] = df[mask][["title", "sold_price"]]
df = df[~mask].copy()

# ── Wrong brand / off-topic listings ─────────────────────────────────────────
# These slipped into the dataset — NASCAR, West Coast Choppers, etc.
WRONG_BRAND = [
    "west coast choppers",
    "jeff gordon",
    "dale earnhardt",
    "nascar",
    "smith and wesson",
]
mask = df["title"].str.lower().str.contains('|'.join(WRONG_BRAND), na=False)
removals["wrong brand"] = df[mask][["title", "sold_price"]]
df = df[~mask].copy()

# ── Damaged items ─────────────────────────────────────────────────────────────
# "faded" alone is a style descriptor; "faded distressed" together signals damage
DAMAGE_PATTERNS = [
    r'\bholes?\b',
    r'\bstains?\b',
    r'\brips?\b',
    r'\btorn\b',
    r'\btear\b',
    r'\bdamages?\b',
    r'\brepairs?\b',
    r'faded.{0,10}distressed|distressed.{0,10}faded',  # combo = damage
]
mask = df["title"].str.lower().str.contains(
    '|'.join(DAMAGE_PATTERNS), regex=True, na=False
)
removals["damaged"] = df[mask][["title", "sold_price"]]
df = df[~mask].copy()

# ── Damaged condition field ───────────────────────────────────────────────────
# eBay condition field occasionally contains damage descriptions or bad data
CONDITION_DAMAGE = [r'\bstain', r'\brip', r'\bworn', r'\bflaw']
cond_mask = df["condition"].str.lower().str.contains(
    '|'.join(CONDITION_DAMAGE), regex=True, na=False
)
# Also drop rows where condition field looks like a title (scraped incorrectly)
bad_cond_mask = df["condition"].str.len() > 60
removals["bad condition field"] = df[cond_mask | bad_cond_mask][["title", "sold_price", "condition"]]
df = df[~(cond_mask | bad_cond_mask)].copy()

# ── Price outliers ────────────────────────────────────────────────────────────
mask = (df["sold_price"] < 10) | (df["sold_price"] > 300)
removals["price outlier"] = df[mask][["title", "sold_price"]]
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
