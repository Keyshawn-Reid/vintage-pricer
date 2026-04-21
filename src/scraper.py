"""
eBay sold listings scraper.
Designed to scale across platforms and brands.

Usage:
    python3 src/scraper.py
    python3 src/scraper.py --query "ed hardy vintage tee" --brand ed_hardy --max 300
    python3 src/scraper.py --query "hysteric glamour tee" --brand hysteric --max 200
"""

import argparse
import os
import re
import time
import random
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

OUTPUT_DIR = "data/raw"
ITEMS_PER_PAGE = 60


def _build_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def _parse_price(raw: str) -> Optional[float]:
    match = re.search(r"[\d,]+\.\d{2}", raw.replace(",", ""))
    return float(match.group().replace(",", "")) if match else None


def _parse_date(raw: str) -> Optional[str]:
    """'Sold  Apr 21, 2026' → '2026-04-21'"""
    raw = re.sub(r"(?i)^sold\s*", "", raw.strip())
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _parse_condition_and_size(subtitle: str) -> tuple:
    """
    eBay subtitle format: 'Pre-Owned ·Size XL' or 'Pre-Owned' or 'Brand New ·Size M'
    Returns (condition, size).
    """
    parts = [p.strip() for p in subtitle.split("·")]
    condition = parts[0] if parts else None

    size = None
    for part in parts[1:]:
        m = re.search(r"(?i)size\s+(\S+)", part)
        if m:
            size = m.group(1).upper()
            break

    return condition, size


def _scrape_page(driver, url: str) -> list[dict]:
    driver.get(url)

    try:
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "li.s-card"))
        )
    except Exception:
        return []

    time.sleep(random.uniform(1.5, 2.5))

    cards = driver.find_elements(By.CSS_SELECTOR, "li.s-card")
    results = []

    for card in cards:
        try:
            # Title
            title_el = card.find_element(By.CSS_SELECTOR, ".s-card__title")
            title = re.sub(r"Opens in a new window or tab$", "", title_el.text).strip()
            title = re.sub(r"(?i)^new\s+listing\s*", "", title).strip()
            if not title or title == "Shop on eBay":
                continue

            # Price — first .s-card__price element is always item price
            try:
                price_raw = card.find_element(By.CSS_SELECTOR, ".s-card__price").text
                price = _parse_price(price_raw)
            except Exception:
                price = None

            # Condition + size — subtitle gives 'Pre-Owned ·Size XL'
            try:
                subtitle = card.find_element(By.CSS_SELECTOR, ".s-card__subtitle").text
                condition, size = _parse_condition_and_size(subtitle)
            except Exception:
                condition, size = None, None

            # Date sold — scan all text spans for 'Sold  Month Day, Year'
            date_sold = None
            try:
                spans = card.find_elements(By.CSS_SELECTOR, ".su-styled-text")
                for span in spans:
                    txt = span.text.strip()
                    if re.match(r"(?i)^sold\s+\w", txt):
                        date_sold = _parse_date(txt)
                        break
            except Exception:
                pass

            results.append({
                "title":      title,
                "sold_price": price,
                "date_sold":  date_sold,
                "condition":  condition,
                "size":       size,
            })

        except Exception:
            continue

    return results


def scrape_ebay(
    query: str,
    brand: str,
    platform: str = "ebay",
    max_listings: int = 500,
    base_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scrape eBay sold listings for a given search query.

    Args:
        query:        eBay search terms, e.g. "harley davidson vintage tee"
        brand:        Short brand identifier used in the output filename, e.g. "harley"
        platform:     Platform name used in the output filename, e.g. "ebay"
        max_listings: Stop after collecting this many listings
        base_url:     Override the full search URL (pagination is appended automatically)

    Output:
        Saves to data/raw/<brand>_<platform>.csv and returns the DataFrame.
    """
    if base_url is None:
        params = {
            "_nkw":         query,
            "_sacat":       "0",
            "LH_Sold":      "1",
            "LH_Complete":  "1",
            "_ipg":         str(ITEMS_PER_PAGE),
        }
        base_url = "https://www.ebay.com/sch/i.html?" + urlencode(params)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{brand}_{platform}.csv")

    print(f"Query:    {query}")
    print(f"Output:   {output_path}")
    print(f"Target:   up to {max_listings} listings")
    print()

    driver = _build_driver()
    all_rows: list[dict] = []
    page = 1

    try:
        while len(all_rows) < max_listings:
            url = f"{base_url}&_pgn={page}"
            print(f"Page {page}  ({len(all_rows)} collected so far)")

            rows = _scrape_page(driver, url)

            if not rows:
                print("  No listings found — stopping.")
                break

            # Attach metadata
            for row in rows:
                row["brand"]    = brand
                row["platform"] = platform

            all_rows.extend(rows)
            print(f"  +{len(rows)} listings  →  {len(all_rows)} total")

            if len(rows) < ITEMS_PER_PAGE * 0.5:
                print("  Partial page — reached end of results.")
                break

            page += 1
            time.sleep(random.uniform(4.0, 7.0))

    finally:
        driver.quit()

    all_rows = all_rows[:max_listings]
    df = pd.DataFrame(all_rows, columns=[
        "title", "sold_price", "date_sold", "condition", "size", "brand", "platform"
    ])

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} listings → {output_path}")
    print(df[["title", "sold_price", "condition", "size"]].head(5).to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape eBay sold listings")
    parser.add_argument("--query",    default="harley davidson vintage tee",
                        help="eBay search query")
    parser.add_argument("--brand",    default="harley",
                        help="Brand identifier for output filename")
    parser.add_argument("--platform", default="ebay",
                        help="Platform name for output filename")
    parser.add_argument("--max",      default=500, type=int,
                        help="Maximum listings to collect")
    args = parser.parse_args()

    scrape_ebay(
        query=args.query,
        brand=args.brand,
        platform=args.platform,
        max_listings=args.max,
    )
