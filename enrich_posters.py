import pandas as pd
import requests
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Iterable, List

API_URL = "https://graphql.anilist.co"
QUERY = """
query ($ids: [Int]) {
  Page(perPage: 50) {
    media(idMal_in: $ids, type: ANIME) {
      idMal
      coverImage { extraLarge }
    }
  }
}
"""


def chunks(iterable: Iterable[int], size: int) -> Iterable[List[int]]:
    """Yield successive lists of length size from iterable."""
    it = list(iterable)
    for i in range(0, len(it), size):
        yield it[i : i + size]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1))
def fetch_covers(ids: List[int]) -> dict:
    """Fetch poster URLs for given MyAnimeList IDs from AniList."""
    resp = requests.post(
        API_URL,
        json={"query": QUERY, "variables": {"ids": ids}},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data


def main() -> None:
    df = pd.read_csv("anime.csv")

    mask = df["cover_url"].isna() | df["cover_url"].astype(str).eq("")
    missing_ids = df.loc[mask, "mal_id"].astype(int).tolist()
    total_rows = len(df)
    missing_before = mask.sum()

    for batch in chunks(missing_ids, 50):
        try:
            data = fetch_covers(batch)
        except Exception:
            time.sleep(0.75)
            continue

        for media in data.get("data", {}).get("Page", {}).get("media", []):
            mal_id = media.get("idMal")
            url = media.get("coverImage", {}).get("extraLarge") or ""
            df.loc[df["mal_id"] == mal_id, "cover_url"] = url

        time.sleep(0.75)

    mask_after = df["cover_url"].isna() | df["cover_url"].astype(str).eq("")
    updated = missing_before - mask_after.sum()

    df.to_csv("anime_with_covers.csv", index=False)
    print(f"Updated {updated} / {total_rows} rows \u2013 wrote anime_with_covers.csv")


if __name__ == "__main__":
    main()
