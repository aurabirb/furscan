# make parallel requests to https://solar.furtrack.com/get/p/{0..513846}
import concurrent.futures
import json
import os
from typing import Iterable
from itertools import islice
import random

import requests

MAX_CHAR_ENTRIES = 10


def get_url(url):
    return requests.get(
        url,
        cookies={
            "cf_chl_3": "61e6c8731e33040",
            "cf_clearance": "FkiDuQ_1uR97vNeNGfgnkGCN1LdpbkFBl.rGD_YlHwM-1712176234-1.0.1.1-7EVLvjYJz1iZcdDXdvSG2Vi3ry2FVl3S2yvZDL7dxEVAlJk_P.Lte4mGouWyIl4tFasG9N5PpH4acZapmqAQsw",
        },
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        },
    )


def get_json(url):
    r = get_url(url)
    if r.status_code != 200:
        print(f"Failed to get {url}")
        print(r.text)
        raise Exception(f"status_code: {r.status_code}, {r.url}")
    j = r.json()
    return j


def make_requests(urls: list[str], max_requests: int = 20, verbose: bool = True):
    # break urls into chunks of max_requests
    url_chunks = [urls[i : i + max_requests] for i in range(0, len(urls), max_requests)]
    total = 0
    for chunk in url_chunks:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            res = executor.map(get_url, chunk)
            total += len(chunk)
            if verbose:
                print(f"Requested {total}/{len(urls)} urls")
            for r in res:
                yield r


def get_char_tags(tags: list[dict]):
    return [
        t["tagName"].removeprefix("1:")
        for t in tags
        if t.get("tagName", "").startswith("1:")
    ]


def check_single_character_tags(tags: list[dict]):
    tagNames = [t.get("tagName", "") for t in tags]
    char_tags = get_char_tags(tags)
    # print(char_tags)
    if len(char_tags) != 1:
        return False
    bad_tags = [t for t in tagNames if t in {"tagging_incomplete", "missing_character"}]
    if len(bad_tags):
        return False
    return True


# def get_full_img_url(j: dict):
#     postId = j.get("post", {}).get("postId", "")
#     postStub = j.get("post", {}).get("metaFingerprint", "")
#     userId = j.get("post", {}).get("submitUserId", "")
#     if not postId or not postStub or not userId:
#         return ""
#     return f"https://orca2.furtrack.com/gallery/{userId}/{postId}-{postStub}.jpg"


def get_img_url(j: dict):
    postId = j.get("post", {}).get("postId", "")
    if not postId:
        return ""
    return f"https://orca2.furtrack.com/thumb/{postId}.jpg"


def check_response(raw: dict, post_id: int):
    tests = [
        lambda raw: (raw.get("success") is True),
        lambda raw: (check_single_character_tags(raw.get("tags", []))),
        lambda raw: (get_img_url(raw)),
        lambda raw: (raw.get("post", {}).get("videoId", "") == ""),
    ]
    rslt = all((test(raw) for test in tests))
    if not tests[0](raw):
        print(post_id, raw)
    return rslt


def get_character(res: dict):
    tags = res.get("tags", [])
    char_tags = get_char_tags(tags)
    if len(char_tags) != 1:
        char_tags = [""]
    return char_tags[0]


def download_images(urls_names: list[tuple[str, str]], folder: str):
    os.makedirs(folder, exist_ok=True)
    # skip if file already exists
    dir_contents = os.listdir(folder)
    len_input = len(urls_names)
    urls_names = [(u, n) for u, n in urls_names if f"{n}.jpg" not in dir_contents]
    if len(urls_names) < len_input:
        print(f"Skipped downloading {len_input - len(urls_names)} images")

    img_urls = [u for u, n in urls_names]
    names = [n for u, n in urls_names]
    if len(img_urls):
        print(f"Downloading {len(img_urls)} images")
    rs = make_requests(img_urls, verbose=False)
    for name, r in zip(names, rs):
        if r.status_code == 200:
            with open(f"{folder}/{name}.jpg", "wb") as f:
                f.write(r.content)
        else:
            print(f"Failed to download {name}.jpg")


def get_furtrack_posts(post_ids: list[int]):
    urls = [f"https://solar.furtrack.com/get/p/{i}" for i in post_ids]
    jsons = [r.json() for r in make_requests(urls)]
    mapped = [
        dict(
            char=get_character(f),
            url=get_img_url(f),
            post_id=str(post_id),
            raw=f,
        )
        if check_response(f, post_id)
        else dict(
            char="",
            url="",
            post_id=str(post_id),
            raw=f,
        )
        for post_id, f in zip(post_ids, jsons)
    ]
    return mapped


# store embeddings in a database
import sqlite3

import numpy as np


def create_table():
    if os.path.exists("furtrack.db"):
        print("furtrack.db already exists")
        return
    conn = sqlite3.connect("furtrack.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE furtrack
                    (post_id text PRIMARY KEY, char text, url text, raw text, embedding_id integer default NULL, date_modified timestamp DEFAULT CURRENT_TIMESTAMP)""")
    # unique constraint
    # c.execute("CREATE UNIQUE INDEX post_id_index ON furtrack (post_id)")
    c.execute("CREATE UNIQUE INDEX u_embedding_id_index ON furtrack(embedding_id)")
    conn.commit()
    conn.close()


def store_furtrack_data(dict_list: list[dict]):
    conn = sqlite3.connect("furtrack.db")
    c = conn.cursor()
    try:
        if len(dict_list) == 0:
            print("No data to store")
            return
        for d in dict_list:
            c.execute(
                "INSERT OR REPLACE INTO furtrack VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (
                    d["post_id"],
                    d["char"],
                    d["url"],
                    json.dumps(d["raw"]),
                    d.get("embedding_id", None),
                    # CURRENT_TIMESTAMP,
                ),
            )
        conn.commit()
        c.execute(
            "SELECT COUNT(*) FROM furtrack UNION ALL SELECT COUNT(*) FROM furtrack WHERE url!=''"
        )
        print(f"Rows in database: {c.fetchone()[0]}, Rows with urls: {c.fetchone()[0]}")
    except sqlite3.IntegrityError:
        print(f"post_id {d['post_id']} already exists")
    finally:
        # print("Closing connection")
        conn.close()


def row2dict(row):
    if row is None:
        return None
    return {
        "post_id": row[0],
        "char": row[1],
        "url": row[2],
        "raw": json.loads(row[3]),
        "embedding_id": row[4],
        "date_modified": row[5],
    }


def recall_furtrack_data_by_id(post_id: str):
    conn = sqlite3.connect("furtrack.db")
    c = conn.cursor()
    c.execute("SELECT * FROM furtrack WHERE post_id=?", (post_id,))
    res = c.fetchone()
    conn.close()
    return row2dict(res)


def count_character_entries_in_db(char: str):
    conn = sqlite3.connect("furtrack.db")
    c = conn.cursor()
    c.execute("SELECT count(*) FROM furtrack WHERE char=?", (char,))
    res = c.fetchone()
    conn.close()
    return res[0]


def recall_furtrack_data_by_embedding_id(embedding_id: int):
    conn = sqlite3.connect("furtrack.db")
    c = conn.cursor()
    c.execute("SELECT * FROM furtrack WHERE embedding_id=?", (str(embedding_id),))
    res = c.fetchone()
    conn.close()
    return row2dict(res)


# def sqlite_add_column():
#     conn = sqlite3.connect("furtrack.db")
#     c = conn.cursor()
#     c.execute("ALTER TABLE furtrack ADD COLUMN embedding_id integer default NULL")
#     c.execute("CREATE UNIQUE INDEX u_embedding_id_index ON furtrack(embedding_id)")
#     conn.commit()
#     conn.close()


def download_posts(post_ids_: Iterable[int], image_folder: str, batch_size: int):
    post_ids = (p for p in post_ids_ if recall_furtrack_data_by_id(str(p)) is None)
    # print(f"Getting {len(ids)}/{len(post_ids)} new posts")
    # process in batches
    while True:
        ids = list(islice(post_ids, batch_size))
        if not ids:
            break
        metas = get_furtrack_posts(ids)
        store_furtrack_data(metas)
        bad_metas = [m["post_id"] for m in metas if not m["url"]]
        if len(bad_metas):
            print(f"{bad_metas} bad posts")
        valid = [m for m in metas if m["url"] and m["char"]]
        print(f"{len(valid)} got metadata")
        skip = [
            m
            for m in valid
            if count_character_entries_in_db(m["char"]) >= MAX_CHAR_ENTRIES
        ]
        if len(skip):
            print(
                f"Skipping download of {len(skip)} images with seen character entries"
            )
            valid = [m for m in valid if m not in skip]
        download_images([(m["url"], m["post_id"]) for m in valid], image_folder)


def batch_download_images(post_ids: Iterable[int], image_folder: str, batch_size: int):
    metas = (recall_furtrack_data_by_id(str(p)) for p in post_ids)
    urls_ids = ((m["url"], m["post_id"]) for m in metas if m and m["url"])
    while True:
        pairs = list(islice(urls_ids, batch_size))
        if not pairs:
            break
        download_images(pairs, image_folder)


def get_recent_posts(page=0):
    print(f"Getting recents page {page}")
    url = "https://solar.furtrack.com/get/all"
    if page:
        url += f"/{page}"
    j = get_json(url)
    return [p["postId"] for p in j["posts"]]


def get_character_post_ids(char: str) -> list[int]:
    url = f"https://solar.furtrack.com/get/index/1:{char}"
    j = get_json(url)
    post_ids = [int(p["postId"]) for p in j["posts"]]
    random.shuffle(post_ids)
    return post_ids


def download_character_list():
    # https://solar.furtrack.com/get/tags/all
    url = "https://solar.furtrack.com/get/tags/all"
    j = get_json(url)
    tags = j["tags"]
    char_tags = [
        t["tagName"].removeprefix("1:")
        for t in tags
        if t.get("tagName", "").startswith("1:")
    ]
    return char_tags


def ingest_character_posts(
    char: str, image_folder: str, indexing: bool, batch_size: int
):
    post_ids = get_character_post_ids(char)
    download_posts(post_ids, image_folder, batch_size)
    if indexing:
        metas = [recall_furtrack_data_by_id(str(p)) for p in post_ids]
        metas = [m for m in metas if m and m["url"] and m["char"]]
        ingest_posts([int(m["post_id"]) for m in metas], image_folder, "embeddings")


def download_all_characters(image_folder: str, batch_size: int, indexing: bool = False):
    char_tags = download_character_list()
    print(f"Downloading {len(char_tags)} characters...")

    random.seed(42)
    random.shuffle(char_tags)
    for i, char in enumerate(char_tags):
        print(f"Downloading character ({i}/{len(char_tags)}): {char}")
        ingest_character_posts(char, image_folder, indexing, batch_size)


def ingest_posts(post_ids: list[int], image_folder: str, embed_folder: str):
    """Download and index posts using the SAM3 system."""
    from sam3_pursuit import SAM3FursuitIdentifier

    metas = get_furtrack_posts(post_ids)
    store_furtrack_data(metas)
    metas = [m for m in metas if m["url"] and m["char"]]
    download_images([(m["url"], m["post_id"]) for m in metas], image_folder)

    # Index using new SAM3 system
    identifier = SAM3FursuitIdentifier()
    for meta in metas:
        img_path = f"{image_folder}/{meta['post_id']}.jpg"
        if os.path.exists(img_path):
            identifier.add_images(meta["char"], [img_path])


if __name__ == "__main__":
    from time import sleep
    # import random

    # random.seed(42)
    # num_samples = 40000
    # print(f"Sampling {num_samples} posts")
    # post_ids = (i for i in random.sample(range(549557), num_samples))
    from itertools import chain

    num_pages = 10000
    post_ids = chain.from_iterable(get_recent_posts(page) for page in range(num_pages))

    create_table()
    # print(recall_furtrack_data_by_id("537727"))
    # print(recall_furtrack_data_by_id("2"))
    while True:
        download_all_characters("furtrack_images", batch_size=40)
        sleep(1)
    # download_posts(post_ids, "furtrack_images", batch_size=40)
    # batch_download_images(post_ids, "furtrack_images", batch_size=40)
