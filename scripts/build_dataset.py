"""Build a weakly-labelled training set for ScamShield v2 from public data.

Sources (downloaded once into data/raw/, with mirrors as fallbacks):
  * SMS Spam Collection (UCI)                  -> text records, ham = Benign, spam = scam
  * Phishing.Database ACTIVE links / OpenPhish -> scam URL records
  * Tranco top sites (fallback: top-10000 list) -> benign URL records

WEAK LABELS. The source datasets only say scam / not-scam. The fraud
*category* and the suspicious *spans* are added by the keyword rules below, so
they are noisy heuristics, not human annotation. Real taxonomy / span quality
needs hand-labelled data. Every record carries "label_source": "weak-rules", and
data/dataset_card.json records how the set was built.

Usage (from the scamshield-v2 folder):
    python scripts/build_dataset.py                    # default sizes
    python scripts/build_dataset.py --max-scam-urls 6000 --max-benign-urls 6000
    python scripts/build_dataset.py --sms-file my_sms.csv --phish-file urls.txt   # use local files instead

Then train:
    python -m src.training.train_distill --train data/train.jsonl --val data/val.jsonl --out artifacts
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import TAXONOMY  # noqa: E402

SMS_SOURCES = [
    ("uci-zip", "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"),
    ("kaggle-csv", "https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"),
]
PHISH_SOURCES = [
    ("openphish", "https://openphish.com/feed.txt"),
    ("phishing-database", "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"),
]
BENIGN_SOURCES = [
    ("tranco-zip", "https://tranco-list.eu/top-1m.csv.zip"),
    ("top-10000", "https://raw.githubusercontent.com/zer0h/top-1000000-domains/master/top-10000-domains"),
]

# --------------------------------------------------------------------------- #
# Weak-labelling rules. Order = priority when several categories match.
# Each pattern match also becomes a suspicious text span.
# --------------------------------------------------------------------------- #
RULES: list[tuple[str, list[str]]] = [
    ("Crypto Fraud", [
        r"\b(bitcoin|btc|crypto(currency)?|ethereum|eth|usdt|wallet|airdrop|token|nft|blockchain)\b",
        r"\b(double your|guaranteed (returns?|profit)|\d+x returns?|trading bot|investment plan)\b",
    ]),
    ("Tech Support Impersonation", [
        r"\b(tech(nical)? support|microsoft|windows|apple support|virus|malware|infected|hacked)\b",
        r"\b(remote access|anydesk|teamviewer|licen[cs]e (has )?expired|your (computer|device|pc))\b",
    ]),
    ("Credential Harvesting", [
        r"\b(password|passcode|pin|otp|one[- ]time (code|password)|cvv|login details|credentials|username)\b",
        r"\b(verify your (account|identity|details)|confirm your (account|details|identity)|kyc|log ?in)\b",
    ]),
    ("Financial Coercion", [
        r"\b(arrest(ed)?|warrant|police|court|legal action|lawsuit|penalty|fine|overdue|debt|tax (refund|due))\b",
        r"\b(gift ?cards?|wire transfer|pay (now|immediately|today)|final (notice|warning)|account (will be )?(blocked|suspended|frozen))\b",
        r"\b(disconnect(ed|ion)?|shut ?off|non-?payment|past due|social security number (has been |is )?suspended)\b",
    ]),
    ("Phishing", [
        r"\b(claim|click|tap|visit|call|text|reply|send)\b[^.!?\n]{0,30}\b(now|today|here|link|to claim|to receive|asap)\b",
        r"\b(won|winner|win|prize|award(ed)?|cash|reward|bonus|free|selected|congratulations|urgent|guaranteed)\b(?!['’])",
        r"\b0[789]\d{8,10}\b",  # UK premium-rate style numbers common in SMS spam
        r"\b(www\.|https?://)\S+",
    ]),
]
COMPILED = [(cat, [re.compile(p, re.IGNORECASE) for p in pats]) for cat, pats in RULES]
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)

SUSPICIOUS_URL_TOKENS = re.compile(
    r"(login|log-in|signin|sign-in|verify|verification|secure|account|update|confirm|banking|wallet|support|"
    r"password|webscr|billing|unlock|suspend|recover|auth|validate|paypal|apple|microsoft|office365|outlook|"
    r"netflix|amazon|chase|wellsfargo|hsbc|hdfc|sbi|icici|metamask|coinbase|binance)", re.IGNORECASE)

BENIGN_PATHS = ["", "/", "/login", "/account", "/account/settings", "/signin", "/help", "/support", "/about",
                "/news", "/search?q=weather", "/products/{n}", "/blog/{n}", "/docs/getting-started",
                "/orders/{n}", "/watch?v=dQw4w9WgXcQ", "/en-us/", "/contact", "/careers", "/privacy"]


def fetch(url: str, dest: Path, timeout: int = 120) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "scamshield-dataset-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def first_available(sources: list[tuple[str, str]], raw: Path) -> tuple[str, Path]:
    errors = []
    for name, url in sources:
        try:
            suffix = ".zip" if url.endswith(".zip") else ".csv" if url.endswith(".csv") else ".txt"
            return name, fetch(url, raw / f"{name}{suffix}")
        except Exception as exc:  # network errors, 404s, blocked hosts
            errors.append(f"{name}: {exc}")
            print(f"  ! {name} unavailable ({exc}); trying next source")
    raise RuntimeError("no source reachable:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_sms(name: str, path: Path) -> list[tuple[str, str]]:
    """Returns [(label 'ham'|'spam', text)]."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            member = next(n for n in z.namelist() if n.endswith("SMSSpamCollection"))
            lines = z.read(member).decode("utf-8", errors="replace").splitlines()
        rows = [ln.split("\t", 1) for ln in lines if "\t" in ln]
    else:
        raw = path.read_bytes().decode("latin-1")
        reader = csv.reader(io.StringIO(raw))
        header = next(reader)
        if [h.strip().lower() for h in header[:2]] not in (["v1", "v2"], ["label", "text"]):
            rows = [header[:2]]
        else:
            rows = []
        rows += [r[:2] for r in reader if len(r) >= 2]
    out = []
    for label, text in rows:
        label = label.strip().lower()
        if label in ("ham", "spam") and text.strip():
            out.append((label, " ".join(text.split())))
    return out


def load_phish_urls(path: Path) -> list[str]:
    urls = []
    for ln in path.read_text("utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if ln.lower().startswith(("http://", "https://")) and len(ln) <= 256 and " " not in ln:
            urls.append(ln)
    return urls


def load_benign_domains(path: Path, limit: int = 20000) -> list[str]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            text = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
        domains = [ln.split(",", 1)[1].strip() for ln in text.splitlines()[:limit] if "," in ln]
    else:
        domains = [ln.strip() for ln in path.read_text("utf-8", errors="replace").splitlines()[:limit] if ln.strip()]
    return [d for d in domains if "." in d and " " not in d]


# --------------------------------------------------------------------------- #
# Weak labelling
# --------------------------------------------------------------------------- #
def merge_spans(spans: list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def label_text(text: str) -> tuple[str, list[list[int]]]:
    """Category = highest-priority rule group with the most matches; spans = all rule matches."""
    best, best_hits, spans = "Phishing", 0, []
    for cat, patterns in COMPILED:
        hits = 0
        for pat in patterns:
            for m in pat.finditer(text):
                hits += 1
                spans.append((m.start(), m.end()))
        if hits > best_hits:
            best, best_hits = cat, hits
    return best, merge_spans(spans)


def label_url(url: str) -> tuple[str, list[list[int]]]:
    low = url.lower()
    if re.search(r"(wallet|metamask|coinbase|binance|crypto|airdrop|token|nft)", low):
        cat = "Crypto Fraud"
    elif re.search(r"(microsoft|windows|support|helpdesk|virus|office365)", low):
        cat = "Tech Support Impersonation"
    elif re.search(r"(login|signin|sign-in|verify|password|auth|account|webscr|credential|owa|outlook)", low):
        cat = "Credential Harvesting"
    else:
        cat = "Phishing"
    spans = [(m.start(), m.end()) for m in SUSPICIOUS_URL_TOKENS.finditer(url)]
    if not spans:  # fall back to highlighting the host
        host = urlsplit(url).netloc
        start = url.find(host)
        if host and start >= 0:
            spans = [(start, start + len(host))]
    return cat, merge_spans(spans)


def benign_url(domain: str, rng: random.Random) -> str:
    path = rng.choice(BENIGN_PATHS).replace("{n}", str(rng.randint(100, 99999)))
    sub = rng.choice(["www.", "www.", "", "m.", "help.", "accounts."]) if not domain.startswith("www.") else ""
    return f"https://{sub}{domain}{path}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--sms-file", help="local SMS file (UCI tab format or CSV with v1,v2 / label,text)")
    ap.add_argument("--phish-file", help="local text file with one phishing URL per line")
    ap.add_argument("--benign-file", help="local file with one domain per line (or Tranco rank,domain CSV)")
    ap.add_argument("--max-scam-urls", type=int, default=4000)
    ap.add_argument("--max-benign-urls", type=int, default=4000)
    ap.add_argument("--augment-templates", type=int, default=0,
                    help="add N toy template records (src/data/synthetic.py) to TRAIN only, to give rare classes "
                         "such as Financial Coercion some coverage; marked label_source=synthetic-template")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    raw = out / "raw"
    sources: dict[str, str] = {}

    print("1/4 SMS messages")
    if args.sms_file:
        sms_name, sms_path = "local", Path(args.sms_file)
    else:
        sms_name, sms_path = first_available(SMS_SOURCES, raw)
    sms = load_sms(sms_name, sms_path)
    sources["sms"] = sms_name
    print(f"    {len(sms)} messages ({sum(l == 'spam' for l, _ in sms)} spam)")

    print("2/4 phishing URLs")
    if args.phish_file:
        ph_name, ph_path = "local", Path(args.phish_file)
    else:
        ph_name, ph_path = first_available(PHISH_SOURCES, raw)
    phish = load_phish_urls(ph_path)
    rng.shuffle(phish)
    phish = list(dict.fromkeys(phish))[: args.max_scam_urls]
    sources["phishing_urls"] = ph_name
    print(f"    {len(phish)} URLs sampled")

    print("3/4 benign domains")
    if args.benign_file:
        bn_name, bn_path = "local", Path(args.benign_file)
    else:
        bn_name, bn_path = first_available(BENIGN_SOURCES, raw)
    domains = load_benign_domains(bn_path)
    sources["benign_domains"] = bn_name
    benign = [benign_url(rng.choice(domains), rng) for _ in range(args.max_benign_urls)]
    print(f"    {len(domains)} domains -> {len(benign)} URLs")

    print("4/4 weak labelling")
    records, seen = [], set()
    for label, text in sms:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        m = URL_RE.search(text)
        url = m.group(0).rstrip(".,;!") if m else None
        if label == "spam":
            cat, spans = label_text(text)
            url_spans = label_url(url)[1] if url else []
            records.append({"text": text, "url": url, "is_scam": 1, "category": cat,
                            "text_spans": spans, "url_spans": url_spans, "label_source": "weak-rules"})
        else:
            records.append({"text": text, "url": url, "is_scam": 0, "category": "Benign",
                            "text_spans": [], "url_spans": [], "label_source": "dataset"})
    for url in phish:
        cat, spans = label_url(url)
        records.append({"text": None, "url": url, "is_scam": 1, "category": cat,
                        "text_spans": [], "url_spans": spans, "label_source": "weak-rules"})
    for url in benign:
        records.append({"text": None, "url": url, "is_scam": 0, "category": "Benign",
                        "text_spans": [], "url_spans": [], "label_source": "dataset"})
    assert all(r["category"] in TAXONOMY for r in records)

    # stratified split by category
    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    train, val = [], []
    for rows in by_cat.values():
        rng.shuffle(rows)
        k = max(1, int(round(len(rows) * args.val_ratio)))
        val += rows[:k]
        train += rows[k:]
    if args.augment_templates:
        from src.data.synthetic import make_synthetic_records

        for rec in make_synthetic_records(args.augment_templates, scam_ratio=0.8, acoustic_prob=0.0, seed=args.seed):
            d = dict(rec.__dict__)
            d.pop("audio_path", None)
            d.pop("acoustic", None)
            d["text_spans"] = [list(s) for s in d["text_spans"]]
            d["url_spans"] = [list(s) for s in d["url_spans"]]
            train.append({**d, "label_source": "synthetic-template"})
    rng.shuffle(train)
    rng.shuffle(val)

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    card = {
        "warning": "category and span labels for scam records come from keyword rules (weak labels), not human annotation",
        "sources": sources,
        "augment_templates_in_train": args.augment_templates,
        "train_size": len(train),
        "val_size": len(val),
        "category_counts": dict(Counter(r["category"] for r in records)),
        "modality_counts": {"text": sum(bool(r["text"]) for r in records), "url": sum(bool(r["url"]) for r in records)},
        "seed": args.seed,
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    print(f"\nwrote {out / 'train.jsonl'} and {out / 'val.jsonl'}")


if __name__ == "__main__":
    main()
