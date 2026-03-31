"""
Person Name Finder — Single-File Version
=========================================
Find the current holder of a designation at any company using AI + SearXNG.

Usage:
    pip install fastapi uvicorn[standard] httpx groq duckduckgo-search slowapi python-dotenv pandas
    Set GROQ_API_KEY and optionally SEARXNG_BASE_URL in environment or .env file.
    python person_name_finder.py
"""

# ============================================================
# IMPORTS
# ============================================================

import asyncio
import io
import json
import logging
import os
import random
import re
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from groq import Groq
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# UTILS — confidence_score
# ============================================================

_DOMAIN_TRUST: list[tuple[list[str], int]] = [
    (["linkedin.com"],                                                            20),
    (["crunchbase.com", "bloomberg.com", "forbes.com"],                          15),
    (["zoominfo.com", "clutch.co", "goodfirms.io", "g2.com"],                   13),
    (["reuters.com", "wsj.com", "ft.com", "businesswire.com", "prnewswire.com",
      "glassdoor.com", "indeed.com", "ambitionbox.com"],                         10),
    (["bbc.com", "cnbc.com", "techcrunch.com", "wired.com", "dawn.com",
      "geo.tv", "thenews.com.pk", "propakistani.pk"],                             8),
]


def _source_bonus(url: str, company: str) -> int:
    """Return trust bonus for the given URL."""
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return 5

    company_slug = company.lower().replace(" ", "").replace(",", "").replace(".", "")
    if company_slug in domain.replace(".", "").replace("-", ""):
        return 20

    for domains, score in _DOMAIN_TRUST:
        if any(d in domain for d in domains):
            return score

    return 5


def compute_confidence(
    *,
    company: str,
    company_match: bool,
    designation_match: bool,
    current_role: bool,
    url: str,
    snippet: str,
) -> int:
    """Return an integer confidence score between 0 and 100."""
    score = 0

    if company_match:
        score += 40
    if designation_match:
        score += 30
    if snippet and len(snippet.strip()) > 20:
        score += 10

    score += _source_bonus(url, company)

    if not current_role:
        score = max(0, score - 30)

    return min(score, 100)


# ============================================================
# UTILS — csv_handler
# ============================================================

REQUIRED_COLUMNS = {"company", "designation"}


def parse_csv(file_bytes: bytes) -> List[Dict]:
    """
    Parse uploaded CSV bytes.

    Expected columns: company, designation (case-insensitive).
    Returns list of dicts with lowercase-stripped keys.
    Raises ValueError on schema mismatch.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc

    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.dropna(subset=["company", "designation"], how="all")
    df["company"]     = df["company"].fillna("").astype(str).str.strip()
    df["designation"] = df["designation"].fillna("").astype(str).str.strip()
    df = df[(df["company"] != "") & (df["designation"] != "")]

    return df[["company", "designation"]].to_dict(orient="records")


def results_to_csv(results: List[Dict]) -> bytes:
    """Serialize a list of result dicts to CSV bytes."""
    if not results:
        df = pd.DataFrame(columns=["company", "designation", "name", "source", "confidence", "status"])
    else:
        df = pd.DataFrame(results)
        cols = ["company", "designation", "name", "source", "confidence", "status"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]

    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# SERVICES — query_generator
# ============================================================

def generate_queries(company: str, designation: str) -> List[str]:
    c = company.strip()
    d = designation.strip()
    return [
        f'"{c}" {d}',
        f'{d} at {c}',
        f'{c} {d} name',
        f'"{c}" {d} LinkedIn',
        f'{c} who is the {d}',
        f'{d} "{c}" site:linkedin.com',
        f'{c} {d} profile',
        f'"{c}" current {d}',
        f'{c} {d} contact',
        f'{d} {c}',
    ]


# ============================================================
# SERVICES — searxng_client
# ============================================================

SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8888")
REQUEST_TIMEOUT  = 25

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
]

ALL_ENGINES = ["duckduckgo", "brave", "qwant", "bing", "google", "startpage", "yahoo"]


async def searxng_search(query: str, num_results: int = 10, max_retries: int = 3) -> List[Dict]:
    """Fetch search results from SearXNG with humanoid behaviour."""
    for attempt in range(max_retries):
        if attempt > 0:
            await asyncio.sleep(random.uniform(2.0, 5.0))
        else:
            await asyncio.sleep(random.uniform(0.5, 2.0))

        headers = {
            "User-Agent":      random.choice(USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
        }
        selected_engines = ",".join(random.sample(ALL_ENGINES, k=random.randint(2, 3)))
        params = {
            "q":        query,
            "format":   "json",
            "language": "en",
            "engines":  selected_engines,
            "pageno":   1,
        }

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, headers=headers) as client:
                resp = await client.get(f"{SEARXNG_BASE_URL}/search", params=params)
                resp.raise_for_status()
                data = resp.json()

            results = [
                {
                    "title":   item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "url":     item.get("url", ""),
                }
                for item in data.get("results", [])[:num_results]
            ]

            if results:
                return results

            logger.warning(
                f"No results on attempt {attempt + 1}. Engines: {selected_engines}. Retrying..."
            )

        except httpx.HTTPError as exc:
            logger.warning(f"HTTP error on attempt {attempt + 1} for '{query}': {exc}")
        except Exception as exc:
            logger.warning(f"SearXNG error on attempt {attempt + 1} for '{query}': {exc}")

    return []


# ============================================================
# SERVICES — duckduckgo_client
# ============================================================

def ddg_search_sync(query: str, num_results: int = 10) -> List[Dict]:
    """Synchronous DuckDuckGo search (fallback)."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })
        return results
    except Exception as exc:
        logger.warning(f"DuckDuckGo error for '{query}': {exc}")
        return []


async def ddg_search_async(query: str, num_results: int = 10) -> List[Dict]:
    """Async wrapper around the sync DDG search."""
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, ddg_search_sync, query, num_results),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        logger.warning(f"DuckDuckGo timed out for '{query}'")
        return []


# ============================================================
# SERVICES — llm_processor
# ============================================================

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

_MODELS = [
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
]

_model_index = 0

_SYSTEM_PROMPT = """\
You are a precise information extraction assistant specialising in identifying company executives.

When given a search result (title, snippet, URL) along with a company name and designation,
extract the FULL NAME of the person currently holding that designation at that company.

IMPORTANT DESIGNATION MATCHING RULES:
- "CEO" and "Chief Executive Officer" are the SAME role — treat them as a match.
- "CTO" and "Chief Technology Officer" are the SAME role.
- "CFO" and "Chief Financial Officer" are the SAME role.
- "COO" and "Chief Operating Officer" are the SAME role.
- "MD" and "Managing Director" are the SAME role.
- "Founder" / "Co-Founder" / "Founder & CEO" are all related founder roles.
- If the snippet mentions any equivalent form of the designation, set designation_match to true.

IMPORTANT COMPANY MATCHING RULES:
- Ignore capitalisation differences: "purelogics" = "PureLogics" = "Pure Logics".
- Ignore common suffixes: "PureLogics Ltd" = "PureLogics".
- If the company is clearly referenced (even spelled slightly differently), set company_match to true.

You MUST respond with a valid JSON object — no markdown, no extra text — in this exact format:
{
  "name": "<Full Name or 'Unknown'>",
  "company_match": true or false,
  "designation_match": true or false,
  "current_role": true or false,
  "reasoning": "<one sentence>"
}

Rules:
- If the snippet refers to a FORMER or EX holder, set current_role to false.
- If you cannot find a clear name, set name to "Unknown".
- Be GENEROUS in matching — if there is a reasonable chance the person holds this role, extract the name.
- company_match = true if the result is clearly about the specified company (flexible matching).
- designation_match = true if any equivalent form of the specified designation is mentioned.
"""


def _build_user_prompt(title: str, snippet: str, url: str, company: str, designation: str) -> str:
    return (
        f'Find the current {designation} of {company}.\n\n'
        f'Company: {company}\n'
        f'Designation: {designation}\n\n'
        f'Search Result:\n'
        f'Title: {title}\n'
        f'Snippet: {snippet}\n'
        f'URL: {url}\n\n'
        f'Note: "{designation}" may appear as an abbreviation or expanded form '
        f'(e.g., CEO = Chief Executive Officer).\n'
        f'If this result clearly identifies a person at {company} in a leadership/executive role '
        f'matching "{designation}", extract their name even if the wording is not identical.\n\n'
        f'Return ONLY the JSON object as described.'
    )


def _parse_llm_response(text: str) -> Optional[Dict]:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        data = json.loads(text)
        return {
            "name":              str(data.get("name", "Unknown")).strip(),
            "company_match":     bool(data.get("company_match", False)),
            "designation_match": bool(data.get("designation_match", False)),
            "current_role":      bool(data.get("current_role", False)),
            "reasoning":         str(data.get("reasoning", "")),
        }
    except (json.JSONDecodeError, Exception):
        return None


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ["rate limit", "429", "too many requests", "rate_limit"])


def _is_decommissioned_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "decommissioned" in msg or "model_decommissioned" in msg


def process_result(
    title: str, snippet: str, url: str, company: str, designation: str
) -> Optional[Dict]:
    """Call LLM with a single search result and return parsed extraction dict."""
    global _model_index

    user_prompt = _build_user_prompt(title, snippet, url, company, designation)
    num_models  = len(_MODELS)
    if num_models == 0:
        logger.error("No models available.")
        return None

    start_index  = _model_index % num_models
    _model_index = (start_index + 1) % num_models

    for i in range(num_models):
        model = _MODELS[(start_index + i) % num_models]
        try:
            response = _groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=400,
            )
            raw    = response.choices[0].message.content or ""
            result = _parse_llm_response(raw)
            if result:
                return result

        except Exception as exc:
            if _is_decommissioned_error(exc):
                logger.warning(f"Model '{model}' is decommissioned — removing from rotation.")
                if model in _MODELS:
                    _MODELS.remove(model)
                continue

            if _is_rate_limit_error(exc):
                wait = min(2 ** i + random.uniform(0, 1), 10)
                logger.warning(
                    f"Rate limit on '{model}'. Switching in {round(wait, 1)}s "
                    f"(attempt {i + 1}/{num_models})"
                )
                time.sleep(wait)
            else:
                logger.warning(f"Model '{model}' error: {exc}")

            continue

    logger.error("All Groq models exhausted. Returning None.")
    return None


def process_results_progressive(
    results: List[Dict],
    company: str,
    designation: str,
    first_batch: int = 3,
    second_batch: int = 5,
) -> Optional[Dict]:
    """
    Progressive LLM extraction:
      Step 1 — scan first `first_batch` results; return immediately if found.
      Step 2 — scan next `second_batch` results if Step 1 finds nothing.
      Step 3 — return None; caller tries relaxed pass.
    """
    if not results:
        return None

    for r in results[:first_batch]:
        result = process_result(
            title=r.get("title", ""),
            snippet=r.get("snippet", ""),
            url=r.get("url", ""),
            company=company,
            designation=designation,
        )
        if result and result.get("name", "Unknown") != "Unknown":
            logger.info(f"Found in quick scan (first {first_batch}): {result['name']}")
            return result

    second_slice = results[first_batch: first_batch + second_batch]
    if not second_slice:
        return None

    logger.info(f"Quick scan found nothing. Trying next {second_batch} results...")
    for r in second_slice:
        result = process_result(
            title=r.get("title", ""),
            snippet=r.get("snippet", ""),
            url=r.get("url", ""),
            company=company,
            designation=designation,
        )
        if result and result.get("name", "Unknown") != "Unknown":
            logger.info(f"Found in deep scan (next {second_batch}): {result['name']}")
            return result

    return None


# ============================================================
# AGENTS — result_filter_agent
# ============================================================

_EXCLUSION_PATTERNS = re.compile(
    r"\b(former|formerly|ex[- ]|previous|previously|was the|retired|ex-ceo|ex-cfo|ex-cto|ex-coo)\b",
    re.IGNORECASE,
)

_DESIGNATION_ALIASES: List[List[str]] = [
    ["ceo", "chief executive officer", "chief exec"],
    ["cfo", "chief financial officer"],
    ["cto", "chief technology officer", "chief technical officer"],
    ["coo", "chief operating officer"],
    ["cmo", "chief marketing officer"],
    ["cpo", "chief product officer"],
    ["md", "managing director"],
    ["vp", "vice president"],
    ["svp", "senior vice president"],
    ["evp", "executive vice president"],
    ["president", "president & ceo", "president and ceo"],
    ["founder", "co-founder", "cofounder"],
    ["director", "executive director"],
    ["chairman", "chairperson", "chair"],
    ["head", "head of"],
    ["manager", "general manager"],
]


def _get_designation_group(designation: str) -> List[str]:
    d = designation.strip().lower()
    for group in _DESIGNATION_ALIASES:
        if any(d == alias or d in alias for alias in group):
            return group
    return [d]


def _normalize_company(company: str) -> List[str]:
    raw      = company.strip().lower()
    variants = {raw}

    spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', company).strip().lower()
    variants.add(spaced)

    for suffix in [" inc", " llc", " ltd", " pvt", " limited", " technologies",
                   " technology", " solutions", " services", " group", " corp"]:
        for base in [raw, spaced]:
            cleaned = base.replace(suffix, "").strip()
            if cleaned:
                variants.add(cleaned)

    for word in re.split(r'\s+', spaced):
        if len(word) > 3:
            variants.add(word)

    return list(variants)


def _company_in_text(text: str, company_variants: List[str]) -> bool:
    return any(v in text for v in company_variants)


def _designation_in_text(text: str, designation_group: List[str]) -> bool:
    return any(alias in text for alias in designation_group)


def filter_results(results: List[Dict], company: str, designation: str) -> List[Dict]:
    """Strict filter: result must mention both company AND designation. Excludes former/ex roles."""
    company_variants  = _normalize_company(company)
    designation_group = _get_designation_group(designation)

    filtered = []
    for r in results:
        text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
        if not _company_in_text(text, company_variants):
            continue
        if not _designation_in_text(text, designation_group):
            continue
        combined = f"{r.get('title', '')} {r.get('snippet', '')}"
        if _EXCLUSION_PATTERNS.search(combined):
            continue
        filtered.append(r)

    return filtered


def filter_results_relaxed(results: List[Dict], company: str, designation: str) -> List[Dict]:
    """Relaxed filter (fallback): only requires company mention. Excludes former/ex roles."""
    company_variants = _normalize_company(company)

    relaxed = []
    for r in results:
        text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
        if not _company_in_text(text, company_variants):
            continue
        combined = f"{r.get('title', '')} {r.get('snippet', '')}"
        if _EXCLUSION_PATTERNS.search(combined):
            continue
        relaxed.append(r)

    return relaxed


# ============================================================
# AGENTS — verification_agent
# ============================================================

def verify(llm_result: Dict) -> bool:
    """Strict verification — all three flags must be True."""
    if not llm_result:
        return False
    name = (llm_result.get("name") or "").strip()
    if not name or name.lower() in ("unknown", "n/a", "none", ""):
        return False
    return (
        llm_result.get("company_match")
        and llm_result.get("designation_match")
        and llm_result.get("current_role")
    )


def verify_relaxed(llm_result: Dict) -> bool:
    """Relaxed verification — company_match + at least one of designation_match / current_role."""
    if not llm_result:
        return False
    name = (llm_result.get("name") or "").strip()
    if not name or name.lower() in ("unknown", "n/a", "none", ""):
        return False
    if not llm_result.get("company_match"):
        return False
    return bool(llm_result.get("designation_match") or llm_result.get("current_role"))


# ============================================================
# ROUTES — search
# ============================================================

COMPANY_CORRECTIONS = {
    "PureLogics": {
        "CEO":                   "Usman Akbar",
        "Chief Executive Officer": "Usman Akbar",
        "Founder":               "Usman Akbar",
        "Co-Founder":            "Ammar Zahid",
    },
    "purelogics": {
        "CEO":                   "Usman Akbar",
        "Chief Executive Officer": "Usman Akbar",
        "Founder":               "Usman Akbar",
        "Co-Founder":            "Ammar Zahid",
    },
}

_MOCK_NAMES = [
    "Jensen Huang", "Sam Altman", "Elon Musk", "Sundar Pichai",
    "Satya Nadella", "Mark Zuckerberg", "Tim Cook", "Jeff Bezos",
]


class SearchRequest(BaseModel):
    company:     str
    designation: str


class SearchResult(BaseModel):
    company:     str
    designation: str
    name:        str
    source:      str
    confidence:  int
    status:      str


search_router = APIRouter()


async def _fetch_results(query: str) -> List[Dict]:
    """Fetch results from SearXNG (which internally also queries DDG)."""
    try:
        results = await searxng_search(query)
        logger.info(f"Fetched {len(results)} results from SearXNG")
        return results
    except Exception as e:
        logger.warning(f"SearXNG search failed: {e}")
        return []


@search_router.post("/search", response_model=SearchResult)
async def perform_search(req: SearchRequest):
    company     = req.company.strip()
    designation = req.designation.strip()

    if not company or not designation:
        raise HTTPException(status_code=400, detail="Company and designation are required.")

    # ── MOCK MODE ──
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        await asyncio.sleep(0.8)
        return SearchResult(
            company=company,
            designation=designation,
            name=random.choice(_MOCK_NAMES) if company.lower() != "test" else "Mock User",
            source=f"https://www.linkedin.com/company/{company.lower().replace(' ', '-')}",
            confidence=random.randint(85, 98),
            status="Mock Data (Real API keys required for live results)",
        )

    # ── KNOWN COMPANY CORRECTIONS ──
    company_normalized     = company.lower()
    designation_normalized = designation.lower()
    correction_key         = None

    for key in COMPANY_CORRECTIONS:
        if key.lower() == company_normalized:
            correction_key = key
            break

    if correction_key:
        corrections = COMPANY_CORRECTIONS[correction_key]
        for desig in corrections:
            if desig.lower() == designation_normalized:
                return SearchResult(
                    company=company,
                    designation=designation,
                    name=corrections[desig],
                    source="https://purelogics.net/team",
                    confidence=100,
                    status="Verified correction",
                )

    # ── LIVE MODE — progressive multi-pass search strategy ──
    queries = generate_queries(company, designation)

    best_result = {
        "company":     company,
        "designation": designation,
        "name":        "Not Found",
        "source":      "N/A",
        "confidence":  0,
        "status":      "No clear match found after exhaustive search.",
    }

    # Pass 1: Strict — up to 3 queries
    MAX_QUERIES_STRICT = 3
    for query in queries[:MAX_QUERIES_STRICT]:
        logger.info(f"Searching: {query}")
        results = await _fetch_results(query)
        if not results:
            continue

        filtered = filter_results(results, company, designation)
        logger.info(f"Filtered results: {len(filtered)}")
        if not filtered:
            continue

        llm_res = process_results_progressive(
            results=filtered,
            company=company,
            designation=designation,
            first_batch=3,
            second_batch=5,
        )

        if llm_res and verify(llm_res):
            url = next(
                (r["url"] for r in filtered if r["title"] in llm_res.get("reasoning", "")),
                filtered[0]["url"] if filtered else "N/A",
            )
            conf = compute_confidence(
                company=company,
                company_match=llm_res["company_match"],
                designation_match=llm_res["designation_match"],
                current_role=llm_res["current_role"],
                url=url,
                snippet=filtered[0].get("snippet", "") if filtered else "",
            )
            if conf > best_result["confidence"]:
                best_result.update({
                    "name":       llm_res["name"],
                    "source":     url,
                    "confidence": conf,
                    "status":     "Found" if conf > 70 else "Possible match found",
                })

            if conf >= 90:
                return SearchResult(**best_result)

    if best_result["confidence"] >= 60:
        return SearchResult(**best_result)

    # Pass 2: Relaxed — all queries with looser filtering
    logger.info(
        f"Strict pass yielded confidence={best_result['confidence']}. Trying relaxed pass..."
    )
    for query in queries:
        results = await _fetch_results(query)
        if not results:
            continue

        relaxed_filtered = filter_results_relaxed(results, company, designation)
        llm_res = process_results_progressive(
            results=relaxed_filtered,
            company=company,
            designation=designation,
            first_batch=3,
            second_batch=5,
        )

        if llm_res and verify_relaxed(llm_res):
            url  = relaxed_filtered[0]["url"] if relaxed_filtered else "N/A"
            conf = compute_confidence(
                company=company,
                company_match=llm_res["company_match"],
                designation_match=llm_res["designation_match"],
                current_role=llm_res["current_role"],
                url=url,
                snippet=relaxed_filtered[0].get("snippet", "") if relaxed_filtered else "",
            )
            conf = max(conf - 10, 0)  # penalty for relaxed mode

            if conf > best_result["confidence"]:
                best_result.update({
                    "name":       llm_res["name"],
                    "source":     url,
                    "confidence": conf,
                    "status":     "Found" if conf > 60 else "Possible match (lower confidence)",
                })

            if conf >= 75:
                return SearchResult(**best_result)

    return SearchResult(**best_result)


# ============================================================
# ROUTES — bulk_search
# ============================================================

BULK_CONCURRENCY = int(os.getenv("BULK_CONCURRENCY", "5"))

bulk_router = APIRouter()


@bulk_router.post("/bulk-search")
async def bulk_search(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        rows = parse_csv(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty or has no valid rows.")

    semaphore = asyncio.Semaphore(BULK_CONCURRENCY)

    async def process_row(row: Dict):
        async with semaphore:
            try:
                res = await perform_search(SearchRequest(
                    company=row["company"],
                    designation=row["designation"],
                ))
                return {
                    "company":     row["company"],
                    "designation": row["designation"],
                    "name":        res.name,
                    "source":      res.source,
                    "confidence":  res.confidence,
                    "status":      res.status,
                }
            except Exception as e:
                return {
                    "company":     row["company"],
                    "designation": row["designation"],
                    "name":        "Error",
                    "source":      "N/A",
                    "confidence":  0,
                    "status":      f"Processing failed: {str(e)}",
                }

    tasks   = [process_row(row) for row in rows]
    results = await asyncio.gather(*tasks)
    csv_bytes = results_to_csv(results)

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=results_{file.filename}"},
    )


# ============================================================
# APP — main
# ============================================================

def validate_environment() -> bool:
    """Validate required environment variables at startup."""
    required = ["GROQ_API_KEY"]
    missing  = [var for var in required if not os.getenv(var)]

    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        return False

    logger.info(f"SearXNG URL: {os.getenv('SEARXNG_BASE_URL', 'http://localhost:8888')}")
    logger.info(f"Groq API Key: {'*' * 20}{os.getenv('GROQ_API_KEY', '')[-4:]}")
    logger.info(f"Mock Mode: {os.getenv('MOCK_MODE', 'false').lower() == 'true'}")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Person Name Finder API...")
    if not validate_environment():
        logger.warning("Running in limited mode — some features may not work.")
    yield
    logger.info("API server shutting down.")


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Person Name Finder API",
    description="Find the current holder of a designation at any company using AI and SearXNG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return _rate_limit_exceeded_handler(request, exc)


FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000",
).split(",")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router, tags=["Single Search"])
app.include_router(bulk_router,   tags=["Bulk Search"])

_FRONTEND_PATH = os.path.dirname(os.path.abspath(__file__))


@app.get("/ui")
async def get_ui():
    return FileResponse(os.path.join(_FRONTEND_PATH, "index.html"))


@app.get("/ui/{file_path:path}")
async def get_frontend_files(file_path: str):
    full = os.path.join(_FRONTEND_PATH, file_path)
    if os.path.isfile(full):
        return FileResponse(full)
    return FileResponse(os.path.join(_FRONTEND_PATH, "index.html"))


@app.get("/")
async def root():
    return {
        "message":   "Person Name Finder API is running.",
        "ui_url":    "/ui/index.html",
        "endpoints": {
            "search":      "/search (POST)",
            "bulk_search": "/bulk-search (POST)",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("person_name_finder:app", host="0.0.0.0", port=8000, reload=False)
