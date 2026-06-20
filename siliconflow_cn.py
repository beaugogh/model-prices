import argparse
import csv
import html
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from http.cookiejar import MozillaCookieJar
from typing import Any, Iterable

# Use system certificate store for SSL (fixes corporate proxy issues)
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass  # truststore not installed, use default SSL handling

# Force UTF-8 output on Windows to handle Chinese characters and ¥ symbol
if sys.platform == "win32":
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding.lower() != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import requests

try:
    import browser_cookie3
except ImportError:  # pragma: no cover - handled at runtime for friendlier errors
    browser_cookie3 = None


DEFAULT_ENDPOINTS = [
    # The cloud endpoints are behind SiliconFlow SSO, so they need cookies.
    "https://cloud.siliconflow.cn/api/v1/models/pricing",
    "https://cloud.siliconflow.cn/api/v1/models",
    # The OpenAI-compatible API needs an API key and may not include pricing.
    "https://api.siliconflow.cn/v1/models",
]
DEFAULT_PRICING_URL = "https://siliconflow.cn/pricing"

LIST_KEYS = ("data", "models", "items", "list", "records", "result")
MODEL_ID_KEYS = ("model_id", "modelId", "model", "id", "name", "display_name", "displayName")
CONTEXT_KEYS = ("context_length", "contextLength", "context_window", "contextWindow", "max_context", "maxContext", "max_tokens")
CACHE_PRICE_KEYS = ("cache_price", "cachePrice", "cache_hit_price", "cacheHitPrice", "cached_input_price", "cachedInputPrice")
INPUT_PRICE_KEYS = ("input_price", "inputPrice", "prompt_price", "promptPrice", "input", "prompt")
OUTPUT_PRICE_KEYS = ("output_price", "outputPrice", "completion_price", "completionPrice", "output", "completion")
DEFAULT_CONFIG_PATH = "config.yaml"

# Type ordering for sorting
TYPE_ORDER = {
    "chat": 0,
    "text-to-image": 1,
    "text-to-video": 2,
    "image-to-video": 3,
    "text-to-speech": 4,
    "speech-to-text": 5,
    "embedding": 6,
    "reranker": 7,
}

# Models to exclude from output
EXCLUDED_MODELS = {
    "DeepSeek-R1",
    "DeepSeek-V3",
    "DeepSeek-V3.1-Terminus",
    "DeepSeek-V3.2",
    "Qwen2.5-7B-Instruct",
}


@dataclass
class ModelPrice:
    org: str
    model_id: str
    category: str
    context: str
    pricing_unit: str
    cache_hit: str
    input_price: str
    output_price: str


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0:1] in ("'", '"') and value[-1:] == value[0]:
        return value[1:-1]

    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}

    config: dict[str, Any] = {}
    current_list_key = None
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("- "):
                if current_list_key is None:
                    raise ValueError(f"{path}: list item without a parent key: {raw_line.rstrip()}")
                config[current_list_key].append(parse_scalar(stripped[2:]))
                continue

            current_list_key = None
            if ":" not in stripped:
                raise ValueError(f"{path}: expected 'key: value': {raw_line.rstrip()}")

            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                config[key] = []
                current_list_key = key
            else:
                config[key] = parse_scalar(value)

    return config


def config_endpoints(config: dict[str, Any]) -> list[str] | None:
    endpoints = config.get("endpoints", config.get("endpoint"))
    if endpoints is None:
        return None
    if isinstance(endpoints, str):
        return [endpoints]
    if isinstance(endpoints, list):
        return [str(endpoint) for endpoint in endpoints if endpoint]
    raise ValueError("config endpoint/endpoints must be a string or list")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=os.getenv("SILICONFLOW_CONFIG", DEFAULT_CONFIG_PATH),
        help="Path to config YAML. Defaults to SILICONFLOW_CONFIG or config.yaml.",
    )
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Fetch SiliconFlow model pricing and print a comparison table.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        default=None,
        help="JSON endpoint to try instead of the public pricing page. Can be passed more than once.",
    )
    parser.add_argument(
        "--pricing-url",
        default=config.get("pricing_url", DEFAULT_PRICING_URL),
        help="Public SiliconFlow pricing page URL. Defaults to config pricing_url or siliconflow.cn/pricing.",
    )
    parser.add_argument(
        "--api-key",
        default=config.get("api_key") or os.getenv("SILICONFLOW_API_KEY"),
        help="SiliconFlow API key. Defaults to config api_key or SILICONFLOW_API_KEY.",
    )
    parser.add_argument(
        "--cookie",
        default=config.get("cookie") or os.getenv("SILICONFLOW_COOKIE"),
        help="Raw Cookie header copied from browser DevTools. Defaults to config cookie or SILICONFLOW_COOKIE.",
    )
    parser.add_argument(
        "--cookie-file",
        default=config.get("cookie_file"),
        help="Netscape/Mozilla cookie file exported from your browser.",
    )
    parser.add_argument(
        "--browser-cookies",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("browser_cookies", False)),
        help="Try loading cookies from the local Chrome profile with browser-cookie3.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(config.get("timeout") or os.getenv("SILICONFLOW_TIMEOUT", "8")),
        help="Per-request timeout in seconds. Defaults to config timeout, SILICONFLOW_TIMEOUT, or 8.",
    )
    parser.add_argument(
        "--use-env-proxy",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("use_env_proxy", False)),
        help="Use HTTP_PROXY/HTTPS_PROXY from the environment. Disabled by default to avoid proxy hangs.",
    )
    parser.add_argument(
        "--debug-json",
        default=config.get("debug_json"),
        help="Write the raw JSON response to this file for endpoint/field inspection.",
    )
    parser.add_argument(
        "--csv",
        default=config.get("csv"),
        help="Write the normalized table to CSV.",
    )
    parser.add_argument(
        "--md",
        "--markdown",
        dest="markdown",
        default=config.get("markdown"),
        help="Write the normalized table to a Markdown file.",
    )
    args = parser.parse_args()
    if args.endpoint is None:
        args.endpoint = config_endpoints(config)
    return args


def build_session(args: argparse.Namespace) -> requests.Session:
    session = requests.Session()
    session.trust_env = args.use_env_proxy
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://cloud.siliconflow.cn",
            "Referer": "https://cloud.siliconflow.cn/models",
        }
    )

    if args.api_key:
        session.headers["Authorization"] = f"Bearer {args.api_key}"
    if args.cookie:
        session.headers["Cookie"] = args.cookie
    if args.cookie_file:
        cookie_jar = MozillaCookieJar(args.cookie_file)
        cookie_jar.load(ignore_discard=True, ignore_expires=True)
        session.cookies.update(cookie_jar)
    if args.browser_cookies:
        if browser_cookie3 is None:
            raise RuntimeError("browser-cookie3 is not installed. Run: pip install -r requirements.txt")
        session.cookies.update(browser_cookie3.chrome(domain_name="siliconflow.cn"))

    return session


def default_endpoints_for(args: argparse.Namespace) -> list[str]:
    if args.endpoint:
        return args.endpoint
    return []


def fetch_first_json(session: requests.Session, endpoints: list[str], timeout: float) -> tuple[str, Any]:
    errors = []
    for endpoint in endpoints:
        print(f"Trying {endpoint} ...", file=sys.stderr)
        try:
            response = session.get(endpoint, timeout=timeout, allow_redirects=False)
        except requests.RequestException as exc:
            errors.append(f"{endpoint}: request failed: {exc}")
            continue

        content_type = response.headers.get("content-type", "")
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location", "")
            errors.append(f"{endpoint}: redirected to {location or 'another URL'}")
            continue
        if response.status_code == 401:
            errors.append(f"{endpoint}: unauthorized; pass --api-key or --cookie")
            continue
        if response.status_code != 200:
            errors.append(f"{endpoint}: HTTP {response.status_code}: {response.text[:200]}")
            continue
        if "json" not in content_type.lower():
            errors.append(f"{endpoint}: non-JSON response ({content_type}): {response.text[:200]}")
            continue

        try:
            return endpoint, response.json()
        except ValueError as exc:
            errors.append(f"{endpoint}: invalid JSON: {exc}")

    raise RuntimeError("No endpoint returned usable JSON.\n" + "\n".join(errors))


class ScriptCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_script = False
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.in_script = tag == "script"

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_script = False

    def handle_data(self, data: str) -> None:
        if self.in_script:
            self.scripts.append(data)


def iter_next_flight_values(page_html: str) -> Iterable[Any]:
    parser = ScriptCollector()
    parser.feed(page_html)

    for script in parser.scripts:
        for match in re.finditer(r"self\.__next_f\.push\((\[.*?\])\)", script, re.S):
            try:
                payload = json.loads(match.group(1))
            except ValueError:
                continue
            if len(payload) < 2 or not isinstance(payload[1], str):
                continue

            for line in payload[1].splitlines():
                line_match = re.match(r"^[0-9a-f]+:(.*)$", line, re.S)
                if not line_match:
                    continue
                try:
                    yield json.loads(line_match.group(1))
                except ValueError:
                    continue


def find_pricing_page_data(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        models = value.get("models")
        categorized_models = [
            model
            for key in ("chats", "images", "audios", "videos")
            for model in value.get(key, [])
            if isinstance(model, dict) and model.get("modelName")
        ]
        pricing_items = value.get("pricingApiItems")
        if (
            isinstance(pricing_items, list)
            and categorized_models
        ):
            return value
        if (
            isinstance(models, list)
            and isinstance(pricing_items, list)
            and any(isinstance(model, dict) and model.get("modelName") for model in models)
        ):
            return value
        for item in value.values():
            found = find_pricing_page_data(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_pricing_page_data(item)
            if found is not None:
                return found
    return None


def fetch_pricing_page_data(session: requests.Session, url: str, timeout: float) -> tuple[str, Any]:
    print(f"Trying {url} ...", file=sys.stderr)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    for value in iter_next_flight_values(response.text):
        data = find_pricing_page_data(value)
        if data is not None:
            return url, data
    raise RuntimeError("Fetched the pricing page, but could not find embedded pricing data.")


def find_model_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = find_model_list(value)
            if nested:
                return nested

    best: list[dict[str, Any]] = []
    stack = list(payload.values())
    while stack:
        value = stack.pop()
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            if len(value) > len(best):
                best = value
        elif isinstance(value, dict):
            stack.extend(value.values())
    return best


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


def pick(flat: dict[str, Any], keys: Iterable[str]) -> Any:
    normalized = {key.lower().replace("_", ""): key for key in flat}
    for wanted in keys:
        direct = normalized.get(wanted.lower().replace("_", ""))
        if direct is not None:
            return flat[direct]

    wanted_fragments = [key.lower().replace("_", "") for key in keys]
    for key, value in flat.items():
        compact = key.lower().replace("_", "")
        if any(fragment in compact for fragment in wanted_fragments):
            return value
    return None


def format_context(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        context = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    if context <= 0:
        return "N/A"
    if context >= 1_000_000:
        return f"{float(context / Decimal(1_000_000)):.1f}M"
    if context >= 1_000:
        return f"{float(context / Decimal(1_000)):.1f}K"
    return f"{context:g}"


def format_price(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    if isinstance(value, str) and ("¥" in value or "$" in value or "免费" in value):
        return value.strip()
    try:
        price = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    return f"¥{price.normalize():f}"


def format_yuan(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        price = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    if price == 0:
        return "免费"
    return f"¥{price.quantize(Decimal('0.01'))}"


def normalize_pricing_key(value: str) -> str:
    return (
        (value or "")
        .lower()
        .replace(".online.", ".")
        .replace(".online", "")
        .replace(".input-tokens", "")
        .replace(".output-tokens", "")
        .replace(".cached-input-tokens", "")
        .replace(".cached-intput-tokens", "")
        .strip()
    )


def pricing_index(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        for key in (
            item.get("playgroundName"),
            (item.get("skuName") or "").replace(".online", ""),
            (item.get("objectName") or "").split(".online")[0],
        ):
            normalized = normalize_pricing_key(str(key or ""))
            if normalized:
                index.setdefault(normalized, []).append(item)
    return index


def model_pricing_items(model: dict[str, Any], index: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    model_name = str(model.get("modelName") or "")
    display_name = str(model.get("DisplayName") or model_name.split("/")[-1] or "")
    manufacturer = str(model.get("mf") or "")
    candidates = [
        model_name,
        re.sub(r"/([^/]+)$", f"/{display_name}", model_name),
        f"{manufacturer}/{display_name}" if manufacturer else "",
        display_name if "/" in display_name else "",
    ]
    for candidate in candidates:
        items = index.get(normalize_pricing_key(candidate))
        if items:
            return items
    return []


def component_prices(items: list[dict[str, Any]], component_code: str) -> str:
    parts = []
    seen = set()
    for item in items:
        if item.get("componentCode") != component_code:
            continue
        dedupe_key = (
            item.get("componentCode"),
            item.get("coordinateDesc"),
            item.get("realTimePriceCnyUnit"),
            item.get("unitZhCnName"),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        value = item.get("realTimePriceCnyUnit")
        try:
            price = Decimal(str(value))
        except InvalidOperation:
            continue
        if "K tokens" in str(item.get("unitZhCnName") or ""):
            price *= Decimal(1000)
        label = clean_coordinate_label(str(item.get("coordinateDesc") or ""))
        rendered = format_yuan(price)
        parts.append(f"{html.escape(label)}: {rendered}" if label else rendered)
    return "<br>".join(parts) if parts else "N/A"


def clean_coordinate_label(value: str) -> str:
    labels = []
    for part in re.split(r"[;；]", value or ""):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^(input|output)-tokens-range:\s*(.+)$", part)
        if match:
            token_type, label = match.groups()
            label = re.sub(r"^(输入|输出)\s*", "输入 " if token_type == "input" else "输出 ", label)
            labels.append(label.replace("+♾️", "+∞").replace("）", ")").strip())
        else:
            labels.append(part.replace("+♾️", "+∞").replace("）", ")").strip())
    return "; ".join(labels)


def model_spec_price(model: dict[str, Any], specification: str) -> Any:
    pricing = model.get("pricing")
    if isinstance(pricing, list):
        for item in pricing:
            if isinstance(item, dict) and item.get("specification") == specification:
                return item.get("price")
    return None


def normalize_pricing_page_models(payload: dict[str, Any]) -> list[ModelPrice]:
    index = pricing_index([item for item in payload.get("pricingApiItems", []) if isinstance(item, dict)])
    rows: list[ModelPrice] = []
    payload_models = payload.get("models")
    if not (
        isinstance(payload_models, list)
        and any(isinstance(model, dict) and model.get("modelName") for model in payload_models)
    ):
        payload_models = [
            model
            for key in ("chats", "embeddings", "images", "audios", "videos")
            for model in payload.get(key, [])
            if isinstance(model, dict)
        ]

    for model in payload_models or []:
        if not isinstance(model, dict):
            continue

        model_type = model.get("type")
        sub_type = model.get("subType")

        # Determine pricing unit based on model type
        if model_type == "text" and sub_type in ("chat", "embedding", "reranker"):
            pricing_unit = "1M tokens"
            items = model_pricing_items(model, index)
            input_price = component_prices(items, "input-tokens")
            output_price = component_prices(items, "output-tokens")
            cache_price = component_prices(items, "cached-input-tokens")

            if input_price == "N/A":
                input_price = format_yuan(model_spec_price(model, "prompt") or model.get("inputPrice"))
            if output_price == "N/A":
                output_price = format_yuan(model_spec_price(model, "completion") or model.get("price"))
        elif model_type == "image":
            pricing_unit = "per image"
            items = model_pricing_items(model, index)
            input_price = component_prices(items, "image-cnt")
            output_price = "N/A"
            cache_price = "N/A"
        elif model_type == "video":
            pricing_unit = "per video"
            items = model_pricing_items(model, index)
            input_price = component_prices(items, "video-cnt")
            output_price = "N/A"
            cache_price = "N/A"
        elif model_type == "audio":
            pricing_unit = "1K chars"
            items = model_pricing_items(model, index)
            input_price = component_prices(items, "utf8-bytes")
            output_price = "N/A"
            cache_price = "N/A"
        else:
            continue

        model_id = str(model.get("modelName") or model.get("targetModelName") or model.get("DisplayName") or "Unknown")
        parts = model_id.split("/")
        if len(parts) >= 3:
            # Handle Pro/org/model-name format
            org = "/".join(parts[:-1])
            model_name = parts[-1]
        elif len(parts) == 2:
            org = parts[0]
            model_name = parts[1]
        else:
            org = ""
            model_name = model_id

        # Skip excluded models
        if model_name in EXCLUDED_MODELS:
            continue

        rows.append(
            ModelPrice(
                org=org,
                model_id=model_name,
                category=str(sub_type or model_type or "text"),
                context=format_context(model.get("contextLen")),
                pricing_unit=pricing_unit,
                cache_hit=cache_price,
                input_price=input_price,
                output_price=output_price,
            )
        )

    return sorted(rows, key=lambda row: (
        TYPE_ORDER.get(row.category, 99),
        row.org.lower(),
        row.model_id.lower(),
        row.input_price,
        row.output_price
    ))


def normalize_models(models: list[dict[str, Any]]) -> list[ModelPrice]:
    rows = []
    for model in models:
        flat = flatten(model)
        model_id = str(pick(flat, MODEL_ID_KEYS) or "Unknown")
        parts = model_id.split("/")
        if len(parts) >= 3:
            org = "/".join(parts[:-1])
            model_name = parts[-1]
        elif len(parts) == 2:
            org = parts[0]
            model_name = parts[1]
        else:
            org = ""
            model_name = model_id

        # Skip excluded models
        if model_name in EXCLUDED_MODELS:
            continue

        rows.append(
            ModelPrice(
                org=org,
                model_id=model_name,
                category=str(pick(flat, ("category", "type", "subType")) or "N/A"),
                context=format_context(pick(flat, CONTEXT_KEYS)),
                pricing_unit="1M tokens",
                cache_hit=format_price(pick(flat, CACHE_PRICE_KEYS)),
                input_price=format_price(pick(flat, INPUT_PRICE_KEYS)),
                output_price=format_price(pick(flat, OUTPUT_PRICE_KEYS)),
            )
        )
    return sorted(rows, key=lambda row: (
        TYPE_ORDER.get(row.category, 99),
        row.org.lower(),
        row.model_id.lower(),
        row.input_price,
        row.output_price
    ))


def markdown_table(rows: list[ModelPrice], source: str) -> str:
    lines = [
        f"Source: {source}",
        "",
        "| Org | Model ID | Type | Context | Unit | Cache | Input | Output |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.org} | {row.model_id} | {row.category} | {row.context} | {row.pricing_unit} | "
            f"{row.cache_hit} | {row.input_price} | {row.output_price} |"
        )
    return "\n".join(lines) + "\n"


def print_markdown_table(rows: list[ModelPrice], source: str) -> None:
    print(markdown_table(rows, source), end="")


def write_csv(rows: list[ModelPrice], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Org", "Model ID", "Type", "Context", "Unit", "Cache", "Input", "Output"])
        for row in rows:
            writer.writerow([row.org, row.model_id, row.category, row.context, row.pricing_unit, row.cache_hit, row.input_price, row.output_price])


def write_markdown(rows: list[ModelPrice], source: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(markdown_table(rows, source))


def main() -> int:
    try:
        args = parse_args()
        endpoints = default_endpoints_for(args)
        session = build_session(args)
        if endpoints:
            source, payload = fetch_first_json(session, endpoints, args.timeout)
        else:
            source, payload = fetch_pricing_page_data(session, args.pricing_url, args.timeout)
    except KeyboardInterrupt:
        print("\nCanceled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Failed to fetch SiliconFlow pricing:\n{exc}", file=sys.stderr)
        return 1

    if args.debug_json:
        with open(args.debug_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    if isinstance(payload, dict) and isinstance(payload.get("pricingApiItems"), list):
        rows = normalize_pricing_page_models(payload)
    else:
        models = find_model_list(payload)
        if not models:
            print("Fetched JSON, but could not find a model list. Use --debug-json to inspect it.", file=sys.stderr)
            return 1
        rows = normalize_models(models)

    if not rows:
        print("Fetched pricing data, but no text model pricing rows were found.", file=sys.stderr)
        return 1

    print_markdown_table(rows, source)
    if args.markdown:
        write_markdown(rows, source, args.markdown)
        print(f"\nWrote Markdown: {args.markdown}")
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nWrote CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
