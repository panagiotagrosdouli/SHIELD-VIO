#!/usr/bin/env python3
"""Resolve an individual EuRoC archive through the ETH DSpace REST API."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_API_ROOT = "https://www.research-collection.ethz.ch/server/api/"
DEFAULT_HANDLE = "20.500.11850/263889"
DEFAULT_FILENAME = "MH_01_easy.zip"
FALLBACK_URL = "https://blog.tofu.icu/media/5076cd8923429f1c51c72ae172e14aea.zip"
USER_AGENT = "SHIELD-VIO GitHub Actions dataset resolver"


def _load_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={"Accept": "application/hal+json", "User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return payload


def _link(resource: dict[str, Any], relation: str) -> str:
    try:
        href = resource["_links"][relation]["href"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"DSpace response has no {relation!r} link") from exc
    if not isinstance(href, str) or not href:
        raise RuntimeError(f"DSpace {relation!r} link is invalid")
    return href


def _resources(payload: dict[str, Any], relation: str) -> list[dict[str, Any]]:
    try:
        resources = payload["_embedded"][relation]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"DSpace response has no embedded {relation!r} resources") from exc
    if not isinstance(resources, list):
        raise RuntimeError(f"DSpace embedded {relation!r} value is not a list")
    return [resource for resource in resources if isinstance(resource, dict)]


def _walk_objects(value: Any):
    """Yield every JSON object, independent of the DSpace HAL nesting version."""

    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _linked_item(
    resource: dict[str, Any],
    *,
    root: str,
    handle: str,
    load_json: Callable[[str], dict[str, Any]],
) -> dict[str, Any] | None:
    """Extract an item from either old or current DSpace discovery wrappers."""

    embedded = resource.get("_embedded")
    indexable = embedded.get("indexableObject") if isinstance(embedded, dict) else None
    if isinstance(indexable, dict) and indexable.get("handle") == handle:
        links = resource.get("_links")
        relation = links.get("indexableObject") if isinstance(links, dict) else None
        href = relation.get("href") if isinstance(relation, dict) else None
        if isinstance(href, str) and href:
            return load_json(urljoin(root, href))
        indexable_links = indexable.get("_links")
        self_link = indexable_links.get("self") if isinstance(indexable_links, dict) else None
        href = self_link.get("href") if isinstance(self_link, dict) else None
        if isinstance(href, str) and href:
            return load_json(urljoin(root, href))
        return indexable

    if resource.get("handle") != handle:
        return None
    links = resource.get("_links")
    self_link = links.get("self") if isinstance(links, dict) else None
    href = self_link.get("href") if isinstance(self_link, dict) else None
    if isinstance(href, str) and href:
        return load_json(urljoin(root, href))
    if isinstance(links, dict) and "bundles" in links:
        return resource

    item_id = resource.get("uuid") or resource.get("id")
    if isinstance(item_id, str) and item_id:
        return load_json(urljoin(root, f"core/items/{item_id}"))
    return None


def _resolve_item(
    root: str,
    handle: str,
    load_json: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    for identifier in (f"hdl:{handle}", handle, f"https://hdl.handle.net/{handle}"):
        pid_url = urljoin(root, "pid/find") + "?" + urlencode({"id": identifier})
        try:
            return load_json(pid_url)
        except HTTPError as exc:
            failures.append(f"{identifier} -> HTTP {exc.code}")

    search_queries = (
        f'"{handle}"',
        handle,
        f'dc.identifier.uri:"http://hdl.handle.net/{handle}"',
    )
    for query in search_queries:
        search_url = urljoin(root, "discover/search/objects") + "?" + urlencode(
            {"query": query, "dsoType": "item", "size": 100}
        )
        try:
            search = load_json(search_url)
        except HTTPError as exc:
            failures.append(f"Discovery {query!r} -> HTTP {exc.code}")
            continue

        for resource in _walk_objects(search):
            try:
                item = _linked_item(
                    resource,
                    root=root,
                    handle=handle,
                    load_json=load_json,
                )
            except HTTPError as exc:
                failures.append(f"Discovery item link -> HTTP {exc.code}")
                continue
            if item is not None:
                return item

        top_level = ", ".join(sorted(search)) or "<empty>"
        failures.append(f"Discovery {query!r} had no matching item (keys: {top_level})")

    detail = "; ".join(failures)
    raise RuntimeError(f"Could not resolve hdl:{handle} through DSpace ({detail})")


def resolve_download_url(
    *,
    handle: str = DEFAULT_HANDLE,
    filename: str = DEFAULT_FILENAME,
    api_root: str = DEFAULT_API_ROOT,
    load_json: Callable[[str], dict[str, Any]] = _load_json,
) -> str:
    """Return the public content URL for one filename in a DSpace item."""

    root = api_root.rstrip("/") + "/"
    item = _resolve_item(root, handle, load_json)

    bundles_url = urljoin(root, _link(item, "bundles"))
    bundles = _resources(load_json(bundles_url + "?size=100"), "bundles")
    original = next((bundle for bundle in bundles if bundle.get("name") == "ORIGINAL"), None)
    if original is None:
        raise RuntimeError(f"No ORIGINAL bundle found for hdl:{handle}")

    bitstreams_url = urljoin(root, _link(original, "bitstreams"))
    bitstreams = _resources(load_json(bitstreams_url + "?size=100"), "bitstreams")
    bitstream = next((entry for entry in bitstreams if entry.get("name") == filename), None)
    if bitstream is None:
        available = ", ".join(str(entry.get("name")) for entry in bitstreams)
        raise RuntimeError(f"{filename!r} not found; available bitstreams: {available}")

    try:
        content_url = urljoin(root, _link(bitstream, "content"))
    except RuntimeError:
        content_url = urljoin(root, _link(bitstream, "self").rstrip("/") + "/content")

    parsed = urlparse(content_url)
    if parsed.scheme != "https" or parsed.hostname != "www.research-collection.ethz.ch":
        raise RuntimeError(f"Refusing unexpected DSpace content URL: {content_url}")
    return content_url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle", default=DEFAULT_HANDLE)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    args = parser.parse_args()
    try:
        print(resolve_download_url(handle=args.handle, filename=args.filename))
    except Exception as exc:  # noqa: BLE001 - CLI must report resolver failures cleanly.
        print(f"Official EuRoC URL resolution failed: {exc}", file=sys.stderr)
        if args.handle != DEFAULT_HANDLE or args.filename != DEFAULT_FILENAME:
            return 1
        print("Using the checksum-verified MH_01_easy research mirror.", file=sys.stderr)
        print(FALLBACK_URL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
