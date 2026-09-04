#!/usr/bin/env python3
"""Report platforms whose published extension package has fallen behind.

The extension factory builds SlicerSOFA for Linux, macOS and Windows
independently, and a platform whose build breaks simply stops publishing while
the others carry on.  That is easy to miss: after the SOFA v26.06 update the
Windows package stayed at a two-week-old revision (r34908, 2026-08-22) while
Linux and macOS kept shipping daily, and nothing surfaced it.

This script compares the newest published package of each platform against the
newest across all of them, and fails when one lags by more than a threshold.
It needs no credentials -- the package server's API is public.

Usage:
    python3 Utilities/CheckPackageFreshness.py [--max-lag-days 3] [--json]

Exit status is 0 when every platform is fresh, 1 when at least one lags, and 2
when the package server could not be reached.
"""

import argparse
import datetime
import json
import sys
import urllib.error
import urllib.request

SLICER_APP_ID = "5f4474d0e1d8c75dfc705482"
API = "https://slicer-packages.kitware.com/api/v1"
EXTENSION_NAME = "SlicerSOFA"
PLATFORMS = ("linux", "macosx", "win")


def newest_package(platform):
    """Return the newest published package for one platform, or None."""
    url = (f"{API}/app/{SLICER_APP_ID}/extension"
           f"?baseName={EXTENSION_NAME}&os={platform}"
           f"&limit=1&sort=created&sortdir=-1")
    with urllib.request.urlopen(url, timeout=30) as response:
        entries = json.load(response)
    if not entries:
        return None
    entry = entries[0]
    return {
        "platform": platform,
        "created": entry["created"][:10],
        "app_revision": int(entry["meta"]["app_revision"]),
        "revision": entry["meta"]["revision"][:9],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-lag-days", type=int, default=3,
                        help="how many days a platform may lag (default: 3)")
    parser.add_argument("--json", action="store_true",
                        help="emit the report as JSON")
    args = parser.parse_args()

    try:
        packages = [newest_package(platform) for platform in PLATFORMS]
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"could not reach the package server: {error}", file=sys.stderr)
        return 2

    missing = [platform for platform, package in zip(PLATFORMS, packages)
               if package is None]
    published = [package for package in packages if package is not None]
    if not published:
        print("no published packages found at all", file=sys.stderr)
        return 2

    newest_date = max(
        datetime.date.fromisoformat(package["created"]) for package in published)
    newest_revision = max(package["app_revision"] for package in published)

    stale = []
    for package in published:
        lag = (newest_date - datetime.date.fromisoformat(package["created"])).days
        package["lag_days"] = lag
        package["revisions_behind"] = newest_revision - package["app_revision"]
        if lag > args.max_lag_days:
            stale.append(package)

    if args.json:
        print(json.dumps({"packages": published, "missing": missing,
                          "stale": [p["platform"] for p in stale]}, indent=2))
    else:
        print(f"newest published package: {newest_date} (Slicer r{newest_revision})")
        for package in published:
            marker = "STALE" if package in stale else "ok"
            print(f"  {package['platform']:<8} {package['created']}  "
                  f"slicer_r{package['app_revision']}  {package['revision']}  "
                  f"lag {package['lag_days']}d  [{marker}]")
        for platform in missing:
            print(f"  {platform:<8} no published package [STALE]")

    if stale or missing:
        names = [package["platform"] for package in stale] + missing
        print(f"\nbehind by more than {args.max_lag_days} days: {', '.join(names)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
