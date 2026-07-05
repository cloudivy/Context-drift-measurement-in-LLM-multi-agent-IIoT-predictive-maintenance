#!/usr/bin/env python3
"""
get_data.py — download CMAPSS FD001 and FD003 training files into data/.
Uses a public GitHub mirror of the NASA CMAPSS dataset.

    python scripts/get_data.py
"""
import os, urllib.request

MIRROR = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData"
FILES = ["train_FD001.txt", "train_FD003.txt"]

os.makedirs("data", exist_ok=True)
for f in FILES:
    dst = os.path.join("data", f)
    if os.path.exists(dst):
        print(f"exists: {dst}"); continue
    url = f"{MIRROR}/{f}"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"  -> {dst}")
print("\nDone. If the mirror is unavailable, download CMAPSS from")
print("https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data")
print("and place train_FD001.txt and train_FD003.txt in data/.")
