"""Download the NASA C-MAPSS dataset into data/CMAPSSData.

Source: NASA Prognostics Center of Excellence data repository
(Saxena & Goebel, "Turbofan Engine Degradation Simulation Data Set", 2008).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
DEST = Path(__file__).resolve().parents[1] / "data" / "CMAPSSData"


def main() -> None:
    if (DEST / "train_FD001.txt").exists():
        print(f"Already present: {DEST}")
        return
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as r:
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    DEST.mkdir(parents=True, exist_ok=True)
    for name in inner.namelist():
        if name.endswith(".txt"):
            (DEST / Path(name).name).write_bytes(inner.read(name))
    print(f"Extracted {len(list(DEST.glob('*.txt')))} files to {DEST}")


if __name__ == "__main__":
    main()
