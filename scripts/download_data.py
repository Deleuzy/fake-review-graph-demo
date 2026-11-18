"""Download the YelpChi graph released with CARE-GNN (Dou et al., CIKM 2020).

The file is not committed. It is fetched from the CARE-GNN GitHub repository and
checked against the SHA-256 recorded below.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip"
ZIP_SHA256 = "3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3"
MAT_SHA256 = "fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(out_dir: str = "data/raw") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zpath, mpath = out / "YelpChi.zip", out / "YelpChi.mat"
    if not zpath.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    if sha256(zpath) != ZIP_SHA256:
        sys.exit("YelpChi.zip checksum mismatch")
    if not mpath.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extract("YelpChi.mat", out)
    if sha256(mpath) != MAT_SHA256:
        sys.exit("YelpChi.mat checksum mismatch")
    print(f"ok: {mpath}")


if __name__ == "__main__":
    main(*sys.argv[1:])
