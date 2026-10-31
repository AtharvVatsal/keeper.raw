"""
Download ExifTool for this operating system into vendor/exiftool/, so the
installer can ship it (end users don't have ExifTool installed).

    python scripts/fetch_exiftool.py

The version is pinned and every download is checked against the SHA-256
published at https://exiftool.org/checksums.txt. Uses only Python's
standard library.

Result (vendor/ is not committed to git):
  Windows        vendor/exiftool/exiftool.exe + exiftool_files/  (bundled Perl)
  macOS / Linux  vendor/exiftool/exiftool + lib/                 (run with the
                                                                  system's perl)

Licence: ExifTool is Copyright Phil Harvey, "free software; you can
redistribute it and/or modify it under the same terms as Perl itself (either
the Perl Artistic License or GPL)". The Windows package also contains
Strawberry Perl, whose licences are in exiftool_files/.
"""

import hashlib
import io
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

VERSION = "13.60"
WINDOWS_ZIP = f"exiftool-{VERSION}_64.zip"
WINDOWS_SHA256 = "548b1f4e0b713cd2a26824b34080c35db4e2d2edd166468eb38b1d55ada90c2e"
PERL_TARBALL = f"Image-ExifTool-{VERSION}.tar.gz"
PERL_SHA256 = "c8e3c42576f46d862e4cd9c192989776e7363931e890c95056bcd6293e83ea36"

# exiftool.org links its downloads to SourceForge, which keeps every version.
URL = "https://downloads.sourceforge.net/project/exiftool/{}"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "vendor", "exiftool")


def download(name, sha256):
    print(f"Downloading {name} ...")
    req = urllib.request.Request(URL.format(name), headers={"User-Agent": "keeper-raw-build"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        sys.exit(f"Checksum mismatch for {name}: expected {sha256}, got {got}")
    print(f"  {len(data) / 1e6:.1f} MB, checksum OK")
    return data


def fetch_windows():
    top = f"exiftool-{VERSION}_64/"
    with zipfile.ZipFile(io.BytesIO(download(WINDOWS_ZIP, WINDOWS_SHA256))) as z:
        for info in z.infolist():
            if not info.filename.startswith(top) or info.is_dir():
                continue
            rel = info.filename[len(top):]
            # "(-k)" makes the exe wait for a key press; renaming removes that.
            if rel == "exiftool(-k).exe":
                rel = "exiftool.exe"
            write(rel, z.read(info))


def fetch_perl():
    top = f"Image-ExifTool-{VERSION}/"
    with tarfile.open(fileobj=io.BytesIO(download(PERL_TARBALL, PERL_SHA256))) as t:
        for member in t.getmembers():
            if not member.isfile() or not member.name.startswith(top):
                continue
            rel = member.name[len(top):]
            # Only what's needed to run: the script, its modules, the licence.
            if rel in ("exiftool", "README") or rel.startswith("lib/"):
                write(rel, t.extractfile(member).read())
    os.chmod(os.path.join(OUT, "exiftool"), 0o755)


def write(rel, data):
    path = os.path.join(OUT, *rel.split("/"))
    # Never write outside the output folder, whatever the archive says.
    if not os.path.abspath(path).startswith(os.path.abspath(OUT) + os.sep):
        sys.exit(f"Refusing to write outside {OUT}: {rel}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    if sys.platform.startswith("win"):
        fetch_windows()
    else:
        fetch_perl()
    files = sum(len(f) for _, _, f in os.walk(OUT))
    print(f"ExifTool {VERSION} ready in {OUT} ({files} files)")


if __name__ == "__main__":
    main()
