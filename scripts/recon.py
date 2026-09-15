import zipfile, plistlib, json, struct, collections, pathlib, hashlib, re, argparse

parser = argparse.ArgumentParser(description="Read-only IPA container reconnaissance")
parser.add_argument(
    "ipa",
    nargs="?",
    default="/Users/yihao/Downloads/com.hypergryph.arknights_2.7.71_und3fined.ipa",
)
p = pathlib.Path(parser.parse_args().ipa)
z = zipfile.ZipFile(p)
names = z.namelist()
out = pathlib.Path(__file__).resolve().parents[1] / "research/ipa"
out.mkdir(parents=True, exist_ok=True)
(out / "inventory.txt").write_text(
    "\n".join(f"{i.file_size}\t{i.filename}" for i in z.infolist())
)
info = next(n for n in names if n.count("/") == 2 and n.endswith(".app/Info.plist"))
pl = plistlib.loads(z.read(info))
root = info[:-10]
report = {
    "file": str(p),
    "size": p.stat().st_size,
    "bundle": pl.get("CFBundleIdentifier"),
    "version": pl.get("CFBundleShortVersionString"),
    "build": pl.get("CFBundleVersion"),
    "executable": pl.get("CFBundleExecutable"),
    "extensions": dict(collections.Counter(pathlib.Path(n).suffix for n in names)),
    "binaries": [],
}
for n in names:
    if n == root + pl["CFBundleExecutable"] or n.endswith("/UnityFramework"):
        b = z.read(n)
        magic = b[:4].hex()
        rec = {"path": n, "magic": magic}
        if magic == "cffaedfe":
            h = struct.unpack_from("<8I", b)
            rec["cpu_type"] = hex(h[1])
            pos = 32
            crypt = []
            for _ in range(h[4]):
                cmd, size = struct.unpack_from("<II", b, pos)
                if cmd in (0x21, 0x2C):
                    crypt.append(struct.unpack_from("<III", b, pos + 8))
                pos += size
            rec["encryption_info"] = crypt
        report["binaries"].append(rec)
for n in names:
    if n.endswith(("globalgamemanagers", "global-metadata.dat")):
        with z.open(n) as f:
            b = f.read(256)
        report.setdefault("headers", []).append(
            {
                "path": n,
                "hex": b[:32].hex(),
                "printable": "".join(chr(c) if 32 <= c < 127 else "." for c in b),
            }
        )
with p.open("rb") as stream:
    report["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
metadata_path = next((n for n in names if n.endswith("/global-metadata.dat")), None)
if metadata_path:
    metadata = z.read(metadata_path)
    if metadata[:4] == bytes.fromhex("af1bb1fa"):
        offset, length = struct.unpack_from("<II", metadata, 24)
        if offset + length <= len(metadata):
            pattern = re.compile(
                r"Battle|Enemy|TargetSelector|AttackState|Blocker|Waypoint|SkillManager|LifePoint|Redeploy|Modifier|Projectile"
            )
            strings = [
                v.decode("utf-8", "replace")
                for v in metadata[offset : offset + length].split(b"\0")
            ]
            (out / "metadata_names.txt").write_text(
                "\n".join(v for v in strings if pattern.search(v))
            )
(out / "facts.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
