#!/usr/bin/env python3
"""Finish local ad-hoc signing after Nuitka's oversized codesign invocation.

Nuitka can include thousands of PyTorch headers in a single codesign command,
exceeding macOS ARG_MAX. Sign nested objects and header resources in bounded
batches, then the executable and bundle.
No dependency or source data is removed; no Developer ID account is required.
"""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--report", type=Path, default=Path("research/desktop/signing.json"))
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    magic = {b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
             b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"}
    files = []
    for path in bundle.rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                if stream.read(4) in magic:
                    files.append(path)
    results = []
    # macOS treats these files nested beneath Contents/MacOS as sealed objects.
    # Sign them as Nuitka intended, using bounded argument lists.
    headers = [p for p in (bundle / "Contents/MacOS/torch/include").rglob("*") if p.is_file() and not p.is_symlink()]
    main_executable = bundle / "Contents/MacOS/run_desktop"
    objects = sorted(set(files + headers) - {main_executable}, key=lambda x: -len(x.parts))
    for offset in range(0, len(objects), 64):
        batch = objects[offset:offset+64]
        result = subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", *map(str,batch)],
                                capture_output=True, text=True)
        results.append({"paths": [str(path.relative_to(bundle)) for path in batch], "returncode": result.returncode,
                        "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(result.stderr)
    final = subprocess.run(["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(bundle)],
                           capture_output=True, text=True)
    verify = subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", "--verbose=2", str(bundle)],
                            capture_output=True, text=True)
    report = {"bundle": str(bundle), "mach_o_files": len(files), "header_files": len(headers), "objects": results,
              "bundle_sign_returncode": final.returncode, "bundle_sign_stderr": final.stderr,
              "verify_returncode": verify.returncode, "verify_stderr": verify.stderr}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if final.returncode or verify.returncode:
        raise RuntimeError(final.stderr + verify.stderr)


if __name__ == "__main__":
    main()
