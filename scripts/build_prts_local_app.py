"""Build a native macOS entry point for this workspace's tested desktop runtime.

The application sits beside the 神经网络 project folder and uses its Python environment.
"""
from pathlib import Path
import argparse
import plistlib
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent / 'Arknights Zero.app'
SOURCE = r'''
#include <mach-o/dyld.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char executable[PATH_MAX], resolved[PATH_MAX], root[PATH_MAX], base[PATH_MAX];
    unsigned int size = sizeof executable;
    if (_NSGetExecutablePath(executable, &size) || !realpath(executable, resolved)) return 1;
    strncpy(root, resolved, sizeof root);
    for (int i = 0; i < 4; ++i) {
        char *slash = strrchr(root, '/');
        if (!slash) return 1;
        *slash = '\0';
    }
    snprintf(base, sizeof base, "%s", root);
    snprintf(root, sizeof root, "%s/神经网络", base);
    char python[PATH_MAX], script[PATH_MAX];
    snprintf(python, sizeof python, "%s/.venv/bin/python", root);
    snprintf(script, sizeof script, "%s/scripts/run_desktop.py", root);
    if (access(python, X_OK) || access(script, R_OK)) {
        fprintf(stderr, "Keep this app beside the 神经网络 project folder.\n");
        return 1;
    }
    char **args = calloc((size_t)argc + 2, sizeof(char *));
    if (!args) return 1;
    args[0] = python;
    args[1] = script;
    for (int i = 1; i < argc; ++i) args[i + 1] = argv[i];
    setenv("ARKNIGHTS_ZERO_WORKSPACE", root, 1);
    if (chdir(root)) return 1;
    execv(python, args);
    perror("Unable to launch Arknights Zero");
    return 1;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=APP)
    args = parser.parse_args()
    app = args.output.resolve()
    if app.parent != ROOT.parent or app.suffix != '.app':
        parser.error('The local launcher must be an .app beside the project folder.')
    executable = app / 'Contents/MacOS/ArknightsZeroPRTS'
    resources = app / 'Contents/Resources'
    executable.parent.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='arknights-prts-') as directory:
        source = Path(directory) / 'launcher.c'
        source.write_text(SOURCE)
        subprocess.run(['clang', '-arch', 'arm64', '-O2', str(source), '-o', str(executable)], check=True)
    icons = ROOT.parent / 'Arknights Zero.app/Contents/Resources/app.icns'
    if not icons.exists():
        icons = ROOT.parent / 'Arknights Zero.app/Contents/Resources/pyside_icon.icns'
    if icons.exists() and icons.resolve() != (resources / 'app.icns').resolve():
        shutil.copy2(icons, resources / 'app.icns')
    with (app / 'Contents/Info.plist').open('wb') as stream:
        plistlib.dump(dict(CFBundleName='Arknights Zero', CFBundleDisplayName='Arknights Zero', CFBundleIdentifier=('local.arknightszero.prts' if 'PRTS' in app.stem else 'local.arknightszero.desktop'), CFBundleExecutable=executable.name, CFBundlePackageType='APPL', CFBundleShortVersionString='0.6.0', CFBundleVersion='6', CFBundleIconFile='app.icns', NSHighResolutionCapable=True), stream)
    shutil.copy2(ROOT / 'data/real/PRTS_NOTICE.txt', resources / 'PRTS_NOTICE.txt')
    subprocess.run(['codesign', '--force', '--sign', '-', str(app)], check=True)
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)
    print(app)


if __name__ == '__main__':
    main()
