#!/usr/bin/env bash
# Install LoopForge for the current Linux user. No sudo or model download.
set -euo pipefail
umask 022

repo="PLASMA-FR/LoopForge"
ref="${LOOPFORGE_REF:-main}"
prefix="${LOOPFORGE_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/loopforge}"
bin_dir="${LOOPFORGE_BIN_DIR:-$HOME/.local/bin}"
python="${LOOPFORGE_PYTHON:-python3}"
source_dir=""
archive_sha=""
temp_dir=""
release_dir=""
activated=0

usage() {
    cat <<'HELP'
Usage: bash install.sh [options]

  --ref REF          Git branch, tag, or commit (default: main)
  --prefix DIR       Installation directory (default: ~/.local/share/loopforge)
  --bin-dir DIR      CLI link directory (default: ~/.local/bin)
  --python PATH      Python 3.11+ executable (default: python3)
  --source DIR       Install an existing source checkout instead of downloading
  --sha256 HEX       Verify downloaded archive bytes against this SHA256
  -h, --help         Show this help

Environment: LOOPFORGE_REF, LOOPFORGE_INSTALL_DIR, LOOPFORGE_BIN_DIR,
LOOPFORGE_PYTHON, XDG_DATA_HOME, XDG_CONFIG_HOME.

Creates a fresh virtual environment for each install, then atomically switches
the active version. Previous versions and existing config files are preserved.
Dependencies are downloaded by pip. Model weights/backends are installed separately.
HELP
}

fail() { printf 'LoopForge installer: %s\n' "$*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref) need_value "$@"; ref="$2"; shift 2 ;;
        --prefix) need_value "$@"; prefix="$2"; shift 2 ;;
        --bin-dir) need_value "$@"; bin_dir="$2"; shift 2 ;;
        --python) need_value "$@"; python="$2"; shift 2 ;;
        --source) need_value "$@"; source_dir="$2"; shift 2 ;;
        --sha256) need_value "$@"; archive_sha="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1 (see --help)" ;;
    esac
done

[[ "$(uname -s)" == Linux ]] || fail "This installer supports Linux."
command -v "$python" >/dev/null 2>&1 || fail "Install Python 3.11+ first, or pass --python /path/to/python."
"$python" -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || fail "Python 3.11+ with venv support is required."
[[ -z "$archive_sha" || "$archive_sha" =~ ^[[:xdigit:]]{64}$ ]] || fail "--sha256 requires exactly 64 hexadecimal characters."
[[ -z "$source_dir" || -z "$archive_sha" ]] || fail "--sha256 applies only to downloaded archives."

# Resolve relative paths once; quoting preserves spaces in home/install paths.
prefix="$("$python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$prefix")"
bin_dir="$("$python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$bin_dir")"
[[ "$prefix" != / && "$bin_dir" != / ]] || fail "Use a dedicated installation/bin directory."
mkdir -p -- "$prefix/releases" "$bin_dir"

# Refuse to replace an unrelated command or a real directory at current.
"$python" - "$prefix" "$bin_dir" <<'PY'
import os, pathlib, sys
prefix, bin_dir = map(pathlib.Path, sys.argv[1:])
current, launcher = prefix / 'current', bin_dir / 'loopforge'
expected = str(current / 'venv/bin/loopforge')
if current.exists() and not current.is_symlink():
    sys.exit('Refusing to replace non-symlink installation/current')
if os.path.lexists(launcher) and (not launcher.is_symlink() or os.readlink(launcher) != expected):
    sys.exit(f'Refusing to overwrite unrelated command: {launcher}')
PY

cleanup() {
    status=$?
    trap - EXIT
    [[ -z "$temp_dir" ]] || rm -rf -- "$temp_dir"
    if [[ "$activated" == 0 && -n "$release_dir" && ! "$prefix/current" -ef "$release_dir" ]]; then
        rm -rf -- "$release_dir"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/loopforge-download.XXXXXXXX")"

if [[ -n "$source_dir" ]]; then
    source_dir="$("$python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$source_dir")"
else
    command -v curl >/dev/null 2>&1 || fail "Install curl first."
    encoded_ref="$("$python" -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$ref")"
    printf 'Downloading %s at %s...\n' "$repo" "$ref"
    curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
        --connect-timeout 15 --max-time 180 --retry 3 \
        --max-filesize 67108864 \
        "https://codeload.github.com/$repo/tar.gz/$encoded_ref" -o "$temp_dir/source.tar.gz"
    source_dir="$("$python" - "$temp_dir/source.tar.gz" "$temp_dir/source" "$archive_sha" <<'PY'
import hashlib, pathlib, shutil, sys, tarfile
archive, target = map(pathlib.Path, sys.argv[1:3])
expected = sys.argv[3].lower()
if archive.stat().st_size > 64 * 1024 * 1024:
    sys.exit('Archive exceeds 64 MiB download limit')
if expected and hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    sys.exit('Archive SHA256 mismatch; installation unchanged')
target.mkdir()
total = count = 0
with tarfile.open(archive, 'r:gz') as tar:
    for item in tar:
        count += 1
        total += item.size
        if count > 10000 or total > 128 * 1024 * 1024:
            sys.exit('Archive exceeds extraction limits')
        name = pathlib.PurePosixPath(item.name)
        if name.is_absolute() or '..' in name.parts or not (item.isfile() or item.isdir()):
            sys.exit('Unsafe archive member: paths, symlinks and special files are restricted')
        dest = target.joinpath(*name.parts)
        if not dest.resolve().is_relative_to(target.resolve()):
            sys.exit('Archive path escapes extraction directory')
        if item.isdir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as src, dest.open('xb') as out:
                shutil.copyfileobj(src, out)
roots = list(target.iterdir())
if len(roots) != 1 or not (roots[0] / 'pyproject.toml').is_file():
    sys.exit('Archive does not contain a single LoopForge project')
print(roots[0])
PY
)"
fi

[[ -f "$source_dir/pyproject.toml" && -f "$source_dir/loopforge/cli.py" ]] || fail "Source directory is not a LoopForge checkout."
release_dir="$(mktemp -d "$prefix/releases/install.XXXXXXXX")"
printf 'Creating isolated environment with %s...\n' "$python"
"$python" -m venv "$release_dir/venv" || fail "Could not create a venv. On Debian/Ubuntu, install python3-venv."
"$release_dir/venv/bin/python" -m pip install --disable-pip-version-check --no-input "$source_dir"
"$release_dir/venv/bin/loopforge" --help >/dev/null

# Stage both links first; current is the final atomic activation point. Venvs
# are built at their permanent path because entry-point shebangs are absolute.
"$python" - "$prefix" "$bin_dir" "$release_dir" <<'PY'
import os, pathlib, sys, uuid
prefix, bin_dir, release = map(pathlib.Path, sys.argv[1:])
current, launcher = prefix / 'current', bin_dir / 'loopforge'
expected = str(current / 'venv/bin/loopforge')
staged_current = prefix / ('.current-' + uuid.uuid4().hex)
staged_launcher = bin_dir / ('.loopforge-' + uuid.uuid4().hex)
created_launcher = False
try:
    staged_current.symlink_to(release, target_is_directory=True)
    if not os.path.lexists(launcher):
        staged_launcher.symlink_to(expected)
        # Hard-link the symlink itself without replacing a concurrently-created command.
        os.link(staged_launcher, launcher, follow_symlinks=False)
        created_launcher = True
    elif not launcher.is_symlink() or os.readlink(launcher) != expected:
        sys.exit(f'Launcher changed during install: {launcher}')
    if current.exists() and not current.is_symlink():
        sys.exit('Active installation changed during install')
    os.replace(staged_current, current)
except BaseException:
    if created_launcher:
        launcher.unlink(missing_ok=True)
    raise
finally:
    staged_current.unlink(missing_ok=True)
    staged_launcher.unlink(missing_ok=True)
PY
activated=1

config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/loopforge"
if [[ -f "$source_dir/examples/loopforge.toml" ]]; then
    # Configuration is optional; installation success does not depend on copying it.
    if mkdir -p -- "$config_dir"; then
        cp --no-clobber -- "$source_dir/examples/loopforge.toml" "$config_dir/config.toml" || true
    fi
fi
printf '\nInstalled LoopForge: %s/loopforge\n' "$bin_dir"
case ":${PATH:-}:" in
    *":$bin_dir:"*) ;;
    *) printf 'Add the CLI directory to PATH in your shell profile:\n  export PATH=%q:"$PATH"\n' "$bin_dir" ;;
esac
printf 'Start a compatible model backend, then run:\n  %q --workspace /path/to/project doctor\n' "$bin_dir/loopforge"
printf 'Chat: %q --workspace /path/to/project --allow-shell chat\n' "$bin_dir/loopforge"
printf 'API:  %q --workspace /path/to/project --allow-shell serve\n' "$bin_dir/loopforge"
printf 'Optional config (preserved on updates): %s/config.toml\n' "$config_dir"
printf 'Use it with: loopforge --config %q --workspace /path/to/project chat\n' "$config_dir/config.toml"
