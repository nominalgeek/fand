#!/bin/bash
# fand installer. Run as root.
#
#   sudo ./install.sh
#
# Builds a wheel from the dev checkout (`uv build`), installs it into a
# root-owned venv at /usr/local/lib/fand/.venv, then writes the systemd unit,
# seeds /etc/fand/config.yaml if absent, and symlinks the CLI shims. The dev
# checkout stays where it is and stays editable — it's not where the daemon
# runs from. To pick up source changes, re-run this script.
#
# Does NOT start the service — operator must:
#   1. sudo fand-calibrate                 (discovers PWM↔fan mapping; ~5 min)
#   2. sudo $EDITOR /etc/fand/zones.yaml   (fill in target_sensors per zone)
#   3. sudo systemctl start fand
set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    echo "install.sh must be run as root" >&2
    exit 1
fi

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME=/usr/local/lib/fand

UV="${UV:-$(command -v uv 2>/dev/null || true)}"
if [ -z "$UV" ] || ! [ -x "$UV" ]; then
    echo "error: uv not found. Install uv on root's PATH, or pass its path" >&2
    echo "explicitly: sudo UV=/path/to/uv $0" >&2
    exit 1
fi

# ---- build wheel from the dev checkout -----------------------------------

BUILD_DIR="$(mktemp -d -t fand-build.XXXXXXXX)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "building wheel from $SRC"
(cd "$SRC" && "$UV" build --wheel --out-dir "$BUILD_DIR")

WHEEL="$(ls "$BUILD_DIR"/fand-*.whl 2>/dev/null | head -n 1)"
if [ -z "$WHEEL" ] || [ ! -f "$WHEEL" ]; then
    echo "error: uv build produced no wheel under $BUILD_DIR" >&2
    exit 1
fi

# ---- runtime venv (root-owned) -------------------------------------------

install -d -m 755 -o root -g root "$RUNTIME"
cd "$RUNTIME"

if [ ! -d .venv ]; then
    echo "creating venv at $RUNTIME/.venv"
    "$UV" venv .venv --python python3.13
fi

# Whole point: the interpreter the daemon executes as root must live in a
# root-owned path. uv may symlink .venv/bin/python to whatever python3.12 it
# finds first — if that resolves to a non-root-owned path (e.g. uv's own
# managed python under ~/.local/share/uv), refuse to install.
PY="$(readlink -f .venv/bin/python)"
PY_OWNER="$(stat -c '%U' "$PY")"
if [ "$PY_OWNER" != "root" ]; then
    echo "error: venv python ($PY) is owned by '$PY_OWNER', not root." >&2
    echo "uv resolved python3.12 to a non-system interpreter. Install a system" >&2
    echo "python3.12 (e.g. 'apt install python3.12'), remove $RUNTIME/.venv, and" >&2
    echo "re-run this script." >&2
    exit 1
fi

echo "installing $(basename "$WHEEL") into $RUNTIME/.venv"
"$UV" pip install --python .venv/bin/python --reinstall "$WHEEL"

# Belt-and-braces: tighten ownership across the whole runtime tree. uv run as
# root should already produce root-owned files; enforcing here means a
# reinstall fixes anything that drifted.
chown -R root:root "$RUNTIME"

# ---- system directories ---------------------------------------------------

install -d -m 755 /etc/fand
install -d -m 755 /var/lib/fand

# ---- config (only if absent — don't overwrite operator edits) -------------

if [ ! -e /etc/fand/config.yaml ]; then
    install -m 644 "$SRC/etc/config.yaml.example" /etc/fand/config.yaml
    echo "wrote /etc/fand/config.yaml"
else
    echo "preserved existing /etc/fand/config.yaml"
fi

# ---- systemd unit ---------------------------------------------------------

install -m 644 "$SRC/systemd/fand.service" /etc/systemd/system/fand.service
echo "wrote /etc/systemd/system/fand.service"

# ---- CLI shims ------------------------------------------------------------
# The wheel's entry_points installed fand-ctl/fand-calibrate/fand-daemon into
# the venv's bin/. Expose the operator-facing ones on PATH. fand-daemon is
# intentionally NOT linked — operators interact with systemctl.

ln -sf /usr/local/lib/fand/.venv/bin/fand-ctl       /usr/local/bin/fand-ctl
ln -sf /usr/local/lib/fand/.venv/bin/fand-calibrate /usr/local/bin/fand-calibrate
echo "linked /usr/local/bin/fand-ctl and /usr/local/bin/fand-calibrate"

systemctl daemon-reload
systemctl enable fand.service >/dev/null

cat <<EOF

fand installed (not yet started).

Wheel built from $SRC and installed into $RUNTIME/.venv (root-owned). The dev
checkout stays where it is and stays editable. To pick up source changes,
re-run:
  sudo $SRC/install.sh

Next steps:
  1) sudo fand-calibrate
       Discovers which PWM controls which fan; writes /etc/fand/zones.yaml.
       Run with the system IDLE (stop ComfyUI / training). Takes ~5 minutes.

  2) sudo \$EDITOR /etc/fand/zones.yaml
       Set 'target_sensors' for each zone — pick the temps you want that fan
       to hold under control. Chip names + labels: \`sensors\` for guidance.

  3) sudo systemctl start fand
       Bring it up. Watch with \`fand-ctl status\` and \`journalctl -u fand -f\`.

To uninstall:
  sudo systemctl stop fand
  sudo systemctl disable fand
  sudo rm /etc/systemd/system/fand.service /usr/local/bin/fand-{ctl,calibrate}
  sudo rm -rf /etc/fand /var/lib/fand /usr/local/lib/fand
  sudo systemctl daemon-reload
EOF
