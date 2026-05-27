#!/bin/bash
# fand installer. Run as root.
#
#   sudo /opt/fand/install.sh
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

ROOT=/opt/fand
cd "$ROOT"

# ---- venv -----------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found in PATH. Install with the project's uv binary first." >&2
    exit 1
fi

if [ ! -d .venv ]; then
    echo "creating venv at $ROOT/.venv"
    uv venv .venv --python python3.12
fi
echo "installing dependencies"
uv pip install --python .venv/bin/python -r requirements.txt

# ---- system directories ---------------------------------------------------

install -d -m 755 /etc/fand
install -d -m 755 /var/lib/fand

# ---- config (only if absent — don't overwrite operator edits) -------------

if [ ! -e /etc/fand/config.yaml ]; then
    install -m 644 etc/config.yaml.example /etc/fand/config.yaml
    echo "wrote /etc/fand/config.yaml"
else
    echo "preserved existing /etc/fand/config.yaml"
fi

# ---- systemd unit ---------------------------------------------------------

install -m 644 systemd/fand.service /etc/systemd/system/fand.service
echo "wrote /etc/systemd/system/fand.service"

# ---- CLI shims ------------------------------------------------------------

cat > /usr/local/bin/fand-ctl <<'EOF'
#!/bin/bash
exec /opt/fand/.venv/bin/python -m fand.cli "$@"
EOF
chmod 755 /usr/local/bin/fand-ctl

cat > /usr/local/bin/fand-calibrate <<'EOF'
#!/bin/bash
exec /opt/fand/.venv/bin/python -m fand.calibrate "$@"
EOF
chmod 755 /usr/local/bin/fand-calibrate

echo "wrote /usr/local/bin/fand-ctl and /usr/local/bin/fand-calibrate"

systemctl daemon-reload
systemctl enable fand.service >/dev/null

cat <<EOF

fand installed (not yet started).

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
  sudo rm -rf /etc/fand /var/lib/fand
  sudo systemctl daemon-reload
EOF
