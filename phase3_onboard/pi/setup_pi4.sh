#!/usr/bin/env bash
# =============================================================================
#  Raspberry Pi 4 Model B bring-up for the Phase 3 rover stack
#  OS: Raspberry Pi OS Lite 64-bit (Bookworm).  Run as the normal user (uses sudo).
#
#    bash phase3_onboard/pi/setup_pi4.sh            # everything
#    bash phase3_onboard/pi/setup_pi4.sh --no-sdk   # skip the YDLidar-SDK build
#
#  What it does (each step is idempotent):
#    0. saves a preflight snapshot (lsusb, dmesg, serial devices, throttling)
#    1. apt packages for building the YDLidar-SDK and Python wheels
#    2. removes brltty if present (it grabs CP210x/CH34x USB-serial adapters)
#    3. header UART: enable_uart=1 + dtoverlay=disable-bt  -> /dev/serial0 = ttyAMA0 (PL011)
#       and removes the serial console from cmdline.txt, disables hciuart
#    4. adds you to dialout; installs the /dev/ydlidar udev rule
#    5. venv ~/rover-venv with numpy, scipy, numba, pyserial
#    6. builds + installs YDLidar-SDK and its Python binding into the venv
#    7. checks numba JIT works and pre-compiles the SLAM kernels
#    8. disables WiFi power saving (lower web-UI latency)
#  A reboot is required afterwards for the UART / group changes.
#
#  Pi 4B notes (differences from the Pi 5 plan):
#   * No USB-PD and no usb_max_current_enable on the Pi 4: the four USB ports
#     share a fixed 1.2 A budget. The G4 (budget 1 A at spin-up) fits only if
#     it is the sole USB load. Do not add a USB hub fed from the Pi.
#   * Power: 5.1 V >= 3 A. Under-voltage trips at ~4.63 V. See docs/pi4_power.md.
#   * Header UART: by default the PL011 belongs to Bluetooth and serial0 is the
#     mini-UART (ttyS0) whose baud follows the core clock.  disable-bt fixes it.
#   * Cooling: the Pi 4B throttles at 80 C under sustained numba load. Fit a
#     heatsink + fan before Gate G3.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${HOME}/rover-venv"
DO_SDK=1
[[ "${1:-}" == "--no-sdk" ]] && DO_SDK=0

BOOT=/boot/firmware
[[ -f $BOOT/config.txt ]] || BOOT=/boot
CONFIG=$BOOT/config.txt
CMDLINE=$BOOT/cmdline.txt

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown)
say "Board: $model"
if [[ "$model" != *"Raspberry Pi 4"* ]]; then
  echo "WARNING: this script is written for the Raspberry Pi 4 Model B."
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: 64-bit OS required (numba wheels are aarch64 only). uname -m = $(uname -m)"; exit 1
fi

say "0. preflight snapshot"
bash "$REPO/phase3_onboard/pi/preflight.sh" || true

say "1. apt packages"
sudo apt-get update
sudo apt-get install -y git cmake swig build-essential pkg-config python3-venv python3-dev \
     libopenblas-dev gfortran iw

say "2. brltty"
if dpkg -l brltty 2>/dev/null | grep -q '^ii'; then sudo apt-get remove -y brltty; else echo "not installed"; fi

say "3. header UART (PL011 on GPIO14/15)"
add_cfg() { grep -qxF "$1" "$CONFIG" || echo "$1" | sudo tee -a "$CONFIG" >/dev/null; }
# our lines must land in an [all] section (a trailing [cm4]/[pi4] block would scope them)
last_section=$(grep -E '^\[' "$CONFIG" | tail -n 1 || true)
if [[ -n "$last_section" && "$last_section" != "[all]" ]]; then echo "[all]" | sudo tee -a "$CONFIG" >/dev/null; fi
add_cfg "enable_uart=1"
add_cfg "dtoverlay=disable-bt"
sudo sed -i -E 's/console=(serial0|ttyAMA0|ttyS0),[0-9]+ ?//g' "$CMDLINE"
sudo systemctl disable --now hciuart.service 2>/dev/null || true
sudo systemctl disable --now serial-getty@ttyAMA0.service 2>/dev/null || true
sudo systemctl disable --now serial-getty@ttyS0.service 2>/dev/null || true
echo "config.txt tail:"; tail -n 5 "$CONFIG"; echo "cmdline: $(cat $CMDLINE)"

say "4. groups + udev"
sudo usermod -aG dialout,video "$USER"
sudo install -m 0644 "$REPO/phase3_onboard/pi/99-ydlidar.rules" /etc/udev/rules.d/99-ydlidar.rules
sudo udevadm control --reload-rules && sudo udevadm trigger || true

say "5. Python venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel setuptools
"$VENV/bin/pip" install -r "$REPO/phase3_onboard/requirements-pi.txt"

if [[ $DO_SDK == 1 ]]; then
  say "6. YDLidar-SDK"
  SDK="${HOME}/YDLidar-SDK"
  [[ -d "$SDK" ]] || git clone --depth 1 https://github.com/YDLIDAR/YDLidar-SDK.git "$SDK"
  mkdir -p "$SDK/build" && cd "$SDK/build"
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j4
  sudo make install
  cd "$SDK"
  "$VENV/bin/pip" install .
  "$VENV/bin/python" -c "import ydlidar; print('ydlidar binding OK')"
  cd "$REPO"
fi

say "7. numba check + SLAM kernel pre-compile (first compile on a Pi 4B takes ~30-60 s)"
cd "$REPO"
"$VENV/bin/python" - <<'EOF'
import time, numba
from phase3_onboard.slam.params import from_config
from phase3_onboard.slam.engine import warmup_jit
t = time.time(); warmup_jit(from_config()); print(f"numba {numba.__version__}: kernels compiled+cached in {time.time()-t:.1f} s")
EOF

say "8. WiFi power saving off (also persisted via NetworkManager if present)"
sudo iw dev wlan0 set power_save off 2>/dev/null || true
if command -v nmcli >/dev/null; then
  for c in $(nmcli -t -f NAME,TYPE con show | awk -F: '$2=="802-11-wireless"{print $1}'); do
    sudo nmcli con modify "$c" 802-11-wireless.powersave 2 || true
  done
fi

say "Done.  REBOOT NOW (sudo reboot), then run:  bash phase3_onboard/pi/preflight.sh"
echo "Expect: /dev/serial0 -> ttyAMA0, /dev/ydlidar present, get_throttled=0x0"
