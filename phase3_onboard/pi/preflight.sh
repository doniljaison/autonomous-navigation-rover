#!/usr/bin/env bash
# Pi 4B platform snapshot — run BEFORE changing anything (to see which G4
# failure was real), after setup_pi4.sh + reboot, and whenever something breaks.
# Output: preflight_<timestamp>.txt in the current directory.
out="preflight_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "### $(date -Is)  $(tr -d '\0' </proc/device-tree/model 2>/dev/null)"
  echo "### uname"; uname -a
  echo "### os"; grep PRETTY /etc/os-release
  echo "### python"; python3 --version
  echo "### throttled (want 0x0)"; vcgencmd get_throttled 2>&1
  echo "### temp / clocks"; vcgencmd measure_temp 2>&1; vcgencmd measure_clock arm 2>&1; vcgencmd measure_volts core 2>&1
  echo "### config.txt (uart / bt / usb)"; grep -nE "enable_uart|disable-bt|miniuart|usb_max|arm_boost|^\[" /boot/firmware/config.txt /boot/config.txt 2>/dev/null
  echo "### cmdline"; cat /boot/firmware/cmdline.txt 2>/dev/null || cat /boot/cmdline.txt
  echo "### serial devices (want /dev/serial0 -> ttyAMA0, /dev/ydlidar -> ttyUSB*)"
  ls -l /dev/serial* /dev/ttyAMA* /dev/ttyS* /dev/ttyUSB* /dev/ttyACM* /dev/ydlidar 2>&1
  echo "### lsusb"; lsusb 2>&1
  echo "### dmesg (usb / tty / voltage)"; sudo dmesg 2>/dev/null | grep -iE "usb|tty|cp210|ch34|brltty|voltage|throttl" | tail -n 50
  echo "### brltty"; dpkg -l brltty 2>/dev/null | tail -n 1 || echo "not installed"
  echo "### groups"; id
  echo "### bluetooth service"; systemctl is-enabled hciuart 2>&1
  echo "### wifi power save"; iw dev wlan0 get power_save 2>&1
  echo "### numba"; "${HOME}/rover-venv/bin/python" -c "import numba, numpy; print('numba', numba.__version__, 'numpy', numpy.__version__)" 2>&1
  echo "### ydlidar binding"; "${HOME}/rover-venv/bin/python" -c "import ydlidar; print('ok')" 2>&1
  echo "### memory"; free -m
} | tee "$out"
echo; echo "saved $out"
