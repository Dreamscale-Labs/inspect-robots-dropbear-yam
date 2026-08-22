#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

fail() {
  echo "Error: $1" >&2
  echo "Next: $2" >&2
  exit 2
}

if [[ "$(uname -s)" != "Linux" ]]; then
  fail \
    "This setup is not running on Linux." \
    "Run these commands on the Ubuntu computer connected to both YAM arms and all cameras."
fi

if [[ ! -r /etc/os-release ]]; then
  fail \
    "Setup cannot identify this Linux distribution." \
    "Use an Ubuntu or Debian robot computer, or send /etc/os-release details to Dreamscale."
fi
. /etc/os-release
case "${ID:-}:${ID_LIKE:-}" in
  ubuntu:*|debian:*|*:debian*) ;;
  *)
    fail \
      "Automatic setup currently supports Ubuntu and Debian only." \
      "Run this on the Ubuntu/Debian YAM computer or contact Dreamscale before installing manually."
    ;;
esac

packages=(
  build-essential
  can-utils
  cmake
  curl
  git
  libgl1
  libglib2.0-0
  libusb-1.0-0-dev
  ninja-build
  pkg-config
  python3-dev
  v4l-utils
)
missing_packages=()
for package in "${packages[@]}"; do
  if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
    missing_packages+=("$package")
  fi
done

if (( ${#missing_packages[@]} )); then
  echo "The following host prerequisites must be installed with sudo:"
  printf '  %s\n' "${missing_packages[@]}"
  read -r -p "Allow this one apt installation step? [y/N] " approve_sudo
  if [[ "$approve_sudo" != "y" && "$approve_sudo" != "Y" ]]; then
    fail \
      "Setup was cancelled before making any sudo changes." \
      "Rerun ./setup.sh and answer y when you are ready to install the listed packages."
  fi
  if ! sudo apt-get update; then
    fail \
      "Ubuntu could not refresh its package list." \
      "Check the internet connection and sudo access, then rerun ./setup.sh."
  fi
  if ! sudo apt-get install -y --no-install-recommends "${missing_packages[@]}"; then
    fail \
      "Ubuntu could not install the required system packages." \
      "Read the apt error above, fix it, then rerun ./setup.sh."
  fi
fi

uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" ]]; then
  if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
    fail \
      "The uv Python installer could not be downloaded or run." \
      "Check internet access to astral.sh, then rerun ./setup.sh."
  fi
  uv_bin="${HOME}/.local/bin/uv"
fi
if [[ ! -x "$uv_bin" ]]; then
  fail \
    "uv was not found at $uv_bin after installation." \
    "Open a new terminal and rerun ./setup.sh; if it repeats, send this message to Dreamscale."
fi

if ! "$uv_bin" sync --project "$repo_dir" --python 3.12 --locked --extra hardware; then
  fail \
    "The locked Dropbear-YAM Python environment could not be installed." \
    "Check the error above and internet access, then rerun ./setup.sh."
fi
set +e
"$uv_bin" run --project "$repo_dir" --python 3.12 --locked --extra hardware dropbear-yam setup
setup_status=$?
set -e
if (( setup_status != 0 )); then
  exit "$setup_status"
fi

echo
echo "Setup complete. Next run:"
echo "  $repo_dir/dropbear-yam doctor"
