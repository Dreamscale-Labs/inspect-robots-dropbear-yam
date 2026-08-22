#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "dropbear-yam setup requires the Linux robot host." >&2
  exit 2
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify this Linux distribution; Debian/Ubuntu is currently supported." >&2
  exit 2
fi
. /etc/os-release
case "${ID:-}:${ID_LIKE:-}" in
  ubuntu:*|debian:*|*:debian*) ;;
  *)
    echo "Automatic prerequisites currently support Debian/Ubuntu only." >&2
    exit 2
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
    echo "Setup cancelled before sudo." >&2
    exit 2
  fi
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${missing_packages[@]}"
fi

uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv_bin="${HOME}/.local/bin/uv"
fi
if [[ ! -x "$uv_bin" ]]; then
  echo "uv installation failed; expected executable at $uv_bin" >&2
  exit 2
fi

"$uv_bin" sync --project "$repo_dir" --locked --extra hardware
"$uv_bin" run --project "$repo_dir" --locked --extra hardware dropbear-yam setup

echo
echo "Setup complete. Next run:"
echo "  $repo_dir/dropbear-yam doctor"
