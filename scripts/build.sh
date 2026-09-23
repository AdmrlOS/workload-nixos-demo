#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
builder=admiral-nixos-builder
builder_image=nixos/nix@sha256:7a007c766426c1877758ddc5cb87a965ac131fc78c582ce0083d922d51ae945c
if ! docker inspect "$builder" >/dev/null 2>&1; then
  docker run -d --name "$builder" --platform linux/arm64 --privileged \
    -v admiral-nixos-store:/nix -v "$PWD:/work" -w /work "$builder_image" sleep infinity
else
  docker start "$builder" >/dev/null
fi
docker exec "$builder" git config --global --add safe.directory '*' || true
docker exec "$builder" nix --extra-experimental-features 'nix-command flakes' \
  --option filter-syscalls false --option sandbox false \
  build /work#image --out-link /tmp/admiral-nixos-image -L
mkdir -p build
docker cp -L "$builder:/tmp/admiral-nixos-image" build/image.tar.gz
docker load -i build/image.tar.gz
image="${IMAGE_REPOSITORY:-ghcr.io/admrlos/workload-nixos-demo}:${IMAGE_TAG:-2026-09-21-nixos26.05-orin-r1}"
docker tag workload-nixos-demo:2026-09-21-nixos26.05-orin-r1 "$image"
docker tag workload-nixos-demo:2026-09-21-nixos26.05-orin-r1 "${IMAGE_REPOSITORY:-ghcr.io/admrlos/workload-nixos-demo}:latest"
echo "Built $image (and ${IMAGE_REPOSITORY:-ghcr.io/admrlos/workload-nixos-demo}:latest)"
