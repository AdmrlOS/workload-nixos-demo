#!/usr/bin/env bash
# Docker Desktop/ARM64 packaging and service checks. Does not attest Jetson GPU.
set -euo pipefail
image="${IMAGE:-alexturner/workload-nixos-demo:2026-09-21-nixos26.05-orin-r1}"
name="admiral-nixos-test-$$"
scratch=$(mktemp -d)
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; rm -rf "$scratch"; }
trap cleanup EXIT
docker run -d --name "$name" --platform linux/arm64 \
  --privileged --cgroupns=private --tmpfs /run --tmpfs /tmp \
  -p 127.0.0.1::22 -p 127.0.0.1::8080 "$image" >/dev/null
in_guest() { docker exec "$name" /run/current-system/sw/bin/bash -e -lc "$1"; }
ready=0
for _ in $(seq 1 60); do
  if in_guest 'systemctl is-active --quiet sshd robotics-demo' 2>/dev/null; then ready=1; break; fi
  sleep 1
done
if [ "$ready" != 1 ]; then docker logs "$name"; exit 1; fi
in_guest 'cat /etc/os-release; nix --version; id demo; systemctl --failed --no-pager; test "$(systemctl --failed --no-legend | wc -l)" = 0'
in_guest 'nix eval --offline nixpkgs#lib.version --raw; nix shell --offline nixpkgs#hello -c hello'
in_guest 'test "$(id -u demo)" = 1000; id -G demo | tr " " "\n" | grep -qx 28'
in_guest 'test -z "$(find /nix/store -name "libcuda.so*" -print -quit)"'
in_guest 'sshd -T | grep -iq "passwordauthentication no"; sshd -T | grep -iq "kbdinteractiveauthentication no"'
web_port=$(docker port "$name" 8080/tcp | awk -F: '{print $NF}')
ssh_port=$(docker port "$name" 22/tcp | awk -F: '{print $NF}')
curl -fsS "http://127.0.0.1:$web_port/" > "$scratch/index.html"
grep -q 'Warehouse patrol' "$scratch/index.html"
curl -fsS "http://127.0.0.1:$web_port/api/status" > "$scratch/status.json"
python3 - "$scratch/status.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert 'NixOS' in s['os'], s
assert s['architecture'] == 'aarch64', s
assert s['uid'] == 1000 and 28 in s['groups'], s
assert s['simulation'] is True, s
# Docker Desktop has no NVIDIA driver: failure must be explicit, never fallback.
assert s['gpu']['result'] == 'FAIL', s
print('PASS live API, non-root identity, honest no-GPU status')
PY
# Temporary key is injected into this disposable container only.
ssh-keygen -q -t ed25519 -N '' -f "$scratch/id"
docker exec -i "$name" /run/current-system/sw/bin/bash -e -lc \
  'mkdir -p /root/.ssh /home/demo/.ssh; cat > /root/.ssh/authorized_keys; cp /root/.ssh/authorized_keys /home/demo/.ssh/authorized_keys; chmod 700 /root/.ssh /home/demo/.ssh; chmod 600 /root/.ssh/authorized_keys /home/demo/.ssh/authorized_keys; chown -R demo:users /home/demo/.ssh' < "$scratch/id.pub"
host_key=$(in_guest 'cat /etc/ssh/ssh_host_ed25519_key.pub')
printf '[127.0.0.1]:%s %s\n' "$ssh_port" "$host_key" > "$scratch/known_hosts"
for user in root demo; do
  ssh -F /dev/null -i "$scratch/id" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o UserKnownHostsFile="$scratch/known_hosts" -o StrictHostKeyChecking=yes \
    -p "$ssh_port" "$user@127.0.0.1" 'id; nix --version'
done
set +e
in_guest 'runuser -u demo -- timeout 30 /run/current-system/sw/bin/demo-gpu'
gpu_exit=$?
set -e
test "$gpu_exit" = 1
in_guest 'systemctl restart robotics-demo; systemctl is-active robotics-demo'
in_guest 'mkdir -p /run/systemd/system/robotics-demo.service.d; printf "[Service]\nExecStopPost=/run/current-system/sw/bin/touch /var/lib/admiral-demo/shutdown.ok\n" > /run/systemd/system/robotics-demo.service.d/shutdown-test.conf; systemctl daemon-reload'
docker stop -t 15 "$name" >/dev/null
# Linux PID-namespace halt reports SIGINT to the parent (128 + 2 = 130).
# Require the service's stop hook below, and reject timeout/SIGKILL (137).
shutdown_exit=$(docker inspect -f '{{.State.ExitCode}}' "$name")
case "$shutdown_exit" in 0|130) ;; *) echo "Unexpected shutdown: $shutdown_exit"; exit 1;; esac
docker start "$name" >/dev/null
for _ in $(seq 1 60); do
  if in_guest 'systemctl is-active --quiet sshd robotics-demo' 2>/dev/null; then break; fi
  sleep 1
done
in_guest 'systemctl is-active --quiet sshd robotics-demo'
in_guest 'test -f /var/lib/admiral-demo/shutdown.ok'
test "$host_key" = "$(in_guest 'cat /etc/ssh/ssh_host_ed25519_key.pub')"
echo 'PASS ARM64 NixOS boot, offline Nix, dashboard, key-only SSH, GID 28, failure reporting, service/container restart and graceful shutdown'
