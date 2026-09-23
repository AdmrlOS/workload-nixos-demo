# Call walkthrough and acceptance

## Before the call

Deploy the release digest from PUBLISHED.md through the Admiral portal. Wait for
the dashboard at `http://DEVICE_IP:8080`. Record the cloud deployment revision
and resolved image digest. Confirm SSH as `demo` and the actual GPU test result.
Do not describe an untested image as board-validated.

## Five-minute walkthrough

**1. Show the dashboard.** “This is a complete NixOS userspace built from a locked
flake. Admiral boots the device and deploys it as an OCI artifact.” The on-device
**MiniCPM5-2B** chat interface demonstrates edge AI inference directly on
Jetson Orin with the official Admiral theming, accelerated via the injected CUDA
driver through llama.cpp. CPU fallback is strictly disabled to guarantee genuine GPU execution.

**2. Show real GPU execution.** The independent GPU panel runs a coordinate
transform on 65,539 synthetic points and compares every output with a CPU
reference, while the on-device LLM executes real tensor operations on the Orin GPU.
Describe the coordinate transform as a compatibility smoke test, not a performance result.

**3. SSH with the existing key.**

```sh
ssh demo@DEVICE_IP
cat /etc/os-release
id
demo-status
demo-chat "Explain how Admiral manages the Jetson GPU driver"
timeout 30 demo-gpu
nix --version
cat /etc/admiral-demo/flake.lock
nix shell --offline nixpkgs#hello -c hello
systemctl status robotics-demo minicpm --no-pager
cat /etc/admiral-demo/source/nix/configuration.nix
```

**4. Show ordinary service management as root.**

```sh
ssh root@DEVICE_IP
journalctl -u robotics-demo --no-pager -n 30
systemctl restart robotics-demo
systemctl is-active robotics-demo
```

**5. Close with the customer integration.** “We bring your Nix packages and
userspace service modules into this image, pin and version the result, and deploy
through Admiral. The next PoC tests your actual robotics application and devices.”

## Jetson acceptance record

Local ARM64 container tests are useful but cannot prove Admiral deployment or
GPU execution. Capture the following from the portal-deployed image:

- Release image digest, cloud deployment revision, host init/BSP versions,
  manifest/profile, boot ID and kernel release.
- `/api/status` and `timeout 30 demo-gpu` JSON as `demo`, then as root. Require
  PASS, a real injected driver mapping, context and kernel stages, all points
  checked, and UID/groups matching the command's account.
- Require `llm.result == "PASS"` and full CUDA layer offload in `/api/status`.
  Ask an unscripted chat question and follow-up; record token counts and decode
  speed. Capture `journalctl -u minicpm` and repeat after restarting the service.
- `/run/admiral/nvidia/manifest.json` and the host readiness evidence through
  the trusted host-management channel. The manifest profile should be
  `tegra234-nvgpu-r39.2.1-v1`; record the exact observed value.
- Kernel logs immediately before and after execution using the host-management
  session, comparing new faults and taint changes. If workload policy blocks
  kernel-log access, collect them on the host rather than weakening policy.
- Restart the application, recreate the userspace through Admiral, reboot, and
  cold boot using the lab's normal procedure. Repeat the identity, SSH, GPU and
  log checks at each stage. Record any regenerated SSH host-key fingerprints.

Do not modify device modes, install a second driver, reload modules or change
firmware to make a test pass. Such changes belong in reproducible platform work.

## If GPU is red

SSH and the dashboard should still work. Inspect the exact failed stage:

```sh
timeout 30 demo-gpu
cat /run/admiral/nvidia/manifest.json
id
printenv LD_LIBRARY_PATH
LD_DEBUG=libs timeout 30 demo-gpu 2>/tmp/loader.log
cat /var/lib/admiral-demo/gpu-result.json
```

Missing driver: check `IsolateDevices=false`, mounts and Admiral launch logs.
Permission failure: confirm numeric GID 28 and actual device ownership.
JIT failure: retain the reported CUDA error and kernel delta; this demo exercises
the BSP's PTX/JIT payload. NixOS's glibc is intentionally retained; host libc and
the host ELF loader must not be copied into the image.
