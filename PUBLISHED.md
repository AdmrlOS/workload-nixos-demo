# Release: NixOS robotics userspace PoC

Release tag: `2026-09-21-nixos26.05-orin-r1` (2026-09-21, Australia/Sydney).

Source repository: https://github.com/AdmrlOS/workload-nixos-demo (public).
Container repository: ghcr.io/admrlos/workload-nixos-demo.

```text
ghcr.io/admrlos/workload-nixos-demo:2026-09-21-nixos26.05-orin-r1
ghcr.io/admrlos/workload-nixos-demo:latest
```

Immutable deployment reference:

```text
alexturner/workload-nixos-demo@sha256:f2b5cbe36a3447d7ef6dc84dc16a7dcbeedc85118a099ff51e69252acdcfb676
```

Platform: `linux/arm64`. Uncompressed image content: 1,663,654,794 bytes.
NixOS 26.05 / Nix 2.34.8 / systemd 260.4 / Python 3.13.15.
Nixpkgs revision: `6d663c0533ff269008fb84e45930151e37c99db9`.

Local Docker image ID:
`sha256:f2b5cbe36a3447d7ef6dc84dc16a7dcbeedc85118a099ff51e69252acdcfb676`.
Use the verified registry reference above for deployments.

Build archive SHA-256 (`build/image.tar.gz`, not tracked in Git):
`ea9594b783fbca5e0c4589232a8c139ee799c624be6b3bd9d200d69e099fa963`.

## Verified locally

- Native ARM64 build using the locked flake and pinned Nix builder image.
- Actual NixOS activation and systemd PID 1 on ARM64 Docker Desktop, with no
  failed systemd units in the final image.
- Live browser display, warehouse simulation, status API and non-root service
  identity: UID 1000, primary GID 100, supplementary GID 28.
- Nix store registration, offline evaluation and offline `nix shell` execution
  of the included hello package, including a separate non-root invocation.
- Real SSH public-key logins as root and demo, password/keyboard-interactive
  authentication disabled. Tests injected a temporary key only into disposable
  containers. The release image contains only the pinned alexanderturner key.
- No bundled `libcuda.so` or NVIDIA kernel-facing driver. With no NVIDIA driver
  on the local machine, the GPU command exits 1 and both JSON and UI report FAIL;
  SSH and the dashboard remain available.
- Service restart, orderly container stop, restart of the same writable
  snapshot, and preservation of its generated SSH host key. A temporary
  `ExecStopPost` marker verified service shutdown before the container stopped.
- PTX assembled successfully for `sm_87` using NVIDIA CUDA 12.6's `ptxas`
  (compiler package 12.6.85), with no spills. This is a compilation check, not
  GPU execution.
- Simulation tests exercise 200 patrol positions and confirm all synthetic
  lidar points remain on the arena boundary.

Docker reported exit 130 for orderly namespace halt, with no forced SIGKILL.
Linux documents this as the parent observing SIGINT when a child PID namespace
halts/powers off; see [reboot(2)](https://man7.org/linux/man-pages/man2/reboot.2.html).
The image uses `SIGRTMIN+3`, systemd's orderly stop signal.

## Not yet verified

This release has **not** been deployed through the Admiral portal to the Jetson.
The new NixOS/glibc combination, injected-driver resolution, actual CUDA JIT and
checked kernel execution, host kernel-log delta, container recreation and
warm/cold board boot remain pending. Earlier Ubuntu workload results do not
establish those properties for this NixOS image.

The demo's CUDA path is the Driver API with embedded PTX; it does not validate
libcudart, ROS, TensorRT, PyTorch, graphics, cameras or the customer's application.
Use [DEMO.md](DEMO.md) to collect the portal deployment and hardware evidence.
