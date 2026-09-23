# NixOS Edge AI Lab on Admiral

A real ARM64 **NixOS 26.05 system userspace**, built with a pinned Nix flake and
packaged as an OCI-compatible container image. It boots systemd, runs an on-device
Edge LLM chat interface with official Admiral theming (admrl.co), exposes key-only SSH,
and executes real GPU tensor operations and coordinate transforms through Admiral's
BSP-owned NVIDIA driver.

- **Source:** `https://github.com/AdmrlOS/workload-nixos-demo`
- **Container Registry (GHCR):** `ghcr.io/admrlos/workload-nixos-demo:latest`
- **Release tag:** `ghcr.io/admrlos/workload-nixos-demo:2026-09-23-nixos26.05-minicpm5-r2`
- See [PUBLISHED.md](PUBLISHED.md) for immutable digests and validation details.

---

## The Core Demo: Taking Existing NixOS Workloads to Jetson Orin with Admiral

Teams using NixOS for robotics, computer vision, and edge computing love its reproducibility and declarative configuration, but running NixOS natively on **NVIDIA Jetson (Orin / Xavier)** is notoriously difficult:
- NVIDIA JetPack and Tegra L4T drivers are tightly coupled to specific kernel releases, device trees, out-of-tree kernel modules, and proprietary driver blobs.
- Packaging CUDA and Jetson BSP drivers directly inside NixOS requires complex custom overlays that break across JetPack versions.

### How Admiral makes it effortless

Admiral separates **platform hardware management** from your **application userspace**:

```
┌─────────────────────────────────────────────────────────────┐
│               YOUR NIXOS SYSTEM USERSPACE                   │
│   • Pinned packages, Nix flakes, your application services  │
│   • Standard systemd units, users, and tools                │
│   • Packaged cleanly as an OCI container image              │
└──────────────────────────────┬──────────────────────────────┘
                               │ (standard OCI image via GHCR)
┌──────────────────────────────▼──────────────────────────────┐
│                    ADMIRAL EDGE PLATFORM                    │
│   • Tested Jetson Orin BSP, kernel, bootloader, OTA updates │
│   • Injected read-only NVIDIA driver (/run/admiral/nvidia)  │
│   • Managed device lifecycle, cgroup v2, networking         │
└─────────────────────────────────────────────────────────────┘
```

### 3 Simple Steps to Port Any NixOS Workload to Admiral

1. **Keep your existing NixOS packages and services:** Your application code, Python/C++ binaries, and `systemd` service declarations remain ordinary NixOS definitions (see `nix/configuration.nix`).
2. **Add the 40-line Admiral adapter (`nix/admiral.nix`):**
   - Configures container mode (`boot.isContainer = true`).
   - Disables host daemon conflicts (udev, host sysctl, networkd).
   - Points `LD_LIBRARY_PATH` to Admiral's injected driver path (`/run/admiral/nvidia/lib`).
3. **Export as an OCI image (`flake.nix`):**
   - Use `pkgs.dockerTools.buildLayeredImage` with `/init` entrypoint.
   - Build, push to GHCR, and deploy to your Jetson devices instantly via Admiral!

No kernel re-compilation, no JetPack hacking, and zero driver blobs inside your container repository.

---

## What to show on the call

1. Open `http://DEVICE_IP:8080`. Interact with the on-device **MiniCPM5-2B**
   model via the chat interface styled with Admiral's theme (admrl.co). It runs
   real model inference using llama.cpp and CUDA on Jetson Orin, with an Admiral
   system prompt. All model layers must be offloaded to CUDA; tokenization and
   sampling use the CPU. Missing CUDA or model failures are reported honestly. The right-hand panel reports real-time inference
   telemetry and an independent GPU coordinate-transform test.
2. `ssh demo@DEVICE_IP` using the private key matching `alexanderturner`'s GitHub
   public key, then run `demo-status`, `demo-chat "What is Admiral OS?"`, and
   `timeout 30 demo-gpu`.
3. Show `/etc/os-release`, `nix --version`, `nix registry list`, and
   `systemctl status robotics-demo`.
4. Show `cat /etc/admiral-demo/flake.lock`, then run
   `nix shell --offline nixpkgs#hello -c hello`. The locked nixpkgs source and
   hello package are included for an offline Nix demonstration.
5. Show `/etc/admiral-demo/source/nix/configuration.nix`: it is an ordinary NixOS
   service and user configuration. The source and lockfile ship inside the image.

The story: **keep the customer's Nix packages and service modules; export the
system closure as an OCI image; let Admiral own the hardware platform and deploy
the image.** This proves that packaging path, not compatibility with every
customer module or arbitrary existing bootable disk image. The customer's own
configuration and application still need integration testing.

## Deploy through Admiral

Create/select workload `nixos-robotics` and deploy the versioned image through
the Admiral portal and cloud service. Use the immutable digest from PUBLISHED.md
when available.

- Target `linux/arm64` / Jetson Orin with the validated BSP and init integration.
- Use `IsolateDevices=false` and a writable root filesystem (`ReadOnlyRoot=false`).
- Preserve entrypoint `/init`; it activates NixOS and starts systemd as PID 1.
- Make TCP **8080** (dashboard) and **22** (SSH) reachable. Port 2222 is the
  existing host-management port and is not used by this image.
- Admiral supplies writable cgroup v2 and `/run`, `/tmp`, `/dev`, `/proc`, `/sys`.
  Keep the standard runtime mounts; do not mount over `/run/admiral`, `/usr/lib`,
  `/nix`, or the driver payload.
- No Docker daemon, NVIDIA container toolkit, extra driver installation,
  kernel installation, or host device-permission edits are needed inside NixOS.

The adapter disables guest network configuration, udev and sysctl management:
Admiral owns those host concerns. NixOS keeps its own glibc, ELF loader and
`/nix/store`. It explicitly exposes `/run/admiral/nvidia/lib` through
`LD_LIBRARY_PATH` to the application and SSH sessions because Nix-built programs
do not normally search Debian-style library locations.

Readiness and driver integrity remain Admiral init's responsibility. The image
does not infer success from a mounted library or a running web page.

## MiniCPM5-2B inference

`minicpm.service` runs the flake-pinned llama.cpp b9190 with CUDA 12.6,
compiled for Orin (`sm_87`). The official Q4_K_M GGUF (1.56 GB) is fetched
at revision `2079a22f3beaa4e306449978533478fe0522f4b3` and SHA-256 verified
by Nix, then included in the image. No model download is needed on the device.
See [OpenBMB's deployment guide](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/llama_cpp.md).

The server binds only `127.0.0.1:8081`, explicitly selects `CUDA0`, disables
memory auto-fitting and requests all layers on GPU. The supervisor records full
offload and a CUDA model buffer; `/api/status` also checks server health before
reporting model readiness. Driver availability or the independent probe alone
cannot make the LLM status PASS. Missing/incompatible CUDA fails without a CPU
fallback. CUDA 12.6 runtime compatibility with the deployed BSP must be checked
on the target; allow space for model, runtime, KV cache and the larger OCI image.

The existing browser and `demo-chat` use the same local model service. The web
chat retains up to six turns, validates history, sends the server-owned Admiral
system prompt, and displays the actual generated answer without a typing delay.
Generation uses the GGUF chat template, an 8192-token context and up to 512 output
tokens. Oversized conversations fail explicitly; Clear starts a fresh chat.
Token counts and decode speed come from llama.cpp; latency is measured wall time.
There are no scripted answers, artificial compute loops or fabricated timings.

Inspect `journalctl -u minicpm -f` and `systemctl status minicpm robotics-demo`.
On Jetson, require `llm.result == "PASS"`, equal nonzero `gpu_layers` and
`total_layers`, and a successful `/api/chat` answer with real token usage. Ask
an unscripted question and a follow-up to confirm generation and context. Restart
`minicpm` and verify chat becomes unavailable until the model is ready again.
The separate coordinate probe must also pass. Local tests mock the inference
server and cannot establish Jetson performance or hardware acceptance.

## GPU check

`demo-gpu` loads `libcuda.so.1`, reports its actual process mappings and profile,
then performs initialization, enumeration, context creation, PTX JIT, allocation,
host/device copies, kernel launch, synchronization and verification of **65,539**
2D point transforms. Every coordinate is compared with a CPU reference (absolute
tolerance `1e-4`); output buffers are poisoned with NaNs before launch to reject
unwritten or stale results, and cleanup completes before PASS. It tests device index 0,
appropriate to this single-GPU Orin target. Every failed stage returns nonzero.
There is **no CPU fallback for the GPU test**.

This small Python/ctypes application uses the **CUDA Driver API** and embedded
PTX 7.0 targeting `sm_80`, JIT-compiled by Orin's injected driver. The probe itself needs no CUDA toolkit. The LLM service additionally ships
Nix-packaged CUDA runtime and cuBLAS libraries; nvcc is a build dependency.
The host NVIDIA driver is still supplied by Admiral. It is not acceptance of a customer's CUDA Runtime API or
framework stack. The pipeline's measured duration includes setup, JIT, transfers
and CPU verification and must not be presented as a GPU performance benchmark.

The dashboard runs the probe in a separate process with a 25-second timeout,
waits 30 seconds between runs, and stays available after failure. Its last result
is in `/var/lib/admiral-demo/gpu-result.json` and the systemd journal. GPU PASS is
displayed only for a checked result; expired or disconnected results are marked.
`/healthz` checks the web service only. Use `/api/status` to inspect the GPU result.

## SSH and identity

`ssh/authorized_keys` is a pinned public-key snapshot from
https://github.com/alexanderturner.keys, retrieved on 2026-09-21:

```text
ED25519 SHA256:yp9F51C0Pp4QoSw3WbEKDpSsnGoh0n0UXjK/p6qqCUE
```

Both `root` and `demo` accept that key; password and keyboard-interactive SSH
authentication are disabled. `demo` is UID 1000 with supplementary GID **28**
(`admiral-video`) and has no sudo access. The web service runs as this same user.
Use root only for administration and root/non-root comparison tests.

Host SSH keys are generated on first boot, never baked into the image. They
survive a restart of the same writable snapshot, but recreation changes them
unless `/etc/ssh` is persisted. Verify new fingerprints using the trusted
management channel before updating known_hosts. Persist `/var/lib/admiral-demo`
if the GPU result history is needed beyond snapshot recreation. Do not persist
`/nix` from an older image over the new image's store.

Rotate access by updating `ssh/authorized_keys` and rebuilding. Keys are never
fetched at runtime. The read-only dashboard is intended for the demo network;
do not expose its HTTP port publicly without an authenticated proxy.

## Build and publish

### Automated via GitHub Actions
Every push to `main` and release tag triggers the GitHub Actions workflow (`.github/workflows/build-and-publish.yml`), which builds the ARM64 image and publishes it to GitHub Container Registry:
- `ghcr.io/admrlos/workload-nixos-demo:latest`
- `ghcr.io/admrlos/workload-nixos-demo:<tag>`

### Local Build & Test

On a native ARM64 Linux Nix builder:

```sh
nix build .#image -L
docker load -i result
docker tag workload-nixos-demo:2026-09-23-nixos26.05-minicpm5-r2 \
  ghcr.io/admrlos/workload-nixos-demo:2026-09-23-nixos26.05-minicpm5-r2
```

On Apple Silicon with Docker Desktop, the helper starts a pinned Linux Nix
builder and keeps downloaded packages in a dedicated named volume:

```sh
bash scripts/build.sh
bash scripts/test-local.sh
docker push ghcr.io/admrlos/workload-nixos-demo:2026-09-23-nixos26.05-minicpm5-r2
docker push ghcr.io/admrlos/workload-nixos-demo:latest
```

The helper expects a Git checkout; new source files must be `git add`ed for Nix
to see them. It reuses the `admiral-nixos-builder` container, whose `/work` mount
must point at this checkout. Remove that container before building a different
checkout. The `admiral-nixos-store` volume may be retained as a cache. An x86 host
needs ARM64 emulation or a native ARM64 builder; this is not a cross-compiled image.

`flake.lock` pins nixpkgs; the builder is pinned by container digest. The image
uses the complete NixOS runtime closure and Nix's store registration, including
the locked nixpkgs source for offline evaluation. No downloads are needed at boot.

## Bring the customer's build

`nix/admiral.nix` is the platform adapter; `nix/configuration.nix` contains the
replaceable demonstration application. Start with a customer's **userspace**
modules and packages, import the adapter, and reuse the `flake.nix` closure/image
construction. Remove the demo-specific application, users and SSH settings as
appropriate. Keep hardware-specific bootloader, filesystems, kernel, udev,
network and NVIDIA-driver modules out of this userspace configuration.

Existing Nix-built ELF applications retain their store paths and runtime
dependencies. Existing NixOS service definitions remain systemd services.
Packaging a disk/VM image directly as a container is not sufficient: the image
needs its Nix closure, activation, store registration and container-safe services.

Consult [DEMO.md](DEMO.md) for a call script and Jetson acceptance checklist.
The official [NixOS container module](https://github.com/NixOS/nixpkgs/blob/6d663c0533ff269008fb84e45930151e37c99db9/nixos/modules/profiles/docker-container.nix)
provides the container activation/store registration behavior, and NVIDIA's
[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/) documents the GPU
calls used by this demo.
