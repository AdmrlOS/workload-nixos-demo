"""Checked CUDA Driver API point transform. No driver, toolkit or CPU fallback.

PTX is deliberately conservative (ISA 7.0 / sm_80) and JITs on Orin sm_87.
The Python interpreter and its ELF loader/glibc are supplied by NixOS.
"""
import ctypes as C
import json
import math
import os
from pathlib import Path
import time

PTX = r"""
.version 7.0
.target sm_80
.address_size 64
.visible .entry transform(
 .param .u64 px, .param .u64 py, .param .u64 pox, .param .u64 poy,
 .param .f32 pc, .param .f32 ps, .param .f32 ptx, .param .f32 pty,
 .param .u32 pn)
{
 .reg .pred %p;
 .reg .b32 %r<5>;
 .reg .b64 %rd<10>;
 .reg .f32 %f<13>;
 mov.u32 %r0, %ctaid.x;
 mov.u32 %r1, %ntid.x;
 mov.u32 %r2, %tid.x;
 mad.lo.u32 %r3, %r0, %r1, %r2;
 ld.param.u32 %r4, [pn];
 setp.ge.u32 %p, %r3, %r4;
 @%p bra DONE;
 mul.wide.u32 %rd0, %r3, 4;
 ld.param.u64 %rd1, [px];
 ld.param.u64 %rd2, [py];
 ld.param.u64 %rd3, [pox];
 ld.param.u64 %rd4, [poy];
 add.u64 %rd5, %rd1, %rd0;
 add.u64 %rd6, %rd2, %rd0;
 add.u64 %rd7, %rd3, %rd0;
 add.u64 %rd8, %rd4, %rd0;
 ld.global.f32 %f0, [%rd5];
 ld.global.f32 %f1, [%rd6];
 ld.param.f32 %f2, [pc];
 ld.param.f32 %f3, [ps];
 ld.param.f32 %f4, [ptx];
 ld.param.f32 %f5, [pty];
 mul.f32 %f6, %f0, %f2;
 mul.f32 %f7, %f1, %f3;
 sub.f32 %f8, %f6, %f7;
 add.f32 %f8, %f8, %f4;
 mul.f32 %f9, %f0, %f3;
 mul.f32 %f10, %f1, %f2;
 add.f32 %f11, %f9, %f10;
 add.f32 %f11, %f11, %f5;
 st.global.f32 [%rd7], %f8;
 st.global.f32 [%rd8], %f11;
DONE:
 ret;
}
"""


def probe():
    started = time.monotonic()
    result = {"result": "FAIL", "stage": "load_driver", "uid": os.getuid(),
              "gid": os.getgid(), "groups": os.getgroups(), "stages": [],
              "profile": None, "timestamp": time.time(), "runtime": "CUDA Driver API / PTX 7.0"}
    ctx, lib = C.c_void_p(), None
    allocations = []
    module = C.c_void_p()

    def api(name, args):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, C.c_int
        return fn

    def check(name, code):
        result["stage"] = name
        if code:
            message = C.c_char_p()
            api("cuGetErrorString", [C.c_int, C.POINTER(C.c_char_p)])(code, C.byref(message))
            raise RuntimeError(f"{name}: CUDA {code}: {(message.value or b'unknown').decode()}")
        result["stages"].append(name)

    try:
        manifest = Path("/run/admiral/nvidia/manifest.json")
        if manifest.exists():
            result["profile"] = json.loads(manifest.read_text()).get("profile")
        lib = C.CDLL("libcuda.so.1", mode=os.RTLD_NOW | os.RTLD_LOCAL)
        result["driver_mappings"] = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libcuda.so" in line})
        result["stages"].append("load_driver")
        check("cuInit", api("cuInit", [C.c_uint])(0))
        version, count, device = C.c_int(), C.c_int(), C.c_int()
        check("cuDriverGetVersion", api("cuDriverGetVersion", [C.POINTER(C.c_int)])(C.byref(version)))
        result["driver_version"] = version.value
        check("cuDeviceGetCount", api("cuDeviceGetCount", [C.POINTER(C.c_int)])(C.byref(count)))
        result["device_count"] = count.value
        if count.value < 1:
            raise RuntimeError("No CUDA devices")
        check("cuDeviceGet", api("cuDeviceGet", [C.POINTER(C.c_int), C.c_int])(C.byref(device), 0))
        name = C.create_string_buffer(256)
        check("cuDeviceGetName", api("cuDeviceGetName", [C.c_void_p, C.c_int, C.c_int])(name, 256, device))
        result["device"] = name.value.decode()
        major, minor = C.c_int(), C.c_int()
        attr = api("cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int])
        check("compute_major", attr(C.byref(major), 75, device))
        check("compute_minor", attr(C.byref(minor), 76, device))
        result["compute_capability"] = f"{major.value}.{minor.value}"
        check("context_create", api("cuCtxCreate_v2", [C.POINTER(C.c_void_p), C.c_uint, C.c_int])(C.byref(ctx), 0, device))
        check("module_JIT", api("cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_void_p])(C.byref(module), C.c_char_p(PTX.encode())))
        kernel = C.c_void_p()
        check("get_kernel", api("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])(C.byref(kernel), module, b"transform"))
        n = 65539  # Includes an incomplete block to exercise bounds checking.
        array = C.c_float * n
        xs = array(*(math.sin(i * .017) * (1 + i % 23) for i in range(n)))
        ys = array(*(math.cos(i * .013) * (1 + i % 19) for i in range(n)))
        ox, oy = array(), array()
        size = C.sizeof(xs)
        for _ in range(4):
            ptr = C.c_uint64()
            check("allocate", api("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t])(C.byref(ptr), size))
            allocations.append(ptr)
        h2d = api("cuMemcpyHtoD_v2", [C.c_uint64, C.c_void_p, C.c_size_t])
        check("copy_x_to_device", h2d(allocations[0], xs, size))
        check("copy_y_to_device", h2d(allocations[1], ys, size))
        values = allocations + [C.c_float(math.cos(.7)), C.c_float(math.sin(.7)), C.c_float(2.5), C.c_float(-1.25), C.c_uint(n)]
        params = (C.c_void_p * len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
        launch = api("cuLaunchKernel", [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p])
        check("kernel_launch", launch(kernel, (n + 255) // 256, 1, 1, 256, 1, 1, 0, None, params, None))
        check("synchronize", api("cuCtxSynchronize", [])())
        d2h = api("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t])
        check("copy_x_to_host", d2h(ox, allocations[2], size))
        check("copy_y_to_host", d2h(oy, allocations[3], size))
        result["stage"] = "verify"
        c, s, tx, ty = [v.value for v in values[4:8]]
        maximum = 0.0
        for i in range(n):
            ex, ey = xs[i] * c - ys[i] * s + tx, xs[i] * s + ys[i] * c + ty
            error = max(abs(ox[i] - ex), abs(oy[i] - ey))
            if not math.isfinite(ox[i]) or not math.isfinite(oy[i]) or error > 0.0001:
                raise RuntimeError(f"Point {i}: incorrect GPU result; error={error}")
            maximum = max(maximum, error)
        result.update(points_verified=n, max_error=maximum, tolerance=0.0001)
        result["stages"].append("verify_all_points")
        # Cleanup is checked too, before reporting PASS.
        for ptr in allocations[:]:
            check("free", api("cuMemFree_v2", [C.c_uint64])(ptr))
            allocations.remove(ptr)
        check("module_unload", api("cuModuleUnload", [C.c_void_p])(module))
        module = C.c_void_p()
        check("context_destroy", api("cuCtxDestroy_v2", [C.c_void_p])(ctx))
        ctx = C.c_void_p()
        result.update(result="PASS", stage="complete")
    except Exception as error:
        result["error"] = str(error)
    finally:
        if lib and ctx.value:
            # Best-effort cleanup after failure; the process exits afterwards.
            try:
                api("cuCtxDestroy_v2", [C.c_void_p])(ctx)
            except Exception:
                pass
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
    return result


if __name__ == "__main__":
    result = probe()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["result"] == "PASS" else 1)
