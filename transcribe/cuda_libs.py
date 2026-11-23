"""Load pip-installed cuBLAS/cuDNN so CTranslate2 finds them without LD_LIBRARY_PATH."""

import ctypes
import importlib.util
import os

_done = False


def preload() -> None:
    global _done
    if _done:
        return
    _done = True
    spec = importlib.util.find_spec("nvidia")
    if not spec or not spec.submodule_search_locations:
        return
    root = list(spec.submodule_search_locations)[0]
    libs = ["cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12"]
    cudnn = os.path.join(root, "cudnn", "lib")
    if os.path.isdir(cudnn):
        libs += [f"cudnn/lib/{n}" for n in sorted(os.listdir(cudnn)) if ".so" in n]
    for lib in libs:
        path = os.path.join(root, lib)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
