"""Builds the `forkunion` wheel: a linkable `libforkunion.so.3`, its headers, and the singleton pool extension.

Unlike a typical extension package, this one ships a *library* other packages build against, so `build_ext` is extended
to produce a standalone shared object with a versioned `SONAME` alongside the extension. The `SONAME` is what `ld.so`
de-duplicates on, and de-duplication is what keeps one process to one pool.
"""

import os
import platform
import shutil
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = open("VERSION").read().strip() if os.path.exists("VERSION") else "3.0.0"
__major__ = __version__.split(".")[0]

_IS_WINDOWS = sys.platform == "win32"
_IS_DARWIN = sys.platform == "darwin"


def _has_libnuma() -> bool:
    """The header decides NUMA support with `__has_include(<numa.h>)`, so link `libnuma` exactly when it is there.

    Build-host detection contradicts the library's own thesis of dispatching on the *run* host, and a wheel built here
    will not load where `libnuma.so.1` is absent. Loading it with `dlopen` instead is tracked separately; until then a
    redistributable wheel must be built without `numa.h` in the include path.
    """
    if not sys.platform.startswith("linux"):
        return False
    return any(os.path.exists(os.path.join(d, "numa.h")) for d in ("/usr/include", "/usr/local/include"))


def _compile_args(cpp: bool) -> list:
    if _IS_WINDOWS:
        return ["/std:c++20", "/O2", "/EHsc"] if cpp else ["/O2"]
    args = ["-O3", "-fPIC", "-fvisibility=hidden"]
    if cpp:
        args += ["-std=c++20", "-fvisibility-inlines-hidden", "-Wno-psabi"]
    else:
        args += ["-std=c11"]
    return args


class build_ext_with_library(build_ext):
    """Builds `libforkunion.so.3` before the extensions, then links `_core` against it."""

    def build_extensions(self):
        # Not `run()`: `self.compiler` is only instantiated once `build_ext.run` gets that far.
        library = self._build_shared_library()
        self._build_openmp_shim(library)
        self._copy_headers()
        # Link by path rather than `-lforkunion`: that spelling needs a `libforkunion.so` alias which the wheel cannot
        # carry anyway. The linker still records the `SONAME`, so `_core` depends on `libforkunion.so.3` either way.
        if _IS_WINDOWS:
            library = os.path.splitext(library)[0] + ".lib"
        for extension in self.extensions:
            extension.extra_link_args.append(library)
        super().build_extensions()

    @property
    def _package_dir(self) -> str:
        target = os.path.join(self.build_lib, "forkunion")
        os.makedirs(target, exist_ok=True)
        return target

    def _build_openmp_shim(self, library: str):
        """Build the libgomp-ABI shim: a library nothing links against, only `LD_PRELOAD`s or `dlopen`s.

        Deliberately unversioned. glibc accepts an unversioned definition for a versioned reference during symbol
        interposition, while a half-versioned library matches nothing at all.
        """
        if _IS_WINDOWS:
            return  # ? No `LD_PRELOAD` on Windows, and MSVC does not build against libgomp's ABI
        objects = self.compiler.compile(
            ["c/forkunion_gomp.cpp"],
            output_dir=self.build_temp,
            include_dirs=["include"],
            extra_postargs=_compile_args(cpp=True),
        )
        libraries = ["numa"] if _has_libnuma() else []
        if sys.platform.startswith("linux"):
            libraries += ["pthread", "atomic"]
        # The shim dispatches onto a pool, so it needs the library beside it in the package. `$ORIGIN` resolves it
        # wherever `site-packages` ends up, and linking by path records the `SONAME` so only one copy is ever mapped.
        origin = "@loader_path" if _IS_DARWIN else "$ORIGIN"
        name = "libforkunion_gomp.dylib" if _IS_DARWIN else "libforkunion_gomp.so"
        self.compiler.link_shared_object(
            objects,
            os.path.join(self._package_dir, name),
            libraries=libraries,
            extra_postargs=[library, f"-Wl,-rpath,{origin}"],
            target_lang="c++",
        )

    def _copy_headers(self):
        """Ship every header a consumer might include - the C ABI and the C++ core both."""
        include_root = os.path.join(self._package_dir, "include")
        if os.path.isdir(include_root):
            shutil.rmtree(include_root)
        shutil.copytree("include", include_root)
        shutil.copy("python/capsule.h", os.path.join(include_root, "forkunion_capsule.h"))

    def _build_shared_library(self):
        """Compile `c/forkunion.cpp` into a versioned shared library, not an extension."""
        compiler = self.compiler
        objects = compiler.compile(
            ["c/forkunion.cpp"],
            output_dir=self.build_temp,
            include_dirs=["include"],
            macros=[("FU_BUILDING_SHARED", "1")],
            extra_postargs=_compile_args(cpp=True),
        )

        libraries = []
        link_args = []
        # A wheel is a ZIP, and ZIP cannot store symlinks - `pip` would materialize each one as a full copy of a 2 MB
        # library. So build exactly the file the `SONAME` names, and let `create_library_symlinks()` add the `-l`
        # spelling at a consumer's build time, the way `pyarrow` does.
        if _IS_WINDOWS:
            name = "forkunion.dll"
        elif _IS_DARWIN:
            # ! macOS has no `SONAME`; the install name plays that role and must match what consumers record.
            name = f"libforkunion.{__major__}.dylib"
            link_args = [f"-Wl,-install_name,@rpath/{name}"]
        else:
            name = f"libforkunion.so.{__major__}"
            link_args = [f"-Wl,-soname,{name}"]
        if _has_libnuma():
            libraries.append("numa")
        if sys.platform.startswith("linux"):
            libraries += ["pthread", "atomic"]

        output = os.path.join(self._package_dir, name)
        compiler.link_shared_object(
            objects,
            output,
            libraries=libraries,
            extra_postargs=link_args,
            target_lang="c++",
        )

        return output


def _core_extension() -> Extension:
    link_args = []
    if not _IS_WINDOWS:
        origin = "@loader_path" if _IS_DARWIN else "$ORIGIN"
        link_args.append(f"-Wl,-rpath,{origin}")
    return Extension(
        "forkunion._core",
        sources=["python/_core.c"],
        include_dirs=["include", "python"],
        extra_compile_args=_compile_args(cpp=False),
        extra_link_args=link_args,  # ? `build_ext` appends the library path itself
        define_macros=[("FU_USING_SHARED", "1")],
    )


setup(
    name="forkunion",
    version=__version__,
    author="Ash Vardanian",
    description="NUMA-aware fork-join thread pool, shared across Python extensions",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=["forkunion"],
    package_dir={"forkunion": "python"},
    # The library and the headers are placed into the package by `build_ext`, not globbed from the source tree. Sources
    # are excluded outright: `capsule.h` is shipped as `include/forkunion_capsule.h`, and `_core.c` has no business in
    # a binary wheel.
    include_package_data=False,
    exclude_package_data={"forkunion": ["*.c", "capsule.h"]},
    ext_modules=[_core_extension()],
    cmdclass={"build_ext": build_ext_with_library},
    python_requires=">=3.9",
    zip_safe=False,
)
