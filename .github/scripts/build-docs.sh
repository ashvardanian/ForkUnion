#!/usr/bin/env bash
# Writes the symbol bundle unum.cloud renders: each binding's own tool output, or its sources where no tool emits one.
#
#     .github/scripts/build-docs.sh        # → build/forkunion_docs_<version>.tar.gz
set -euo pipefail
version=$(cat VERSION)
name=forkunion_docs_$version out=build/forkunion_docs_$version
# Rustdoc's JSON is unstable, and `rust-version` needs 1.100, which reaches stable on 2026-11-12.
toolchain=${FORKUNION_RUST_TOOLCHAIN:-beta-2026-10-18}
rm -rf "$out" && mkdir -p "$out"/{doxygen,headers,rust,mojo}

rm -rf build/xml && (cd docs && doxygen conf.dox) && cp -r build/xml/. "$out/doxygen"
git ls-files c/forkunion/forkunion.h include/forkunion.hpp 'include/forkunion/*.hpp' | xargs cp --parents -t "$out/headers"
RUSTC_BOOTSTRAP=1 cargo +"$toolchain" rustdoc --lib -- -Z unstable-options --output-format json
cp "${CARGO_TARGET_DIR:-target}/doc/forkunion.json" "$out/rust/"
git ls-files zig/forkunion.zig 'zig/forkunion/*.zig' | xargs cp --parents -t "$out"
mojo doc -o "$out/mojo/forkunion.json" mojo/forkunion
cp mojo/forkunion/__init__.mojo "$out/mojo/"
# Rustdoc, `mojo doc` and Doxygen's body locations hold absolute paths, which would leak the build machine's directories.
sed -i "s|$PWD/||g; s|$PWD|.|g; s|$HOME|~|g" "$out/rust/forkunion.json" "$out/mojo/forkunion.json" "$out"/doxygen/*.xml
printf '{"project":"forkunion","version":"%s","commit":"%s"}\n' "$version" "$(git rev-parse HEAD)" > "$out/manifest.json"
tar -czf "build/$name.tar.gz" -C build "$name"
