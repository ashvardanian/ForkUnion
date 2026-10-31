#!/usr/bin/env bash
# Mints a release: stamps the version everywhere, commits, tags, pushes and publishes the release page.
#
#     .github/scripts/mint-release.sh 3.0.4          # stable
#     .github/scripts/mint-release.sh 4.0.0-rc.1     # candidate, published as a prerelease
set -euo pipefail
version=$1 core=${1%%-*}
branch=$(git branch --show-current)
previous=$(git describe --tags --abbrev=0)
git diff --quiet HEAD || { echo "mint-release: commit or stash your changes first"; exit 1; }

IFS=. read -r major minor patch <<< "$core"
export version core major minor patch
echo "$version" > VERSION
perl -pi -e 's/^(version = ")[^"]+/$1$ENV{version}/' Cargo.toml
perl -0pi -e 's/^(name = "forkunion"\nversion = ")[^"]+/$1$ENV{version}/m' Cargo.lock
perl -pi -e 's/^version: .+$/version: $ENV{version}/' CITATION.cff
perl -pi -e 's/^(    \.version = ")[^"]+/$1$ENV{version}/' build.zig.zon
perl -pi -e 's/^([[:space:]]*GIT_TAG v)[^[:space:]]+$/$1$ENV{version}/; s/^(forkunion = (\{ version = )?")[^"]+/$1$ENV{version}/; s/(ForkUnion\/archive\/refs\/tags\/v)[^\/[:space:]]+(\.tar\.gz)/$1$ENV{version}$2/' README.md
perl -pi -e 's/^([[:space:]]+VERSION )[0-9.]+$/$1$ENV{core}/' CMakeLists.txt
perl -pi -e 's/^(#define FORKUNION_VERSION_MAJOR )[0-9]+$/$1$ENV{major}/; s/^(#define FORKUNION_VERSION_MINOR )[0-9]+$/$1$ENV{minor}/; s/^(#define FORKUNION_VERSION_PATCH )[0-9]+$/$1$ENV{patch}/' include/forkunion.hpp include/forkunion/types.hpp
git commit -am "Release: v$version" # commit-msg rejects it if any site was missed
git tag -a "v$version" -m "Release: v$version"
git push --atomic origin "$branch" "v$version"

# Candidates never take "Latest", and neither does a patch to an older line.
flags=()
[[ $version == *-rc.* ]] && flags+=(--prerelease --title "Candidate v$version")
[ "$(git tag -l 'v*' --sort=-v:refname | grep -v -- - | head -1)" = "v$version" ] || flags+=(--latest=false)
gh release create "v$version" --verify-tag --notes "$(git log --no-merges --format='- %s (%h)' "$previous..v$version^")" "${flags[@]}"
