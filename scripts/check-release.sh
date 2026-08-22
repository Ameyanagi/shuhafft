#!/usr/bin/env bash

set -euo pipefail

tag_name=${1:-}
expected_commit=${2:-HEAD}

if [[ ! "$tag_name" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    echo "release tag '$tag_name' must match vMAJOR.MINOR.PATCH" >&2
    exit 1
fi
release_version=${BASH_REMATCH[1]}
tag_ref="refs/tags/$tag_name"

if ! git rev-parse --verify --quiet "$tag_ref" >/dev/null; then
    echo "release tag '$tag_name' is not available in the checkout" >&2
    exit 1
fi
if [[ $(git cat-file -t "$tag_ref") != "tag" ]]; then
    echo "release tag '$tag_name' must be an annotated tag" >&2
    exit 1
fi

tag_commit=$(git rev-parse "$tag_ref^{commit}")
checkout_commit=$(git rev-parse "$expected_commit^{commit}")
if [[ "$tag_commit" != "$checkout_commit" ]]; then
    echo "release tag '$tag_name' resolves to $tag_commit, not $checkout_commit" >&2
    exit 1
fi
: "${GITHUB_SHA:=$expected_commit}"
if ! git merge-base --is-ancestor "$GITHUB_SHA" refs/remotes/origin/main; then
    echo "release commit $GITHUB_SHA is not contained in origin/main" >&2
    exit 1
fi

pixi_version=$(sed -nE 's/^version = "([^"]+)"$/\1/p' pixi.toml)
recipe_version=$(sed -nE 's/^  version: "([^"]+)"$/\1/p' conda.recipe/recipe.yaml)
if [[ "$pixi_version" != "$release_version" ]]; then
    echo "release tag '$tag_name' does not match pixi.toml version '$pixi_version'" >&2
    exit 1
fi
if [[ "$recipe_version" != "$release_version" ]]; then
    echo "release tag '$tag_name' does not match recipe version '$recipe_version'" >&2
    exit 1
fi

pixi_compiler=$(sed -nE 's/^mojo = "==([^"]+)"$/\1/p' pixi.toml)
if [[ -z "$pixi_compiler" ]]; then
    echo "pixi.toml must pin mojo with an exact ==VERSION constraint" >&2
    exit 1
fi

for section in build host run; do
    recipe_pin=$(
        awk -v target="$section" '
            $0 == "  " target ":" { in_section = 1; next }
            in_section && /^  [[:alnum:]_-]+:$/ { exit }
            in_section && /^    - mojo-compiler / {
                sub(/^    - mojo-compiler /, "")
                print
            }
        ' conda.recipe/recipe.yaml
    )
    if [[ "$recipe_pin" != "=$pixi_compiler" ]]; then
        echo "recipe $section requirement must be exactly mojo-compiler =$pixi_compiler" >&2
        exit 1
    fi
done

escaped_version=${release_version//./\\.}
changelog_dates=$(
    sed -nE "s/^## \\[$escaped_version\\] - ([0-9]{4}-[0-9]{2}-[0-9]{2})$/\\1/p" CHANGELOG.md
)
if [[ ! "$changelog_dates" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "CHANGELOG.md must contain one dated [$release_version] heading" >&2
    exit 1
fi

echo "release contract verified: $tag_name, Mojo $pixi_compiler, changelog $changelog_dates"
