# Releasing

1. Update `pixi.toml`, `conda.recipe/recipe.yaml`, the dated changelog entry,
   and compatibility notes for `X.Y.Z`. Keep the Pixi Mojo constraint and the
   recipe's build, host, and run `mojo-compiler` requirements exact and equal.
2. Run `pixi lock --check`, `pixi run --locked check`, and
   `pixi run --locked package` on a clean tree.
3. Create an annotated tag for the exact tested commit with
   `git tag -a vX.Y.Z -m "ShuhaFFT vX.Y.Z"`.
4. Replace the local `source.path` in the modular-community recipe submission
   with the repository URL and full 40-character tag commit SHA.
5. Reset the Conda build number to zero for a new version; increment it only
   when rebuilding the same source version.
6. Build the recipe and verify its installed-package smoke test.
7. Publish benchmark results only with the checked-in methodology.

The tag workflow rejects lightweight tags, commits not already in `origin/main`,
and any mismatch among the tag, package versions, exact Mojo compiler pins, and
dated changelog entry. It runs the locked checks and installed-package build on
all three supported native platforms before creating a source archive.
Publishing to modular-community is a separate reviewed operation.
