# Compatibility

## Toolchain

Development currently pins Mojo `1.0.0`. Precompiled `.mojoc` files are tied to
the exact compiler version that produced them, so both the Pixi environment and
Conda recipe pin the compiler. Compiler upgrades are explicit compatibility
events and require the full locked test suite.

## Platforms

| Platform | Status |
| --- | --- |
| macOS ARM64 | CI target |
| Linux x86-64 | CI target |
| Linux ARM64 | CI target |
| Windows/WSL | Not yet supported or tested |
| GPU | Not supported unless explicitly listed in the roadmap |

The v0.1 API is experimental and has no source-compatibility promise across
0.x releases. Each release names the exact compiler used to build it.
