# Lighter Signers

This directory contains various signer implementations for the Lighter Protocol.

## Available Signers

### Native Binary Signers

- `signer-amd64.so` - Linux AMD64 native signer
- `signer-arm64.dylib` - macOS ARM64 native signer
- `signer-amd64.dll` - Windows AMD64 native signer

These binaries are compiled from the Go implementation in [lighter-go](https://github.com/elliottech/lighter-go) and
provide high-performance cryptographic operations for the Lighter Protocol.

## Usage

The Python SDK automatically selects the correct native binary signer based on your platform:

- ✅ **Linux (x86_64)**: Uses `signer-amd64.so`
- ✅ **macOS (ARM64)**: Uses `signer-arm64.dylib`
- ✅ **Windows (x86_64)**: Uses `signer-amd64.dll`

No additional configuration is required - the SDK detects your platform and loads the appropriate signer.

## Building Signers

All native signers are built from the Go implementation in [lighter-go](https://github.com/elliottech/lighter-go).

To build signers locally,Import lighter-go repo:

```bash
# From lighter-go/ directory
just build-linux-local        # Linux AMD64
just build-darwin-local       # macOS ARM64
just build-windows-local      # Windows AMD64
```

Or use Docker for cross-compilation:

```bash
just build-linux-docker
just build-windows-docker
```

## Supported Platforms

The Python SDK supports the following platforms out of the box:

| Platform | Architecture          | Binary               |
|----------|-----------------------|----------------------|
| Linux    | x86_64                | `signer-amd64.so`    |
| macOS    | ARM64 (Apple Silicon) | `signer-arm64.dylib` |
| Windows  | x86_64                | `signer-amd64.dll`   |

If you encounter issues with missing binaries, ensure the appropriate signer binary is present in this directory. You
can build it from `lighter-go/` using the commands above.