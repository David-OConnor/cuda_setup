# Code to make working with CUDA, via the CUDARC lib, easier.

[![Crate](https://img.shields.io/crates/v/cuda_setup.svg)](https://crates.io/crates/lin_alg)
[![Docs](https://docs.rs/cuda_setup/badge.svg)](https://docs.rs/cuda_setup)


This library abstracts over some of the boilerplate needed to use the [Cudarc library](https://github.com/coreylowman/cudarc), for using 
CUDA GPU compute in the rust language. Primarily, it allows compiling of CUDA files (e.g. kernels and supporting code)
to PTX during application build.

To use, create a `build.rs` file like this:

```rust
//! We use this to automatically compile CUDA C++ code when building.

use cuda_setup::{build, GpuArchitecture};

fn main() {
    // The second parameter is a list of paths to all kernels to compile.
    // The first kernel passed must be the entry point. All others are just to watch for changes to trigger
    // a new compilation.
    build(GpuArchitecture::Rtx4, &vec!["src/cuda/cuda.cu", "src/cuda/util.cu"]);
}
```

Or if your application has CUDA feature-gated:
```rust
//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "cuda")]
use cuda_setup::{build, GpuArchitecture};

fn main() {
    #[cfg(feature = "cuda")]
    build(GpuArchitecture::Rtx4, &vec!["src/cuda/cuda.cu", "src/cuda/util.cu"]);
}
```


Include this in `Cargo.toml`:
```toml
[build-dependencies]
# For compiling kernels to PTX.
cuda_setup = "0.1.4"
```