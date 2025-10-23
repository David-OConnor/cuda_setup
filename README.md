# Code to make working with CUDA, via the CUDARC lib, easier.

[![Crate](https://img.shields.io/crates/v/cuda_setup.svg)](https://crates.io/crates/lin_alg)
[![Docs](https://docs.rs/cuda_setup/badge.svg)](https://docs.rs/cuda_setup)


This library compiles CUDA GPU kernels and CUDA host-side FFI (e.g. for cuFFT). It contains functions that run
in your application or library's `build.rs` file, in preparation for running CUDA kernels using FFI or [Cudarc ](https://github.com/coreylowman/cudarc).
Primarily, it compiles of CUDA files (e.g. kernels and supporting code) to PTX during application build, 
using the `nvcc` compiler that comes with CUDA. `nvcc` must be installed on your system, e.g. via the [Cuda toolkit](https://developer.nvidia.com/cuda-toolkit).

Note: You must set the environment var `LD_LIBARARY_PATH` (Linux) or `PATH` (Windows) to your CUDA bin
directory, e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`. You may also need the build tools
containing `cl.exe` or similar in the path, e.g.: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`

To use, create a `build.rs` file like this. It will automatically compile a PTX file. You should 
package this PTX with your program, either as a file, or included in the binary.

```rust
//! We use this to automatically compile CUDA C++ code when building.

use cuda_setup::{build, GpuArchitecture};

fn main() {
    // -This first parameter is the minimum GPU architecture that will be compatible.
    // -The second parameter is a list of paths to all kernels to compile.
    // The first kernel passed must be the entry point. Others are just to watch for changes to trigger
    // a new compilation.
    // -The third parameter is the name of the PTX file to output.
    build(GpuArchitecture::Rtx4, &vec!["src/cuda/cuda.cu", "src/cuda/util.cu"], "my_program");
}
```

Or if your application has CUDA feature-gated, you may wish to use this pattern.
```rust
//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "cuda")]
use cuda_setup::{build, GpuArchitecture};

fn main() {
    #[cfg(feature = "cuda")]
    build(GpuArchitecture::Rtx4, &vec!["src/cuda/cuda.cu", "src/cuda/util.cu"]);
}
```

If you are compilng host FFI code, e.g. to use cuFFT:
```rust
    cuda_setup::build_host(
        GpuArchitecture::Rtx3,
        &["src/cuda/cufft.cu", "src/cuda/kernels.cu"],
        "spme",
    );
```


Include this in `Cargo.toml`:
```toml
[build-dependencies]
# For compiling kernels to PTX.
cuda_setup = "0.1.8"
```