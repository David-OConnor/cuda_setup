//! [Tutorial](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
//!
//! When compiling, you must have this or equivalent in the `PATH` environment variable (Windows), or
//! `LD_LIBRARY_PATH1 (Linux):
//! `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64`
//!
//! You may also need the build tools
//! containing `cl.exe` or similar in the path, e.g.: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`

use std::process::Command;
// use std::sync::Arc;

// use cudarc::{
//     driver::{CudaContext, CudaModule, CudaStream},
// };

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum GpuArchitecture {
    /// "Turing"
    Rtx2 = 75,
    /// "Ampere"
    Rtx3 = 86,
    /// "Ada"
    Rtx4 = 89,
    /// "Blackwell"
    Rtx5 = 100,
}

impl GpuArchitecture {
    /// [Selecting architecture, by nVidia series](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
    pub fn gencode_val(&self) -> String {
        let v = (*self) as u8;
        String::from(format!("arch=compute_{v},code=sm_{v}"))
    }

    pub fn compute_val(&self) -> String {
        let v = (*self) as u8;
        format!("-arch=compute_{v}")
    }
}

// #[derive(Debug, Clone, Default)]
// pub enum ComputationDevice {
//     #[default]
//     Cpu,
//     Gpu((Arc<CudaStream>, Arc<CudaModule>)),
// }

/// Call this in `build.rs` to compile the kernal.
///
/// See [These CUDA docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
/// for info about these flags.
///
/// Compiles our CUDA program using Nvidia's NVCC compiler
/// Call this in the `main()` fn of `build.rs`. `cuda_files`'s first parameter
/// must be the entry point. The other parameters are just so the build step
/// knows when to re-compile.
///
/// The architecture provided is the minimum supported one the PTX will output.
pub fn build(min_arch: GpuArchitecture, cuda_files: &[&str], ptx_filename: &str) {
    if cuda_files.len() < 1 {
        return;
    }

    // Tell Cargo that if the given file changes, to rerun this build script.
    for kernel in cuda_files {
        println!("cargo:rerun-if-changed={kernel}");
    }

    let compilation_result = Command::new("nvcc")
        .args([
            cuda_files[0],
            // "-gencode",
            // &architecture.gencode_val_sm(),
            &min_arch.compute_val(),
            "-ptx",
            "-O3", // optimized/release mode.
            "-o",
            &format!("{ptx_filename}.ptx"),
        ])
        .output();

    match compilation_result {
        Ok(output) => {
            if output.status.success() {
                println!("Compiled the following CUDA files: {cuda_files:?}");
            } else {
                // eprintln!(
                panic!(
                    "CUDA compilation problem:\nstatus: {}\nstdout: {}\nstderr: {}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
        Err(e) => eprintln!("Unable to compile CUDA files: {e}"),
    }
}
