//! [Tutorial](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
//!
//! You must have this or equivalent in the PATH environment variable:
//! `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64`
//!

// use cudarc::{
//     driver::{CudaContext, CudaModule, CudaStream},
// };
use std::process::Command;
// use std::sync::Arc;

#[derive(Copy, Clone)]
pub enum GpuArchitecture {
    Rtx2,
    Rtx3,
    Rtx4,
    Rtx5,
}

impl GpuArchitecture {
    /// [Selecting architecture, by nVidia series](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
    pub fn gencode_val(&self) -> String {
        let version: u8 = match self {
            Self::Rtx2 => 75,
            Self::Rtx3 => 86,
            Self::Rtx4 => 89,
            Self::Rtx5 => 100,
        };

        String::from(format!("arch=compute_{version},code=sm_{version}"))
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
pub fn build(architecture: GpuArchitecture, cuda_files: &[&str]) {
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
            "-gencode",
            &architecture.gencode_val(),
            "-ptx",
            "-O3", // optimized/release mode.
        ])
        .output();

    match compilation_result {
        Ok(output) => {
            if output.status.success() {
                println!("Compiled the following CUDA files: {cuda_files:?}");
            } else {
                eprintln!(
                    "CUDA compilation problem:\nstatus: {}\nstdout: {}\nstderr: {}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
        Err(e) => eprintln!("Unable to compile CUDA files: {e}")
    }


}
