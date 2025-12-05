use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/cartpole.cu");

    // Find nvcc - check PATH first, then standard locations
    let nvcc = if Command::new("nvcc").arg("--version").output().is_ok() {
        "nvcc".to_string()
    } else {
        // Try environment variables
        let cuda_path = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        format!("{}/bin/nvcc", cuda_path)
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_src = "kernels/cartpole.cu";
    let ptx_out = out_dir.join("cartpole.ptx");

    // Compile CUDA kernel to PTX
    let output = Command::new(&nvcc)
        .args(&[
            kernel_src,
            "-ptx",
            "-o",
            ptx_out.to_str().unwrap(),
            "--std=c++14",
            "-O3",                          // Maximum optimization
            "--use_fast_math",              // Fast math operations
            "-arch=sm_86",                  // Target RTX 3090 (Ampere)
            "--expt-relaxed-constexpr",     // Allow constexpr in device code
        ])
        .output()
        .expect("Failed to execute nvcc. Is CUDA toolkit installed?");

    if !output.status.success() {
        panic!(
            "CUDA kernel compilation failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("Successfully compiled CartPole CUDA kernel to PTX");
}
