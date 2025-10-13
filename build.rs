fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    if target_os == "macos" && target_arch == "aarch64" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:warning=Linking Accelerate framework for macOS ARM64");
    }
}
