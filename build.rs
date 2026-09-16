// Emit the linker arguments an extension module needs on macOS.
//
// This call is a no-op on every non-Apple target, so CI is unaffected.
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
