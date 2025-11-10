fn main() {
    // We need to trigger a re-run if magdir changed
    // we cannot rely on proc-macro because we embed
    // only a compiled version of the rules within the
    // final binary. File tracking with include_byte!
    // seems to work only when the file is not eliminated
    // by DCE.
    println!("cargo::rerun-if-changed=../magic/src/magdir");
}
