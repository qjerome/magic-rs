#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let db = magic_db::global().unwrap();
    let _ = db.best_magic_slice(data);
});
