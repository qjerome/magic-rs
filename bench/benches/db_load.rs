//! Times loading the embedded `magic-db` ruleset, i.e. gunzip + decode of
//! the blob produced by `MagicDb::serialize`, to compare serialization
//! backends.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_deserialize(c: &mut Criterion) {
    c.bench_function("db_load/deserialize", |b| {
        b.iter_with_large_drop(|| {
            black_box(magic_db::load().expect("load embedded magic-db ruleset"))
        })
    });
}

criterion_group!(benches, bench_deserialize);
criterion_main!(benches);
