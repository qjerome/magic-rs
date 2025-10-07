use criterion::{Criterion, criterion_group, criterion_main};
use lazy_cache::LazyCache;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use tempfile::NamedTempFile;

// Helper: Create a test file with the given size and pattern
fn create_test_file(size: usize) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    let mut buffer = vec![0u8; 4096]; // 4KB buffer
    for (i, b) in buffer.iter_mut().enumerate() {
        *b = (i % 256) as u8; // Fill with a repeating pattern
    }
    for _ in 0..(size / buffer.len()) {
        file.write_all(&buffer).unwrap();
    }
    file.seek(SeekFrom::Start(0)).unwrap();
    file
}

// Benchmark direct random access using `File`
fn benchmark_file_random_access<R: Read + Seek>(
    file: &mut R,
    offsets: &[u64],
    sizes: &[usize],
) -> std::io::Result<()> {
    let mut buf = vec![0u8; *sizes.iter().max().unwrap()];
    for (&offset, &size) in offsets.iter().zip(sizes.iter()) {
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut buf[..size])?;
    }
    Ok(())
}

// Benchmark random access using `RandomAccessFileCache`
fn benchmark_cache_random_access(
    cache: &mut LazyCache<File>,
    offsets: &[usize],
    sizes: &[usize],
) -> std::io::Result<()> {
    for (&offset, &size) in offsets.iter().zip(sizes.iter()) {
        let _ = cache.read_range(offset as u64..(offset + size) as u64)?;
    }
    Ok(())
}

// Generate random offsets and sizes for benchmarking
fn generate_random_accesses(file_size: usize, num_accesses: usize) -> (Vec<usize>, Vec<usize>) {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut offsets = Vec::with_capacity(num_accesses);
    let mut sizes = Vec::with_capacity(num_accesses);
    for _ in 0..num_accesses {
        let offset = rng.random_range(0..file_size);
        let size = rng.random_range(1..1024); // 1B to 1KB reads
        offsets.push(offset);
        sizes.push(size);
    }
    (offsets, sizes)
}

fn benchmark_random_access_lazy_cache(c: &mut Criterion) {
    // Create a 1GB test file
    let file_size = 1 << 30;
    let temp_file = create_test_file(file_size);
    let file_path = temp_file.path();

    // Generate random accesses
    let (offsets, sizes) = generate_random_accesses(file_size, 1000);

    // Convert offsets to u64 for `File` benchmark
    let offsets_u64: Vec<u64> = offsets.iter().map(|&x| x as u64).collect();

    // Benchmark direct file access
    let mut file = File::open(file_path).unwrap();
    c.bench_function("file_random_access", |b| {
        b.iter(|| {
            benchmark_file_random_access(&mut file, &offsets_u64, &sizes).unwrap();
        })
    });

    // Benchmark direct buffer reader access
    let mut br = io::BufReader::new(File::open(file_path).unwrap());
    c.bench_function("buffer_reader_random_access", |b| {
        b.iter(|| {
            benchmark_file_random_access(&mut br, &offsets_u64, &sizes).unwrap();
        })
    });

    // Open the file with cache (1MB cache, 4KB block size)
    let mut cache = LazyCache::open(file_path)
        .unwrap()
        .with_hot_cache(1 << 20)
        .unwrap();

    // Benchmark cached access
    c.bench_function("1mb_cache_random_access", |b| {
        b.iter(|| {
            benchmark_cache_random_access(&mut cache, &offsets, &sizes).unwrap();
        })
    });

    // Open the file with cache (5MB cache, 4KB block size)
    let mut cache = LazyCache::open(file_path)
        .unwrap()
        .with_hot_cache(5 << 20)
        .unwrap()
        .with_warm_cache(10 << 20);

    // Benchmark cached access
    c.bench_function("5mb_cache_random_access", |b| {
        b.iter(|| {
            benchmark_cache_random_access(&mut cache, &offsets, &sizes).unwrap();
        })
    });
}

criterion_group!(benches, benchmark_random_access_lazy_cache);
criterion_main!(benches);
