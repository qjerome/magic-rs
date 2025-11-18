<!-- cargo-rdme start -->

# Pymagic-rs: Pure Rust Python Bindings for File Type Detection

`pymagic-rs` is a Python module that provides bindings to the [`magic-rs`](https://crates.io/crates/magic-rs) crate, allowing you to detect file types, MIME types, and other metadata using a pure Rust alternative to the C `libmagic` library. This module uses an **embedded magic database**, so everything works out-of-the-box with **no need** of a compiled database or magic rule files.

## Features
- **Pure Rust implementation**: No C dependencies, easily cross-compilable for great compatibility
- **Embedded magic database**: No database file or magic rules needed
- Detect file types from buffers or files
- Retrieve MIME types, creator codes, and file extensions
- Convert results to Python dictionaries for easy integration
- Supports both "first match", "best match" and "all matches" detection strategies

## Installation
```bash
pip install pymagic-rs
```

## Usage

### Initializing the Database
```python
from pymagic import MagicDb
# Initialize the Magic database (uses embedded database)
db = MagicDb()
```

### Detecting File Types from Buffers
```python
# Detect the first match for a buffer (e.g., PNG data)
png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
result = db.first_magic_buffer(png_data)
print(f"Detected: {result.message} (MIME: {result.mime_type})")
```

```python
# Detect the best match for a buffer
result = db.best_magic_buffer(png_data)
print(f"Best match: {result.message}")
```

```python
# Get all possible matches for a buffer
results = db.all_magics_buffer(png_data)
for r in results:
    print(f"Match: {r.message}, MIME: {r.mime_type}, Strength: {r.strength}")
```

### Detecting File Types from Files
```python
# Detect the first match for a file
result = db.first_magic_file("example.png")
print(f"File type: {result.message}, Extensions: {result.extensions}")
```

```python
# Detect the best match for a file
result = db.best_magic_file("example.png")
print(f"Best match: {result.message}")
```

```python
# Get all possible matches for a file
results = db.all_magics_file("example.png")
for r in results:
    print(f"Match: {r.message}, MIME: {r.mime_type}")
```

### Converting Results to Dictionaries
```python
result = db.first_magic_buffer(png_data)
result_dict = result.to_dict()
print(result_dict)
# Output: {'source': None, 'message': 'PNG image data', 'mime_type': 'image/png', ...}
```

### Handling Errors
```python
try:
    result = db.first_magic_file("nonexistent_file.txt")
except IOError as e:
    print(f"Error opening file: {e}")
```

## License
This project is licensed under the **GPL-3 License**.

<!-- cargo-rdme end -->
