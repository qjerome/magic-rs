//! # Pymagic-rs: Pure Rust Python Bindings for File Type Detection
//!
//! `pymagic-rs` is a Python module that provides bindings to the [`magic-rs`](https://crates.io/crates/magic-rs) crate, allowing you to detect file types, MIME types, and other metadata using a pure Rust alternative to the C `libmagic` library. This module uses an **embedded magic database**, so everything works out-of-the-box with **no need** of a compiled database or magic rule files.
//!
//! ## Features
//! - **Pure Rust implementation**: No C dependencies, easily cross-compilable for great compatibility
//! - **Embedded magic database**: No database file or magic rules needed
//! - Detect file types from buffers or files
//! - Retrieve MIME types, creator codes, and file extensions
//! - Convert results to Python dictionaries for easy integration
//! - Supports both "first match", "best match" and "all matches" detection strategies
//!
//! ## Installation
//! ```bash
//! pip install pymagic-rs
//! ```
//!
//! ## Usage
//!
//! ### Initializing the Database
//! ```python
//! from pymagic import MagicDb
//! # Initialize the Magic database (uses embedded database)
//! db = MagicDb()
//! ```
//!
//! ### Detecting File Types from Buffers
//! ```python
//! # Detect the first match for a buffer (e.g., PNG data)
//! png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
//! result = db.first_magic_buffer(png_data)
//! print(f"Detected: {result.message} (MIME: {result.mime_type})")
//! ```
//!
//! ```python
//! # Detect the best match for a buffer
//! result = db.best_magic_buffer(png_data)
//! print(f"Best match: {result.message}")
//! ```
//!
//! ```python
//! # Get all possible matches for a buffer
//! results = db.all_magics_buffer(png_data)
//! for r in results:
//!     print(f"Match: {r.message}, MIME: {r.mime_type}, Strength: {r.strength}")
//! ```
//!
//! ### Detecting File Types from Files
//! ```python
//! # Detect the first match for a file
//! result = db.first_magic_file("example.png")
//! print(f"File type: {result.message}, Extensions: {result.extensions}")
//! ```
//!
//! ```python
//! # Detect the best match for a file
//! result = db.best_magic_file("example.png")
//! print(f"Best match: {result.message}")
//! ```
//!
//! ```python
//! # Get all possible matches for a file
//! results = db.all_magics_file("example.png")
//! for r in results:
//!     print(f"Match: {r.message}, MIME: {r.mime_type}")
//! ```
//!
//! ### Converting Results to Dictionaries
//! ```python
//! result = db.first_magic_buffer(png_data)
//! result_dict = result.to_dict()
//! print(result_dict)
//! # Output: {'source': None, 'message': 'PNG image data', 'mime_type': 'image/png', ...}
//! ```
//!
//! ### Handling Errors
//! ```python
//! try:
//!     result = db.first_magic_file("nonexistent_file.txt")
//! except IOError as e:
//!     print(f"Error opening file: {e}")
//! ```
//!
//! ## License
//! This project is licensed under the **GPL-3 License**.

use std::fs::File;
use std::io::{self};
use std::path::PathBuf;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn py_value_err(from: magic_rs::Error) -> PyErr {
    PyValueError::new_err(from.to_string())
}

#[pyclass]
pub struct Magic {
    #[pyo3(get)]
    source: Option<String>,
    #[pyo3(get)]
    message: String,
    #[pyo3(get)]
    mime_type: String,
    #[pyo3(get)]
    creator_code: Option<String>,
    #[pyo3(get)]
    strength: u64,
    #[pyo3(get)]
    extensions: Vec<String>,
}

impl From<magic_rs::Magic<'_>> for Magic {
    fn from(value: magic_rs::Magic<'_>) -> Self {
        Self {
            source: value.source().map(|s| s.to_string()),
            message: value.message(),
            mime_type: value.mime_type().to_string(),
            creator_code: value.creator_code().map(|c| c.to_string()),
            strength: value.strength(),
            extensions: value.extensions().iter().map(|e| e.to_string()).collect(),
        }
    }
}

#[pymethods]
impl Magic {
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = PyDict::new(py);
        m.set_item("source", self.source.clone())?;
        m.set_item("message", self.message.clone())?;
        m.set_item("mime_type", self.mime_type.clone())?;
        m.set_item("creator_code", self.creator_code.clone())?;
        m.set_item("strength", self.strength)?;
        m.set_item("extensions", self.extensions.clone())?;
        Ok(m)
    }
}

#[pyclass]
struct MagicDb(magic_rs::MagicDb);

#[pymethods]
impl MagicDb {
    #[new]
    pub fn new() -> PyResult<Self> {
        magic_db::CompiledDb::open().map(Self).map_err(py_value_err)
    }

    pub fn first_magic_buffer(&self, input: &[u8], extension: Option<&str>) -> PyResult<Magic> {
        let mut cursor = io::Cursor::new(input);
        self.0
            .magic_first(&mut cursor, extension)
            .map(Magic::from)
            .map_err(py_value_err)
    }

    pub fn first_magic_file(&self, path: PathBuf) -> PyResult<Magic> {
        let mut file = File::open(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let ext = path.extension();
        self.0
            .magic_first(&mut file, ext.and_then(|e| e.to_str()))
            .map(Magic::from)
            .map_err(py_value_err)
    }

    pub fn best_magic_buffer(&self, input: &[u8]) -> PyResult<Magic> {
        let mut cursor = io::Cursor::new(input);
        self.0
            .magic_best(&mut cursor)
            .map(Magic::from)
            .map_err(py_value_err)
    }

    pub fn best_magic_file(&self, path: PathBuf) -> PyResult<Magic> {
        let mut file = File::open(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.0
            .magic_best(&mut file)
            .map(Magic::from)
            .map_err(py_value_err)
    }

    pub fn all_magics_buffer(&self, input: &[u8]) -> PyResult<Vec<Magic>> {
        let mut cursor = io::Cursor::new(input);
        self.0
            .magic_all(&mut cursor)
            .map(|magics| magics.into_iter().map(Magic::from).collect())
            .map_err(py_value_err)
    }

    pub fn all_magics_file(&self, path: PathBuf) -> PyResult<Vec<Magic>> {
        let mut file = File::open(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.0
            .magic_all(&mut file)
            .map(|magics| magics.into_iter().map(Magic::from).collect())
            .map_err(py_value_err)
    }
}

#[pymodule]
fn pymagic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Magic>()?;
    m.add_class::<MagicDb>()?;

    Ok(())
}
