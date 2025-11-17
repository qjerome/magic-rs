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
