//! Zero-copy buffer protocol implementation for PyTorch interop.
//!
//! Implements PEP 3118 buffer protocol so PyTorch can consume data directly
//! without numpy dependency. Uses Arc<Vec<f32>> for true zero-copy sharing
//! between Rust and Python with proper metadata (dtype, shape, strides).

use pyo3::prelude::*;
use std::sync::Arc;

/// 1D buffer for rewards, values, etc.
///
/// Implements buffer protocol for zero-copy PyTorch consumption.
/// Uses Arc for shared ownership - no cloning required!
#[pyclass]
pub struct Buffer1D {
    data: Arc<Vec<f32>>,
}

#[pymethods]
impl Buffer1D {
    /// Create from vec
    #[staticmethod]
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data: Arc::new(data) }
    }

    /// Shape for array-like interface
    #[getter]
    fn shape(&self) -> (usize,) {
        (self.data.len(),)
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    /// Convert to list (fallback for torch.tensor())
    fn tolist(&self) -> Vec<f32> {
        (*self.data).clone()
    }

    /// Provide numpy-like array interface for PyTorch
    /// Returns self to allow torch.from_numpy() style access
    fn __array__<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<PyObject> {
        // Create a numpy-compatible array interface dict
        // PyTorch will use this to access the data
        let dict = pyo3::types::PyDict::new(py);

        // Data pointer - get pointer to Vec's data, not Arc metadata!
        let ptr = self.data.as_ptr() as usize;
        dict.set_item("data", (ptr, false))?; // (ptr, read_only)

        // Shape
        dict.set_item("shape", (self.data.len(),))?;

        // Strides (contiguous f32)
        dict.set_item("strides", (4,))?; // 4 bytes per f32

        // Dtype
        dict.set_item("typestr", "<f4")?; // little-endian 4-byte float

        // Version
        dict.set_item("version", 3)?;

        Ok(dict.into())
    }

    /// NumPy array interface attribute
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        self.__array__(py, None)
    }

    /// Convert to numpy array (uses numpy.asarray via array interface)
    fn numpy<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyObject> {
        // Import numpy and call asarray on self
        let np = py.import("numpy")?;
        let asarray = np.getattr("asarray")?;
        let result = asarray.call1((slf,))?;
        Ok(result.into())
    }
}

impl Buffer1D {
    /// Create from Arc'd vec (zero-copy, Rust-only API)
    pub fn from_vec_arc(data: Arc<Vec<f32>>) -> Self {
        Self { data }
    }
}

/// 2D buffer for observations, actions, etc.
///
/// Implements buffer protocol with 2D shape metadata.
#[pyclass]
pub struct Buffer2D {
    data: Arc<Vec<f32>>,
    shape: (usize, usize),  // (rows, cols)
}

#[pymethods]
impl Buffer2D {
    /// Create from flat vec with shape
    #[staticmethod]
    pub fn from_flat(data: Vec<f32>, rows: usize, cols: usize) -> PyResult<Self> {
        if data.len() != rows * cols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Data length {} != rows {} * cols {}", data.len(), rows, cols)
            ));
        }
        Ok(Self {
            data: Arc::new(data),
            shape: (rows, cols),
        })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn __len__(&self) -> usize {
        self.shape.0  // Number of rows
    }

    /// Provide numpy-like array interface for PyTorch
    fn __array__<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);

        // Data pointer - get pointer to Vec's data, not Arc metadata!
        let ptr = self.data.as_ptr() as usize;
        dict.set_item("data", (ptr, false))?;

        // Shape (rows, cols)
        dict.set_item("shape", self.shape)?;

        // Strides (row-major: cols*4 bytes per row, 4 bytes per element)
        dict.set_item("strides", (self.shape.1 * 4, 4))?;

        // Dtype
        dict.set_item("typestr", "<f4")?; // little-endian 4-byte float

        // Version
        dict.set_item("version", 3)?;

        Ok(dict.into())
    }

    /// NumPy array interface attribute
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        self.__array__(py, None)
    }

    /// Convert to numpy array (uses numpy.asarray via array interface)
    fn numpy<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let asarray = np.getattr("asarray")?;
        let result = asarray.call1((slf,))?;
        Ok(result.into())
    }
}

impl Buffer2D {
    /// Create from Arc'd vec with shape (zero-copy, Rust-only API)
    pub fn from_flat_arc(data: Arc<Vec<f32>>, rows: usize, cols: usize) -> operant_core::Result<Self> {
        if data.len() != rows * cols {
            return Err(operant_core::OperantError::BufferSizeMismatch {
                expected: rows * cols,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            shape: (rows, cols),
        })
    }
}

/// 3D buffer for batch operations
#[pyclass]
pub struct Buffer3D {
    data: Arc<Vec<f32>>,
    shape: (usize, usize, usize),  // (depth, rows, cols)
}

#[pymethods]
impl Buffer3D {
    /// Create from flat vec with shape
    #[staticmethod]
    pub fn from_flat(data: Vec<f32>, depth: usize, rows: usize, cols: usize) -> PyResult<Self> {
        if data.len() != depth * rows * cols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Data length {} != depth {} * rows {} * cols {}", data.len(), depth, rows, cols)
            ));
        }
        Ok(Self {
            data: Arc::new(data),
            shape: (depth, rows, cols),
        })
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    fn __len__(&self) -> usize {
        self.shape.0  // Depth dimension
    }

    /// Provide numpy-like array interface for PyTorch
    fn __array__<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);

        // Data pointer - get pointer to Vec's data, not Arc metadata!
        let ptr = self.data.as_ptr() as usize;
        dict.set_item("data", (ptr, false))?;

        // Shape (depth, rows, cols)
        dict.set_item("shape", self.shape)?;

        // Strides (row-major)
        let stride_depth = self.shape.1 * self.shape.2 * 4;
        let stride_rows = self.shape.2 * 4;
        let stride_cols = 4;
        dict.set_item("strides", (stride_depth, stride_rows, stride_cols))?;

        // Dtype
        dict.set_item("typestr", "<f4")?;

        // Version
        dict.set_item("version", 3)?;

        Ok(dict.into())
    }

    /// NumPy array interface attribute
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        self.__array__(py, None)
    }

    /// Convert to numpy array (uses numpy.asarray via array interface)
    fn numpy<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let asarray = np.getattr("asarray")?;
        let result = asarray.call1((slf,))?;
        Ok(result.into())
    }
}

impl Buffer3D {
    /// Create from Arc'd vec with shape (zero-copy, Rust-only API)
    pub fn from_flat_arc(data: Arc<Vec<f32>>, depth: usize, rows: usize, cols: usize) -> operant_core::Result<Self> {
        if data.len() != depth * rows * cols {
            return Err(operant_core::OperantError::BufferSizeMismatch {
                expected: depth * rows * cols,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            shape: (depth, rows, cols),
        })
    }
}
