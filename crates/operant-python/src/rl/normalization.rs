//! Running normalization using Welford's online algorithm.
//!
//! Provides SIMD-optimized observation and reward normalization for RL algorithms.

use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Running normalizer using Welford's online algorithm.
///
/// Computes running mean and variance in a numerically stable way,
/// then normalizes data to zero mean and unit variance.
///
/// Optionally clips normalized values to [-clip_range, clip_range].
#[pyclass]
pub struct RunningNormalizer {
    dim: usize,
    mean: Vec<f32>,
    m2: Vec<f32>, // Sum of squared differences for Welford's algorithm
    count: u64,
    clip_range: f32,
    eps: f32,
}

#[pymethods]
impl RunningNormalizer {
    /// Create a new running normalizer.
    ///
    /// # Arguments
    /// * `dim` - Dimension of data to normalize
    /// * `clip_range` - Clip normalized values to [-clip_range, clip_range] (0.0 = no clipping)
    /// * `eps` - Small constant for numerical stability (default: 1e-8)
    #[new]
    #[pyo3(signature = (dim, clip_range=10.0, eps=1e-8))]
    pub fn new(dim: usize, clip_range: f32, eps: f32) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("Dimension must be > 0"));
        }

        Ok(Self {
            dim,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
            count: 0,
            clip_range,
            eps,
        })
    }

    /// Update running statistics with new data.
    ///
    /// Uses Welford's online algorithm for numerically stable mean and variance:
    /// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    ///
    /// # Arguments
    /// * `data` - Flattened array of shape (batch_size * dim,) or (dim,)
    pub fn update<'py>(&mut self, data: &Bound<'py, PyArray1<f32>>) -> PyResult<()> {
        let data_slice = unsafe { data.as_slice()? };

        if data_slice.len() % self.dim != 0 {
            return Err(PyValueError::new_err(format!(
                "Data length {} not divisible by dim {}",
                data_slice.len(),
                self.dim
            )));
        }

        let batch_size = data_slice.len() / self.dim;

        // Update statistics for each sample in batch
        for batch_idx in 0..batch_size {
            self.count += 1;
            let offset = batch_idx * self.dim;

            for i in 0..self.dim {
                let x = data_slice[offset + i];
                let delta = x - self.mean[i];
                self.mean[i] += delta / self.count as f32;
                let delta2 = x - self.mean[i];
                self.m2[i] += delta * delta2;
            }
        }

        Ok(())
    }

    /// Normalize data using running statistics.
    ///
    /// Computes: (data - mean) / sqrt(var + eps)
    /// Optionally clips to [-clip_range, clip_range]
    ///
    /// # Arguments
    /// * `data` - Flattened array to normalize, shape (batch_size * dim,) or (dim,)
    ///
    /// # Returns
    /// Normalized array with same shape as input
    pub fn normalize<'py>(&self, py: Python<'py>, data: &Bound<'py, PyArray1<f32>>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let data_slice = unsafe { data.as_slice()? };

        if data_slice.len() % self.dim != 0 {
            return Err(PyValueError::new_err(format!(
                "Data length {} not divisible by dim {}",
                data_slice.len(),
                self.dim
            )));
        }

        let mut output = vec![0.0f32; data_slice.len()];

        #[cfg(feature = "simd")]
        {
            self.normalize_simd(data_slice, &mut output);
        }

        #[cfg(not(feature = "simd"))]
        {
            self.normalize_scalar(data_slice, &mut output);
        }

        Ok(output.to_pyarray(py))
    }

    /// Normalize data in-place.
    ///
    /// More efficient than `normalize()` when you don't need to keep original data.
    ///
    /// # Arguments
    /// * `data` - Mutable array to normalize in-place
    pub fn normalize_inplace<'py>(&self, data: &Bound<'py, PyArray1<f32>>) -> PyResult<()> {
        let data_slice = unsafe { data.as_slice_mut()? };

        if data_slice.len() % self.dim != 0 {
            return Err(PyValueError::new_err(format!(
                "Data length {} not divisible by dim {}",
                data_slice.len(),
                self.dim
            )));
        }

        #[cfg(feature = "simd")]
        {
            let mut temp = vec![0.0f32; data_slice.len()];
            self.normalize_simd(data_slice, &mut temp);
            data_slice.copy_from_slice(&temp);
        }

        #[cfg(not(feature = "simd"))]
        {
            self.normalize_scalar(data_slice, data_slice);
        }

        Ok(())
    }

    /// Get current mean.
    #[getter]
    pub fn get_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.mean.to_pyarray(py)
    }

    /// Get current standard deviation.
    #[getter]
    pub fn get_std<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let std: Vec<f32> = self
            .m2
            .iter()
            .map(|&m2| {
                if self.count > 1 {
                    (m2 / (self.count - 1) as f32).sqrt()
                } else {
                    1.0
                }
            })
            .collect();
        std.to_pyarray(py)
    }

    /// Get current variance.
    #[getter]
    pub fn get_var<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let var: Vec<f32> = self
            .m2
            .iter()
            .map(|&m2| {
                if self.count > 1 {
                    m2 / (self.count - 1) as f32
                } else {
                    1.0
                }
            })
            .collect();
        var.to_pyarray(py)
    }

    /// Get sample count.
    #[getter]
    pub fn get_count(&self) -> u64 {
        self.count
    }

    /// Reset statistics to initial state.
    pub fn reset(&mut self) {
        self.mean.fill(0.0);
        self.m2.fill(0.0);
        self.count = 0;
    }
}

impl RunningNormalizer {
    /// Scalar implementation of normalization (fallback for non-SIMD).
    #[allow(dead_code)]
    #[inline]
    fn normalize_scalar(&self, input: &[f32], output: &mut [f32]) {
        let batch_size = input.len() / self.dim;

        for batch_idx in 0..batch_size {
            let offset = batch_idx * self.dim;

            for i in 0..self.dim {
                let x = input[offset + i];

                let var = if self.count > 1 {
                    self.m2[i] / (self.count - 1) as f32
                } else {
                    1.0
                };

                let mut normalized = (x - self.mean[i]) / (var + self.eps).sqrt();

                if self.clip_range > 0.0 {
                    normalized = normalized.clamp(-self.clip_range, self.clip_range);
                }

                output[offset + i] = normalized;
            }
        }
    }

    /// SIMD implementation of normalization using f32x8 (AVX2).
    #[cfg(feature = "simd")]
    #[inline]
    fn normalize_simd(&self, input: &[f32], output: &mut [f32]) {
        use std::simd::{f32x8, num::SimdFloat, StdFloat};

        let batch_size = input.len() / self.dim;

        // Process 8 dimensions at a time with SIMD
        let simd_chunks = self.dim / 8;
        let _remainder = self.dim % 8;

        for batch_idx in 0..batch_size {
            let offset = batch_idx * self.dim;

            for chunk_idx in 0..simd_chunks {
                let base_idx = offset + chunk_idx * 8;

                let x = f32x8::from_slice(&input[base_idx..]);
                let mean = f32x8::from_slice(&self.mean[chunk_idx * 8..]);
                let m2 = f32x8::from_slice(&self.m2[chunk_idx * 8..]);

                let var = if self.count > 1 {
                    m2 / f32x8::splat((self.count - 1) as f32)
                } else {
                    f32x8::splat(1.0)
                };

                let mut normalized = (x - mean) / (var + f32x8::splat(self.eps)).sqrt();

                if self.clip_range > 0.0 {
                    let clip_min = f32x8::splat(-self.clip_range);
                    let clip_max = f32x8::splat(self.clip_range);
                    normalized = normalized.simd_clamp(clip_min, clip_max);
                }

                normalized.copy_to_slice(&mut output[base_idx..]);
            }

            for i in (simd_chunks * 8)..self.dim {
                let x = input[offset + i];
                let var = if self.count > 1 {
                    self.m2[i] / (self.count - 1) as f32
                } else {
                    1.0
                };

                let mut normalized = (x - self.mean[i]) / (var + self.eps).sqrt();

                if self.clip_range > 0.0 {
                    normalized = normalized.clamp(-self.clip_range, self.clip_range);
                }

                output[offset + i] = normalized;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ToPyArray;

    #[test]
    fn test_welford_algorithm() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let mut normalizer = RunningNormalizer::new(2, 0.0, 1e-8).unwrap();

            let data1 = vec![1.0f32, 2.0].to_pyarray(py);
            let data2 = vec![3.0f32, 4.0].to_pyarray(py);
            let data3 = vec![5.0f32, 6.0].to_pyarray(py);

            normalizer.update(&data1).unwrap();
            normalizer.update(&data2).unwrap();
            normalizer.update(&data3).unwrap();

            let mean = normalizer.get_mean(py);
            let mean_slice = unsafe { mean.as_slice().unwrap() };
            assert!((mean_slice[0] - 3.0).abs() < 1e-5);
            assert!((mean_slice[1] - 4.0).abs() < 1e-5);

            let std = normalizer.get_std(py);
            let std_slice = unsafe { std.as_slice().unwrap() };
            assert!((std_slice[0] - 2.0).abs() < 1e-5);
            assert!((std_slice[1] - 2.0).abs() < 1e-5);
        });
    }

    #[test]
    fn test_normalization() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let mut normalizer = RunningNormalizer::new(2, 0.0, 1e-8).unwrap();

            let data = vec![-1.0f32, -1.0, 0.0, 0.0, 1.0, 1.0].to_pyarray(py);
            normalizer.update(&data).unwrap();

            let test_data = vec![1.0f32, 1.0].to_pyarray(py);
            let normalized = normalizer.normalize(py, &test_data).unwrap();
            let normalized_slice = unsafe { normalized.as_slice().unwrap() };

            assert!((normalized_slice[0] - 1.0).abs() < 0.2);
            assert!((normalized_slice[1] - 1.0).abs() < 0.2);
        });
    }

    #[test]
    fn test_clipping() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let mut normalizer = RunningNormalizer::new(1, 2.0, 1e-8).unwrap();

            let data = vec![-1.0f32, 0.0, 1.0].to_pyarray(py);
            normalizer.update(&data).unwrap();

            let test_data = vec![10.0f32].to_pyarray(py);
            let normalized = normalizer.normalize(py, &test_data).unwrap();
            let normalized_slice = unsafe { normalized.as_slice().unwrap() };

            assert!(normalized_slice[0] <= 2.0);
            assert!(normalized_slice[0] >= -2.0);
        });
    }
}
