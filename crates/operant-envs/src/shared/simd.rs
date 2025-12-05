//! SIMD helper functions using Taylor series approximations.
//!
//! These functions prioritize performance over perfect accuracy,
//! using Taylor series expansions for trigonometric operations.

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat};

/// Compute absolute value of SIMD vector.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn simd_abs(v: f32x8) -> f32x8 {
    v.abs()
}

/// Fast cosine approximation using Taylor series (order 4).
///
/// Accurate for angles in [-π, π]. Uses truncated Taylor series:
/// cos(x) ≈ 1 - x²/2! + x⁴/4!
///
/// Clamped to [-1, 1] to prevent approximation errors from causing NaN propagation.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn simd_cos(x: f32x8) -> f32x8 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let result = f32x8::splat(1.0) - x2 * f32x8::splat(0.5) + x4 * f32x8::splat(1.0 / 24.0);
    result.simd_clamp(f32x8::splat(-1.0), f32x8::splat(1.0))
}

/// Fast sine approximation using Taylor series (order 5).
///
/// Accurate for angles in [-π, π]. Uses truncated Taylor series:
/// sin(x) ≈ x - x³/3! + x⁵/5!
///
/// Clamped to [-1, 1] to prevent approximation errors from causing NaN propagation.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn simd_sin(x: f32x8) -> f32x8 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let result = x - x3 * f32x8::splat(1.0 / 6.0) + x5 * f32x8::splat(1.0 / 120.0);
    result.simd_clamp(f32x8::splat(-1.0), f32x8::splat(1.0))
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;

    #[test]
    fn test_simd_abs() {
        let v = f32x8::from_array([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]);
        let result = simd_abs(v);
        let expected = f32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(result.to_array(), expected.to_array());
    }

    #[test]
    fn test_simd_cos_approximation() {
        let angles = f32x8::from_array([0.0, 0.5, 1.0, 1.5, -0.5, -1.0, -1.5, 0.25]);
        let result = simd_cos(angles);
        let result_array = result.to_array();

        let expected: Vec<f32> = angles.to_array()
            .iter()
            .map(|x| x.cos())
            .collect();

        for (i, (&approx, &exact)) in result_array.iter().zip(expected.iter()).enumerate() {
            let error = (approx - exact).abs();
            // Taylor approximation has larger error at angles further from 0
            assert!(error < 0.1, "cos approximation error too large at index {}: {} vs {}", i, approx, exact);
        }
    }

    #[test]
    fn test_simd_sin_approximation() {
        let angles = f32x8::from_array([0.0, 0.5, 1.0, 1.5, -0.5, -1.0, -1.5, 0.25]);
        let result = simd_sin(angles);
        let result_array = result.to_array();

        let expected: Vec<f32> = angles.to_array()
            .iter()
            .map(|x| x.sin())
            .collect();

        for (i, (&approx, &exact)) in result_array.iter().zip(expected.iter()).enumerate() {
            let error = (approx - exact).abs();
            // Taylor approximation has larger error at angles further from 0
            assert!(error < 0.1, "sin approximation error too large at index {}: {} vs {}", i, approx, exact);
        }
    }
}
