//! Random number generation utilities for environments.

pub use rand_xoshiro::Xoshiro256StarStar;

/// Generate a random float in the range [low, high).
#[inline]
pub fn random_uniform(rng: &mut Xoshiro256StarStar, low: f32, high: f32) -> f32 {
    use rand::Rng;
    rng.gen::<f32>() * (high - low) + low
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_random_uniform() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        let value = random_uniform(&mut rng, -1.0, 1.0);
        assert!(value >= -1.0 && value < 1.0);
    }

    #[test]
    fn test_random_uniform_range() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        for _ in 0..100 {
            let value = random_uniform(&mut rng, 5.0, 10.0);
            assert!(value >= 5.0 && value < 10.0);
        }
    }
}
