/**
 * GPU-accelerated CartPole environment kernel
 *
 * Physics based on OpenAI Gym CartPole-v1
 * https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
 *
 * Each thread handles one environment instance.
 */

extern "C" {

// Constants (matching CPU implementation)
const float GRAVITY = 9.8f;
const float MASS_CART = 1.0f;
const float MASS_POLE = 0.1f;
const float TOTAL_MASS = MASS_CART + MASS_POLE;
const float LENGTH = 0.5f;  // Actually half the pole's length
const float POLE_MASS_LENGTH = MASS_POLE * LENGTH;
const float FORCE_MAG = 10.0f;
const float TAU = 0.02f;  // Seconds between state updates

// Thresholds for termination
const float THETA_THRESHOLD = 12.0f * (3.14159265359f / 180.0f);  // ~0.2095 radians (12 degrees)
const float X_THRESHOLD = 2.4f;

/**
 * Reset kernel - initialize environments to random states
 *
 * @param states: [num_envs, 4] - output state buffer (x, x_dot, theta, theta_dot)
 * @param rng_states: [num_envs] - RNG state for each environment
 * @param num_envs: number of parallel environments
 */
__global__ void cartpole_reset(
    float* states,
    unsigned int* rng_states,
    int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    // Simple LCG random number generator (fast on GPU)
    unsigned int seed = rng_states[idx];

    // Generate random state in range [-0.05, 0.05]
    auto rand_uniform = [&seed]() -> float {
        seed = seed * 1103515245u + 12345u;
        return (float)(seed & 0x7FFFFFFFu) / (float)0x7FFFFFFFu * 0.1f - 0.05f;
    };

    int base = idx * 4;
    states[base + 0] = rand_uniform();  // x
    states[base + 1] = rand_uniform();  // x_dot
    states[base + 2] = rand_uniform();  // theta
    states[base + 3] = rand_uniform();  // theta_dot

    // Update RNG state
    rng_states[idx] = seed;
}

/**
 * Step kernel - advance physics simulation for all environments
 *
 * @param states: [num_envs, 4] - state buffer (x, x_dot, theta, theta_dot) [IN/OUT]
 * @param actions: [num_envs] - discrete actions (0=left, 1=right)
 * @param rewards: [num_envs] - output rewards (1.0 if not terminal)
 * @param terminals: [num_envs] - output terminal flags (1.0 if done)
 * @param truncations: [num_envs] - output truncation flags (always 0.0 for CartPole)
 * @param num_envs: number of parallel environments
 */
__global__ void cartpole_step(
    float* states,
    const int* actions,
    float* rewards,
    float* terminals,
    float* truncations,
    int num_envs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    int base = idx * 4;

    // Load state
    float x = states[base + 0];
    float x_dot = states[base + 1];
    float theta = states[base + 2];
    float theta_dot = states[base + 3];

    // Apply action force
    float force = (actions[idx] == 1) ? FORCE_MAG : -FORCE_MAG;

    // Physics calculations
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    // Temporary calculation for acceleration
    float temp = (force + POLE_MASS_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS;
    float theta_acc = (GRAVITY * sin_theta - cos_theta * temp) /
                      (LENGTH * (4.0f/3.0f - MASS_POLE * cos_theta * cos_theta / TOTAL_MASS));
    float x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

    // Update state using Euler integration
    x = x + TAU * x_dot;
    x_dot = x_dot + TAU * x_acc;
    theta = theta + TAU * theta_dot;
    theta_dot = theta_dot + TAU * theta_acc;

    // Check termination conditions
    bool done = (x < -X_THRESHOLD) || (x > X_THRESHOLD) ||
                (theta < -THETA_THRESHOLD) || (theta > THETA_THRESHOLD);

    // Write outputs
    states[base + 0] = x;
    states[base + 1] = x_dot;
    states[base + 2] = theta;
    states[base + 3] = theta_dot;

    rewards[idx] = done ? 0.0f : 1.0f;
    terminals[idx] = done ? 1.0f : 0.0f;
    truncations[idx] = 0.0f;  // CartPole doesn't use truncation
}

}  // extern "C"
