//! Ratatui-based TUI logger for training metrics visualization.
//!
//! Provides a high-performance terminal UI with minimal overhead on the training loop.
//! Data is passed via atomic writes to shared memory, with a separate render thread.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{stdout, BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Sparkline},
    Frame, Terminal,
};

/// Metrics buffer for lock-free updates from Python.
/// Uses atomic types for thread-safe access without locks.
#[derive(Default)]
pub struct MetricsBuffer {
    pub steps: AtomicU64,
    pub episodes: AtomicU64,
    pub mean_reward_bits: AtomicU64,
    pub sps_bits: AtomicU64,
    pub policy_loss_bits: AtomicU64,
    pub value_loss_bits: AtomicU64,
    pub entropy_bits: AtomicU64,
    // Device monitoring metrics (Rust-side automatic + Python optional)
    pub cpu_usage_bits: AtomicU64,        // CPU percentage (0-100)
    pub gpu_usage_bits: AtomicU64,        // GPU percentage (0-100, 0 if not provided)
    pub memory_usage_mb: AtomicU64,       // RAM usage in MB
    pub gpu_memory_mb: AtomicU64,         // VRAM usage in MB (0 if not provided)
    pub total_memory_mb: AtomicU64,       // Total system RAM in MB (static)
    pub total_gpu_memory_mb: AtomicU64,   // Total VRAM in MB (0 if not provided)
}

impl MetricsBuffer {
    fn get_steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }

    fn get_episodes(&self) -> u64 {
        self.episodes.load(Ordering::Relaxed)
    }

    fn get_mean_reward(&self) -> f64 {
        f64::from_bits(self.mean_reward_bits.load(Ordering::Relaxed))
    }

    fn get_sps(&self) -> f64 {
        f64::from_bits(self.sps_bits.load(Ordering::Relaxed))
    }

    fn get_policy_loss(&self) -> f64 {
        f64::from_bits(self.policy_loss_bits.load(Ordering::Relaxed))
    }

    fn get_value_loss(&self) -> f64 {
        f64::from_bits(self.value_loss_bits.load(Ordering::Relaxed))
    }

    fn get_entropy(&self) -> f64 {
        f64::from_bits(self.entropy_bits.load(Ordering::Relaxed))
    }

    fn get_cpu_usage(&self) -> f64 {
        f64::from_bits(self.cpu_usage_bits.load(Ordering::Relaxed))
    }

    fn get_gpu_usage(&self) -> f64 {
        f64::from_bits(self.gpu_usage_bits.load(Ordering::Relaxed))
    }

    fn get_memory_usage_mb(&self) -> u64 {
        self.memory_usage_mb.load(Ordering::Relaxed)
    }

    fn get_gpu_memory_mb(&self) -> u64 {
        self.gpu_memory_mb.load(Ordering::Relaxed)
    }

    fn get_total_memory_mb(&self) -> u64 {
        self.total_memory_mb.load(Ordering::Relaxed)
    }

    fn get_total_gpu_memory_mb(&self) -> u64 {
        self.total_gpu_memory_mb.load(Ordering::Relaxed)
    }
}

/// History buffer for sparkline visualization.
struct HistoryBuffer {
    steps: VecDeque<u64>,      // Step numbers for x-axis labels
    rewards: VecDeque<f64>,
    sps: VecDeque<f64>,
    policy_loss: VecDeque<f64>,
    value_loss: VecDeque<f64>,
    cpu: VecDeque<f64>,        // CPU usage % history
    gpu: VecDeque<f64>,        // GPU usage % history
    memory: VecDeque<f64>,     // RAM usage % history
    gpu_memory: VecDeque<f64>, // VRAM usage % history
    max_len: usize,
}

impl HistoryBuffer {
    fn new(max_len: usize) -> Self {
        Self {
            steps: VecDeque::with_capacity(max_len),
            rewards: VecDeque::with_capacity(max_len),
            sps: VecDeque::with_capacity(max_len),
            policy_loss: VecDeque::with_capacity(max_len),
            value_loss: VecDeque::with_capacity(max_len),
            cpu: VecDeque::with_capacity(max_len),
            gpu: VecDeque::with_capacity(max_len),
            memory: VecDeque::with_capacity(max_len),
            gpu_memory: VecDeque::with_capacity(max_len),
            max_len,
        }
    }

    fn push(
        &mut self,
        step: u64,
        reward: f64,
        sps: f64,
        policy_loss: f64,
        value_loss: f64,
        cpu: f64,
        gpu: f64,
        memory: f64,
        gpu_memory: f64,
    ) {
        if self.steps.len() >= self.max_len {
            self.steps.pop_front();
            self.rewards.pop_front();
            self.sps.pop_front();
            self.policy_loss.pop_front();
            self.value_loss.pop_front();
            self.cpu.pop_front();
            self.gpu.pop_front();
            self.memory.pop_front();
            self.gpu_memory.pop_front();
        }
        self.steps.push_back(step);
        self.rewards.push_back(reward);
        self.sps.push_back(sps);
        self.policy_loss.push_back(policy_loss);
        self.value_loss.push_back(value_loss);
        self.cpu.push_back(cpu);
        self.gpu.push_back(gpu);
        self.memory.push_back(memory);
        self.gpu_memory.push_back(gpu_memory);
    }
}

/// Display mode for the TUI.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TUIMode {
    Minimal,
    Dashboard,
    ChartDashboard,
}

/// Shared state between Python and render thread.
struct SharedState {
    buffer: Arc<MetricsBuffer>,
    history: Arc<Mutex<HistoryBuffer>>,
    running: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    mode: Arc<Mutex<TUIMode>>,
    quit_requested: Arc<AtomicBool>,
}

/// Configuration for system monitoring thread.
struct MonitorConfig {
    buffer: Arc<MetricsBuffer>,
    running: Arc<AtomicBool>,
    sample_interval_ms: u64,
}

/// Background thread for automatic CPU/RAM monitoring using sysinfo.
#[cfg(feature = "tui")]
fn spawn_monitor_thread(config: MonitorConfig) -> JoinHandle<()> {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

    thread::spawn(move || {
        let mut sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        let sample_interval = Duration::from_millis(config.sample_interval_ms);

        // Get total memory once (static value)
        sys.refresh_memory();
        let total_mem_mb = sys.total_memory() / 1024 / 1024;
        config.buffer.total_memory_mb.store(total_mem_mb, Ordering::Relaxed);

        // Try to initialize NVML for GPU monitoring (optional - graceful failure)
        let nvml = nvml_wrapper::Nvml::init().ok();
        let device = nvml.as_ref()
            .and_then(|n| n.device_by_index(0).ok());

        // Get total GPU memory once if available (static value)
        if let Some(dev) = &device {
            if let Ok(mem) = dev.memory_info() {
                let total_gpu_mb = mem.total / 1024 / 1024;
                config.buffer.total_gpu_memory_mb.store(total_gpu_mb, Ordering::Relaxed);
            }
        }

        while config.running.load(Ordering::SeqCst) {
            // Refresh system stats
            sys.refresh_cpu_all();
            sys.refresh_memory();

            // Calculate CPU usage (average across all cores)
            let cpu_usage = sys.global_cpu_usage() as f64;

            // Calculate memory usage
            let used_mem_mb = sys.used_memory() / 1024 / 1024;

            // Update atomic buffer for CPU/RAM
            config.buffer.cpu_usage_bits.store(cpu_usage.to_bits(), Ordering::Relaxed);
            config.buffer.memory_usage_mb.store(used_mem_mb, Ordering::Relaxed);

            // GPU monitoring (optional - no panic if unavailable)
            if let Some(dev) = &device {
                // Get GPU utilization
                if let Ok(util) = dev.utilization_rates() {
                    config.buffer.gpu_usage_bits.store(
                        (util.gpu as f64).to_bits(),
                        Ordering::Relaxed,
                    );
                }

                // Get GPU memory
                if let Ok(mem) = dev.memory_info() {
                    let used_mb = mem.used / 1024 / 1024;
                    config.buffer.gpu_memory_mb.store(
                        used_mb,
                        Ordering::Relaxed,
                    );
                }
            }

            thread::sleep(sample_interval);
        }
    })
}

/// High-performance TUI logger with ratatui.
///
/// Renders training metrics in a terminal UI with minimal overhead.
/// Python updates metrics via atomic writes; a separate thread handles rendering.
#[pyclass]
pub struct TUILogger {
    state: SharedState,
    render_thread: Option<JoinHandle<()>>,
    monitor_thread: Option<JoinHandle<()>>,
    csv_writer: Option<Arc<Mutex<BufWriter<File>>>>,
    start_time: Instant,
    last_display_update: AtomicU64, // Last step that updated display
    update_interval: u64,             // Steps between display updates
}

#[pymethods]
impl TUILogger {
    /// Create a new TUI logger.
    ///
    /// # Arguments
    ///
    /// * `mode` - Display mode: "minimal", "dashboard", or "chart"
    /// * `render_interval_ms` - How often to refresh the display (default: 100ms)
    /// * `history_len` - Number of data points for sparklines (default: 100)
    /// * `csv_path` - Optional path to write CSV log file
    /// * `monitor_interval_ms` - How often to sample CPU/RAM (default: 1000ms)
    /// * `update_interval_steps` - Steps between display updates (default: 10000)
    #[new]
    #[pyo3(signature = (mode="minimal", render_interval_ms=100, history_len=100, csv_path=None, monitor_interval_ms=None, update_interval_steps=None))]
    pub fn new(
        mode: &str,
        render_interval_ms: u64,
        history_len: usize,
        csv_path: Option<&str>,
        monitor_interval_ms: Option<u64>,
        update_interval_steps: Option<u64>,
    ) -> PyResult<Self> {
        let tui_mode = match mode {
            "minimal" => TUIMode::Minimal,
            "dashboard" => TUIMode::Dashboard,
            "chart" => TUIMode::ChartDashboard,
            _ => {
                return Err(PyValueError::new_err(
                    "mode must be 'minimal', 'dashboard', or 'chart'",
                ))
            }
        };

        let buffer = Arc::new(MetricsBuffer::default());
        let history = Arc::new(Mutex::new(HistoryBuffer::new(history_len)));
        let running = Arc::new(AtomicBool::new(true));
        let paused = Arc::new(AtomicBool::new(false));
        let mode_arc = Arc::new(Mutex::new(tui_mode));
        let quit_requested = Arc::new(AtomicBool::new(false));

        // Set up CSV writer if path provided
        let csv_writer = if let Some(path) = csv_path {
            let file = File::create(path)
                .map_err(|e| PyValueError::new_err(format!("Failed to create CSV file: {}", e)))?;
            let mut writer = BufWriter::new(file);
            // Write header
            writeln!(
                writer,
                "timestamp,steps,episodes,mean_reward,sps,policy_loss,value_loss,entropy,cpu_usage,gpu_usage,memory_mb,gpu_memory_mb"
            )
            .map_err(|e| PyValueError::new_err(format!("Failed to write CSV header: {}", e)))?;
            Some(Arc::new(Mutex::new(writer)))
        } else {
            None
        };

        let state = SharedState {
            buffer: Arc::clone(&buffer),
            history: Arc::clone(&history),
            running: Arc::clone(&running),
            paused: Arc::clone(&paused),
            mode: Arc::clone(&mode_arc),
            quit_requested: Arc::clone(&quit_requested),
        };

        // Spawn system monitor thread
        let monitor_interval = monitor_interval_ms.unwrap_or(1000); // Default 1s
        let monitor_config = MonitorConfig {
            buffer: Arc::clone(&buffer),
            running: Arc::clone(&running),
            sample_interval_ms: monitor_interval,
        };
        let monitor_thread = spawn_monitor_thread(monitor_config);

        // Spawn render thread
        let render_state = SharedState {
            buffer: Arc::clone(&buffer),
            history: Arc::clone(&history),
            running: Arc::clone(&running),
            paused: Arc::clone(&paused),
            mode: Arc::clone(&mode_arc),
            quit_requested: Arc::clone(&quit_requested),
        };

        let render_thread = thread::spawn(move || {
            if let Err(e) = run_tui(render_state, render_interval_ms) {
                eprintln!("TUI error: {}", e);
            }
        });

        Ok(Self {
            state,
            render_thread: Some(render_thread),
            monitor_thread: Some(monitor_thread),
            csv_writer,
            start_time: Instant::now(),
            last_display_update: AtomicU64::new(0),
            update_interval: update_interval_steps.unwrap_or(10000),
        })
    }

    /// Update metrics from Python training loop.
    ///
    /// This method is optimized for minimal overhead - uses atomic writes only.
    /// GPU metrics are optional and can be provided by Python if available.
    #[pyo3(signature = (steps, episodes, mean_reward, sps, policy_loss=None, value_loss=None, entropy=None, gpu_usage=None, gpu_memory=None, total_gpu_memory=None))]
    pub fn update(
        &self,
        steps: u64,
        episodes: u64,
        mean_reward: f64,
        sps: f64,
        policy_loss: Option<f64>,
        value_loss: Option<f64>,
        entropy: Option<f64>,
        gpu_usage: Option<f64>,        // GPU utilization % (0-100)
        gpu_memory: Option<f64>,       // GPU memory used in MB
        total_gpu_memory: Option<f64>, // Total GPU memory in MB
    ) {
        let policy_loss = policy_loss.unwrap_or(0.0);
        let value_loss = value_loss.unwrap_or(0.0);
        let entropy = entropy.unwrap_or(0.0);

        // Atomic updates - no locks on hot path
        self.state.buffer.steps.store(steps, Ordering::Relaxed);
        self.state.buffer.episodes.store(episodes, Ordering::Relaxed);
        self.state
            .buffer
            .mean_reward_bits
            .store(mean_reward.to_bits(), Ordering::Relaxed);
        self.state
            .buffer
            .sps_bits
            .store(sps.to_bits(), Ordering::Relaxed);
        self.state
            .buffer
            .policy_loss_bits
            .store(policy_loss.to_bits(), Ordering::Relaxed);
        self.state
            .buffer
            .value_loss_bits
            .store(value_loss.to_bits(), Ordering::Relaxed);
        self.state
            .buffer
            .entropy_bits
            .store(entropy.to_bits(), Ordering::Relaxed);

        // Update GPU metrics if provided (Python optional)
        if let Some(gpu_pct) = gpu_usage {
            self.state
                .buffer
                .gpu_usage_bits
                .store(gpu_pct.to_bits(), Ordering::Relaxed);
        }
        if let Some(gpu_mem) = gpu_memory {
            self.state
                .buffer
                .gpu_memory_mb
                .store(gpu_mem as u64, Ordering::Relaxed);
        }
        if let Some(total_gpu_mem) = total_gpu_memory {
            self.state
                .buffer
                .total_gpu_memory_mb
                .store(total_gpu_mem as u64, Ordering::Relaxed);
        }

        // Step-based throttling for display updates
        let last_update = self.last_display_update.load(Ordering::Relaxed);
        let should_update_display = steps.saturating_sub(last_update) >= self.update_interval;

        // Update history only when display should update (throttled)
        // This reduces CPU overhead and visual noise from constant updates
        if should_update_display {
            if let Ok(mut history) = self.state.history.try_lock() {
            let cpu = self.state.buffer.get_cpu_usage();
            let gpu = self.state.buffer.get_gpu_usage();

            let total_mem = self.state.buffer.get_total_memory_mb() as f64;
            let used_mem = self.state.buffer.get_memory_usage_mb() as f64;
            let mem_pct = if total_mem > 0.0 {
                (used_mem / total_mem) * 100.0
            } else {
                0.0
            };

            let total_gpu_mem = self.state.buffer.get_total_gpu_memory_mb() as f64;
            let used_gpu_mem = self.state.buffer.get_gpu_memory_mb() as f64;
            let gpu_mem_pct = if total_gpu_mem > 0.0 {
                (used_gpu_mem / total_gpu_mem) * 100.0
            } else {
                0.0
            };

            history.push(
                steps,
                mean_reward,
                sps,
                policy_loss,
                value_loss,
                cpu,
                gpu,
                mem_pct,
                gpu_mem_pct,
            );

                // Update last display step
                self.last_display_update.store(steps, Ordering::Relaxed);
            }
        }

        // Write to CSV if configured (no throttling for data logging)
        if let Some(ref writer) = self.csv_writer {
            if let Ok(mut w) = writer.try_lock() {
                let cpu = self.state.buffer.get_cpu_usage();
                let gpu = self.state.buffer.get_gpu_usage();
                let mem = self.state.buffer.get_memory_usage_mb();
                let gpu_mem = self.state.buffer.get_gpu_memory_mb();

                let _ = writeln!(
                    w,
                    "{:.3},{},{},{:.4},{:.0},{:.6},{:.6},{:.6},{:.2},{:.2},{},{}",
                    self.start_time.elapsed().as_secs_f64(),
                    steps,
                    episodes,
                    mean_reward,
                    sps,
                    policy_loss,
                    value_loss,
                    entropy,
                    cpu,
                    gpu,
                    mem,
                    gpu_mem
                );
            }
        }
    }

    /// Check if quit was requested via keyboard.
    pub fn should_quit(&self) -> bool {
        self.state.quit_requested.load(Ordering::Relaxed)
    }

    /// Close the TUI and restore terminal.
    pub fn close(&mut self) {
        self.state.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.render_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.monitor_thread.take() {
            let _ = handle.join();
        }
        // Flush CSV
        if let Some(ref writer) = self.csv_writer {
            if let Ok(mut w) = writer.lock() {
                let _ = w.flush();
            }
        }
    }
}

impl Drop for TUILogger {
    fn drop(&mut self) {
        self.close();
    }
}

/// Main TUI render loop (runs in separate thread).
fn run_tui(state: SharedState, render_interval_ms: u64) -> std::io::Result<()> {
    // Set up terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(ratatui::backend::CrosstermBackend::new(stdout()))?;

    let render_interval = Duration::from_millis(render_interval_ms);
    let mut last_render = Instant::now();

    while state.running.load(Ordering::SeqCst) {
        // Handle keyboard input
        if event::poll(Duration::from_millis(10))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => {
                            state.quit_requested.store(true, Ordering::SeqCst);
                            break;
                        }
                        KeyCode::Char('p') => {
                            let current = state.paused.load(Ordering::Relaxed);
                            state.paused.store(!current, Ordering::Relaxed);
                        }
                        KeyCode::Char('m') => {
                            if let Ok(mut mode) = state.mode.lock() {
                                *mode = match *mode {
                                    TUIMode::Minimal => TUIMode::Dashboard,
                                    TUIMode::Dashboard => TUIMode::ChartDashboard,
                                    TUIMode::ChartDashboard => TUIMode::Minimal,
                                };
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Render at configured interval
        if last_render.elapsed() >= render_interval && !state.paused.load(Ordering::Relaxed) {
            let mode = *state.mode.lock().unwrap();
            terminal.draw(|frame| {
                match mode {
                    TUIMode::Minimal => render_minimal(frame, &state),
                    TUIMode::Dashboard => render_dashboard(frame, &state),
                    TUIMode::ChartDashboard => render_chart_dashboard(frame, &state),
                }
            })?;
            last_render = Instant::now();
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

/// Render minimal single-line display.
fn render_minimal(frame: &mut Frame, state: &SharedState) {
    let area = frame.area();

    let steps = state.buffer.get_steps();
    let episodes = state.buffer.get_episodes();
    let reward = state.buffer.get_mean_reward();
    let sps = state.buffer.get_sps();

    let text = format!(
        " Steps: {:>12} │ Episodes: {:>8} │ Reward: {:>10.2} │ SPS: {:>12.0} │ [q]uit [p]ause [m]ode ",
        format_number(steps),
        format_number(episodes),
        reward,
        sps
    );

    let style = Style::default().fg(Color::Cyan);
    let paragraph = Paragraph::new(text)
        .style(style)
        .block(Block::default().borders(Borders::ALL).title(" Operant "));

    frame.render_widget(paragraph, area);
}

/// Render full dashboard with sparklines.
fn render_dashboard(frame: &mut Frame, state: &SharedState) {
    let area = frame.area();

    // Layout: stats, device bar, sparklines, help
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // Training Stats
            Constraint::Length(3),  // Device Stats Bar
            Constraint::Min(8),     // Sparklines Grid
            Constraint::Length(3),  // Help
        ])
        .split(area);

    // Training stats section
    render_stats(frame, chunks[0], state);

    // Device stats bar
    render_device_stats(frame, chunks[1], state);

    // Sparklines section
    render_sparklines(frame, chunks[2], state);

    // Help section
    render_help(frame, chunks[3]);
}

fn render_stats(frame: &mut Frame, area: Rect, state: &SharedState) {
    let steps = state.buffer.get_steps();
    let episodes = state.buffer.get_episodes();
    let reward = state.buffer.get_mean_reward();
    let sps = state.buffer.get_sps();
    let policy_loss = state.buffer.get_policy_loss();
    let value_loss = state.buffer.get_value_loss();
    let entropy = state.buffer.get_entropy();

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // Steps & Episodes
    let stats1 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Steps: ", Style::default().fg(Color::Gray)),
            Span::styled(format_number(steps), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("Episodes: ", Style::default().fg(Color::Gray)),
            Span::styled(format_number(episodes), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title(" Progress "));
    frame.render_widget(stats1, chunks[0]);

    // Reward & SPS
    let stats2 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Reward: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{:.2}", reward), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("SPS: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{:.0}", sps), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title(" Performance "));
    frame.render_widget(stats2, chunks[1]);

    // Losses
    let stats3 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Policy: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{:.6}", policy_loss), Style::default().fg(Color::Red)),
        ]),
        Line::from(vec![
            Span::styled("Value: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{:.6}", value_loss), Style::default().fg(Color::Blue)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title(" Losses "));
    frame.render_widget(stats3, chunks[2]);

    // Entropy gauge
    let entropy_pct = (entropy.abs().min(1.0) * 100.0) as u16;
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Entropy "))
        .gauge_style(Style::default().fg(Color::Green))
        .percent(entropy_pct)
        .label(format!("{:.4}", entropy));
    frame.render_widget(gauge, chunks[3]);
}

/// Render device/resource monitoring bar
fn render_device_stats(frame: &mut Frame, area: Rect, state: &SharedState) {
    let cpu_pct = state.buffer.get_cpu_usage();
    let gpu_pct = state.buffer.get_gpu_usage();
    let mem_mb = state.buffer.get_memory_usage_mb();
    let total_mem_mb = state.buffer.get_total_memory_mb();
    let gpu_mem_mb = state.buffer.get_gpu_memory_mb();
    let total_gpu_mem_mb = state.buffer.get_total_gpu_memory_mb();

    let mem_gb = mem_mb as f64 / 1024.0;
    let total_mem_gb = total_mem_mb as f64 / 1024.0;
    let gpu_mem_gb = gpu_mem_mb as f64 / 1024.0;
    let total_gpu_mem_gb = total_gpu_mem_mb as f64 / 1024.0;

    // Check if GPU is available (NVML initialized successfully)
    let has_gpu = total_gpu_mem_mb > 0;

    // Dynamic layout based on GPU availability
    let chunks = if has_gpu {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25), // CPU
                Constraint::Percentage(25), // GPU
                Constraint::Percentage(25), // RAM
                Constraint::Percentage(25), // VRAM
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50), // CPU
                Constraint::Percentage(50), // RAM
            ])
            .split(area)
    };

    // CPU panel (always shown)
    let cpu_text = if cpu_pct > 0.0 {
        format!("{:.1}%", cpu_pct)
    } else {
        "N/A".to_string()
    };
    let cpu_color = if cpu_pct > 80.0 {
        Color::Red
    } else if cpu_pct > 60.0 {
        Color::Yellow
    } else {
        Color::Green
    };

    let cpu_widget = Paragraph::new(vec![Line::from(vec![
        Span::styled("CPU: ", Style::default().fg(Color::Gray)),
        Span::styled(
            cpu_text,
            Style::default().fg(cpu_color).add_modifier(Modifier::BOLD),
        ),
    ])])
    .block(Block::default().borders(Borders::ALL).title(" CPU "));
    frame.render_widget(cpu_widget, chunks[0]);

    if has_gpu {
        // GPU panel (only when GPU available)
        let gpu_text = if gpu_pct > 0.0 {
            format!("{:.1}%", gpu_pct)
        } else {
            "N/A".to_string()
        };
        let gpu_color = if gpu_pct > 80.0 {
            Color::Red
        } else if gpu_pct > 60.0 {
            Color::Yellow
        } else {
            Color::Green
        };

        let gpu_widget = Paragraph::new(vec![Line::from(vec![
            Span::styled("GPU: ", Style::default().fg(Color::Gray)),
            Span::styled(
                gpu_text,
                Style::default().fg(gpu_color).add_modifier(Modifier::BOLD),
            ),
        ])])
        .block(Block::default().borders(Borders::ALL).title(" GPU "));
        frame.render_widget(gpu_widget, chunks[1]);

        // RAM panel (index 2 when GPU present)
        let mem_text = if total_mem_mb > 0 {
            format!("{:.1}/{:.1} GB", mem_gb, total_mem_gb)
        } else {
            "N/A".to_string()
        };
        let mem_pct = if total_mem_mb > 0 {
            (mem_mb as f64 / total_mem_mb as f64) * 100.0
        } else {
            0.0
        };
        let mem_color = if mem_pct > 80.0 {
            Color::Red
        } else if mem_pct > 60.0 {
            Color::Yellow
        } else {
            Color::Cyan
        };

        let mem_widget = Paragraph::new(vec![Line::from(vec![
            Span::styled("RAM: ", Style::default().fg(Color::Gray)),
            Span::styled(
                mem_text,
                Style::default().fg(mem_color).add_modifier(Modifier::BOLD),
            ),
        ])])
        .block(Block::default().borders(Borders::ALL).title(" Memory "));
        frame.render_widget(mem_widget, chunks[2]);

        // VRAM panel (only when GPU available)
        let vram_text = if total_gpu_mem_mb > 0 {
            format!("{:.1}/{:.1} GB", gpu_mem_gb, total_gpu_mem_gb)
        } else {
            "N/A".to_string()
        };
        let vram_pct = if total_gpu_mem_mb > 0 {
            (gpu_mem_mb as f64 / total_gpu_mem_mb as f64) * 100.0
        } else {
            0.0
        };
        let vram_color = if vram_pct > 80.0 {
            Color::Red
        } else if vram_pct > 60.0 {
            Color::Yellow
        } else {
            Color::Magenta
        };

        let vram_widget = Paragraph::new(vec![Line::from(vec![
            Span::styled("VRAM: ", Style::default().fg(Color::Gray)),
            Span::styled(
                vram_text,
                Style::default().fg(vram_color).add_modifier(Modifier::BOLD),
            ),
        ])])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" GPU Memory "),
        );
        frame.render_widget(vram_widget, chunks[3]);
    } else {
        // RAM panel (index 1 when no GPU)
        let mem_text = if total_mem_mb > 0 {
            format!("{:.1}/{:.1} GB", mem_gb, total_mem_gb)
        } else {
            "N/A".to_string()
        };
        let mem_pct = if total_mem_mb > 0 {
            (mem_mb as f64 / total_mem_mb as f64) * 100.0
        } else {
            0.0
        };
        let mem_color = if mem_pct > 80.0 {
            Color::Red
        } else if mem_pct > 60.0 {
            Color::Yellow
        } else {
            Color::Cyan
        };

        let mem_widget = Paragraph::new(vec![Line::from(vec![
            Span::styled("RAM: ", Style::default().fg(Color::Gray)),
            Span::styled(
                mem_text,
                Style::default().fg(mem_color).add_modifier(Modifier::BOLD),
            ),
        ])])
        .block(Block::default().borders(Borders::ALL).title(" Memory "));
        frame.render_widget(mem_widget, chunks[1]);
    }
}

/// Render sparklines grid: Reward | CPU/Memory | GPU/VRAM
fn render_sparklines(frame: &mut Frame, area: Rect, state: &SharedState) {
    // Check if GPU is available (NVML initialized successfully)
    let total_gpu_mem_mb = state.buffer.get_total_gpu_memory_mb();
    let has_gpu = total_gpu_mem_mb > 0;

    // Dynamic layout based on GPU availability
    let chunks = if has_gpu {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33), // Reward
                Constraint::Percentage(33), // CPU/RAM
                Constraint::Percentage(34), // GPU/VRAM
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50), // Reward
                Constraint::Percentage(50), // CPU/RAM
            ])
            .split(area)
    };

    if let Ok(history) = state.history.lock() {
        // Chart 1: Reward History - Normalized for clean line graphs
        let reward_data: Vec<u64> = if history.rewards.len() >= 2 {
            let min = history.rewards.iter().copied().fold(f64::INFINITY, f64::min);
            let max = history.rewards.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = (max - min).max(1.0); // Avoid division by zero

            history
                .rewards
                .iter()
                .map(|&r| {
                    let normalized = ((r - min) / range) * 100.0;
                    normalized.max(0.0).min(100.0) as u64
                })
                .collect()
        } else {
            vec![50] // Default value when insufficient data
        };

        let reward_sparkline = Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title(" Reward History "))
            .data(&reward_data)
            .max(100) // CRITICAL: Set max for consistent scaling
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(reward_sparkline, chunks[0]);

        // Chart 2: CPU & Memory (stacked)
        let cpu_mem_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);

        let cpu_data: Vec<u64> = history
            .cpu
            .iter()
            .map(|&c| c.max(0.0).min(100.0) as u64)
            .collect();

        let cpu_sparkline = Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title(" CPU % "))
            .data(&cpu_data)
            .max(100) // Consistent scaling
            .style(Style::default().fg(Color::Green));
        frame.render_widget(cpu_sparkline, cpu_mem_chunks[0]);

        let mem_data: Vec<u64> = history
            .memory
            .iter()
            .map(|&m| m.max(0.0).min(100.0) as u64)
            .collect();

        let mem_sparkline = Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title(" RAM % "))
            .data(&mem_data)
            .max(100) // Consistent scaling
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(mem_sparkline, cpu_mem_chunks[1]);

        // Chart 3: GPU & VRAM (stacked) - only when GPU available
        if has_gpu {
            let gpu_mem_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[2]);

            let gpu_data: Vec<u64> = history
                .gpu
                .iter()
                .map(|&g| g.max(0.0).min(100.0) as u64)
                .collect();

            let gpu_sparkline = Sparkline::default()
                .block(Block::default().borders(Borders::ALL).title(" GPU % "))
                .data(&gpu_data)
                .max(100) // Consistent scaling
                .style(Style::default().fg(Color::Green));
            frame.render_widget(gpu_sparkline, gpu_mem_chunks[0]);

            let vram_data: Vec<u64> = history
                .gpu_memory
                .iter()
                .map(|&v| v.max(0.0).min(100.0) as u64)
                .collect();

            let vram_sparkline = Sparkline::default()
                .block(Block::default().borders(Borders::ALL).title(" VRAM % "))
                .data(&vram_data)
                .max(100) // Consistent scaling
                .style(Style::default().fg(Color::Magenta));
            frame.render_widget(vram_sparkline, gpu_mem_chunks[1]);
        }
    }
}

fn render_help(frame: &mut Frame, area: Rect) {
    let help_text = " [q] Quit  [p] Pause  [m] Toggle Mode  [s] Save Checkpoint ";
    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(help, area);
}

/// Format large numbers with commas.
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

// ============================================================================
// Chart Dashboard Mode - Modern visualization with proper axes and labels
// ============================================================================

/// Convert step and value VecDeques to chart-compatible Vec<(f64, f64)>
fn history_to_chart_data(steps: &VecDeque<u64>, values: &VecDeque<f64>) -> Vec<(f64, f64)> {
    steps
        .iter()
        .zip(values.iter())
        .map(|(&s, &v)| (s as f64, v))
        .collect()
}

/// Calculate axis bounds with optional padding percentage
fn calculate_bounds(data: &[(f64, f64)], padding: f64) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let x_min = data.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let x_max = data.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
    let y_min = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

    let y_range = (y_max - y_min).max(0.001); // Avoid zero range
    let y_padding = y_range * padding;

    (
        x_min,
        x_max.max(x_min + 1.0), // Ensure x range is at least 1
        y_min - y_padding,
        y_max + y_padding,
    )
}

/// Format step counts for axis labels (e.g., 1000 -> "1K", 1500000 -> "1.5M")
fn format_steps_label(n: f64) -> String {
    if n >= 1_000_000.0 {
        format!("{:.1}M", n / 1_000_000.0)
    } else if n >= 1_000.0 {
        format!("{:.1}K", n / 1_000.0)
    } else {
        format!("{:.0}", n)
    }
}

/// Create x-axis with step labels
fn create_x_axis(x_min: f64, x_max: f64) -> Axis<'static> {
    Axis::default()
        .style(Style::default().fg(Color::DarkGray))
        .bounds([x_min, x_max])
        .labels(vec![
            Span::from(format_steps_label(x_min)),
            Span::from(format_steps_label((x_min + x_max) / 2.0)),
            Span::from(format_steps_label(x_max)),
        ])
}

/// Create y-axis with auto-scaled value labels
fn create_y_axis(y_min: f64, y_max: f64, title: &'static str) -> Axis<'static> {
    Axis::default()
        .title(Span::styled(title, Style::default().fg(Color::DarkGray)))
        .style(Style::default().fg(Color::DarkGray))
        .bounds([y_min, y_max])
        .labels(vec![
            Span::from(format!("{:.2}", y_min)),
            Span::from(format!("{:.2}", (y_min + y_max) / 2.0)),
            Span::from(format!("{:.2}", y_max)),
        ])
}

/// Create fixed 0-100% y-axis for resource charts
fn create_percentage_y_axis(title: &'static str) -> Axis<'static> {
    Axis::default()
        .title(Span::styled(title, Style::default().fg(Color::DarkGray)))
        .style(Style::default().fg(Color::DarkGray))
        .bounds([0.0, 100.0])
        .labels(vec![
            Span::from("0"),
            Span::from("50"),
            Span::from("100"),
        ])
}

/// Main entry point for Chart Dashboard mode
fn render_chart_dashboard(frame: &mut Frame, state: &SharedState) {
    let area = frame.area();

    // Main layout: Stats bar | Device bar | Training Charts | Help bar
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // Stats bar (same as Dashboard)
            Constraint::Length(3),  // Device stats bar (compact CPU/GPU/RAM/VRAM)
            Constraint::Min(10),    // Training charts (rewards + losses)
            Constraint::Length(3),  // Help bar
        ])
        .split(area);

    // Render stats bar (reuse existing)
    render_stats(frame, main_chunks[0], state);

    // Render device stats bar (reuse existing - already compact)
    render_device_stats(frame, main_chunks[1], state);

    // Render training charts (NEW: training metrics only)
    render_training_charts(frame, main_chunks[2], state);

    // Render help with updated text
    render_chart_help(frame, main_chunks[3]);
}

/// Render training metrics charts (rewards + policy/value losses)
fn render_training_charts(frame: &mut Frame, area: Rect, state: &SharedState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60),  // Reward chart (main focus)
            Constraint::Percentage(40),  // Losses (policy + value side-by-side)
        ])
        .split(area);

    // Top: Large reward chart
    render_reward_chart(frame, chunks[0], state);

    // Bottom: Policy and Value loss side-by-side
    let loss_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(chunks[1]);

    render_policy_loss_chart(frame, loss_chunks[0], state);
    render_value_loss_chart(frame, loss_chunks[1], state);
}

/// Render the chart grid with dynamic columns based on GPU availability
fn render_chart_grid(frame: &mut Frame, area: Rect, state: &SharedState) {
    let total_gpu_mem_mb = state.buffer.get_total_gpu_memory_mb();
    let has_gpu = total_gpu_mem_mb > 0;

    // Dynamic layout based on GPU availability
    let chunks = if has_gpu {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(40), // Reward/Losses column
                Constraint::Percentage(30), // CPU/RAM column
                Constraint::Percentage(30), // GPU/VRAM column
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50), // Reward/Losses column
                Constraint::Percentage(50), // CPU/RAM column
            ])
            .split(area)
    };

    // Column 1: Reward and Losses charts
    render_reward_loss_column(frame, chunks[0], state);

    // Column 2: CPU and RAM charts
    render_system_resources_column(frame, chunks[1], state);

    // Column 3: GPU and VRAM charts (only if GPU available)
    if has_gpu {
        render_gpu_resources_column(frame, chunks[2], state);
    }
}

/// Render Column 1: Reward (60%) and Losses (40%)
fn render_reward_loss_column(frame: &mut Frame, area: Rect, state: &SharedState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60), // Reward chart
            Constraint::Percentage(40), // Losses chart
        ])
        .split(area);

    render_reward_chart(frame, chunks[0], state);
    render_losses_chart(frame, chunks[1], state);
}

/// Render reward history chart with auto-scaling y-axis
fn render_reward_chart(frame: &mut Frame, area: Rect, state: &SharedState) {
    if let Ok(history) = state.history.lock() {
        if history.steps.is_empty() {
            let placeholder = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default().borders(Borders::ALL).title(" Reward History "));
            frame.render_widget(placeholder, area);
            return;
        }

        let data = history_to_chart_data(&history.steps, &history.rewards);
        let (x_min, x_max, y_min, y_max) = calculate_bounds(&data, 0.05);

        let dataset = Dataset::default()
            .name("Reward")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(&data);

        let chart = Chart::new(vec![dataset])
            .block(Block::default().borders(Borders::ALL).title(" Reward History "))
            .x_axis(create_x_axis(x_min, x_max))
            .y_axis(create_y_axis(y_min, y_max, "Reward"));

        frame.render_widget(chart, area);
    }
}

/// Render dual-series losses chart (policy + value)
fn render_losses_chart(frame: &mut Frame, area: Rect, state: &SharedState) {
    if let Ok(history) = state.history.lock() {
        if history.steps.is_empty() {
            let placeholder = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default().borders(Borders::ALL).title(" Losses "));
            frame.render_widget(placeholder, area);
            return;
        }

        let policy_data = history_to_chart_data(&history.steps, &history.policy_loss);
        let value_data = history_to_chart_data(&history.steps, &history.value_loss);

        // Calculate combined bounds
        let all_data: Vec<(f64, f64)> = policy_data.iter().chain(value_data.iter()).copied().collect();
        let (x_min, x_max, y_min, y_max) = calculate_bounds(&all_data, 0.05);

        let datasets = vec![
            Dataset::default()
                .name("Policy")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Red))
                .data(&policy_data),
            Dataset::default()
                .name("Value")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Blue))
                .data(&value_data),
        ];

        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title(" Losses (Policy/Value) "))
            .x_axis(create_x_axis(x_min, x_max))
            .y_axis(create_y_axis(y_min, y_max, "Loss"));

        frame.render_widget(chart, area);
    }
}

/// Render policy loss chart (single series)
fn render_policy_loss_chart(frame: &mut Frame, area: Rect, state: &SharedState) {
    if let Ok(history) = state.history.lock() {
        if history.steps.is_empty() {
            let placeholder = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default().borders(Borders::ALL).title(" Policy Loss "));
            frame.render_widget(placeholder, area);
            return;
        }

        let data = history_to_chart_data(&history.steps, &history.policy_loss);
        let (x_min, x_max, y_min, y_max) = calculate_bounds(&data, 0.05);

        let dataset = Dataset::default()
            .name("Policy")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Red))
            .data(&data);

        let chart = Chart::new(vec![dataset])
            .block(Block::default().borders(Borders::ALL).title(" Policy Loss "))
            .x_axis(create_x_axis(x_min, x_max))
            .y_axis(create_y_axis(y_min, y_max, "Loss"));

        frame.render_widget(chart, area);
    }
}

/// Render value loss chart (single series)
fn render_value_loss_chart(frame: &mut Frame, area: Rect, state: &SharedState) {
    if let Ok(history) = state.history.lock() {
        if history.steps.is_empty() {
            let placeholder = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default().borders(Borders::ALL).title(" Value Loss "));
            frame.render_widget(placeholder, area);
            return;
        }

        let data = history_to_chart_data(&history.steps, &history.value_loss);
        let (x_min, x_max, y_min, y_max) = calculate_bounds(&data, 0.05);

        let dataset = Dataset::default()
            .name("Value")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Blue))
            .data(&data);

        let chart = Chart::new(vec![dataset])
            .block(Block::default().borders(Borders::ALL).title(" Value Loss "))
            .x_axis(create_x_axis(x_min, x_max))
            .y_axis(create_y_axis(y_min, y_max, "Loss"));

        frame.render_widget(chart, area);
    }
}

/// Render Column 2: CPU (50%) and RAM (50%)
fn render_system_resources_column(frame: &mut Frame, area: Rect, state: &SharedState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50), // CPU chart
            Constraint::Percentage(50), // RAM chart
        ])
        .split(area);

    render_cpu_chart(frame, chunks[0], state);
    render_memory_chart(frame, chunks[1], state);
}

/// Render CPU usage chart (0-100% fixed scale)
fn render_cpu_chart(frame: &mut Frame, area: Rect, state: &SharedState) {
    if let Ok(history) = state.history.lock() {
        if history.steps.is_empty() {
            let placeholder = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default().borders(Borders::ALL).title(" CPU % "));
            frame.render_widget(placeholder, area);
            return;
        }

        let data = history_to_chart_data(&history.steps, &history.cpu);
        let (x_min, x_max, _, _) = calculate_bounds(&data, 0.0);

        let dataset = Dataset::default()
            .name("CPU")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&data);

        let chart = Chart::new(vec![dataset])
            .block(Block::default().borders(Borders::ALL).title(" CPU % "))
            .x_axis(create_x_axis(x_min, x_max))
            .y_axis(create_percentage_y_axis("%"));

        frame.render_widget(chart, area);
    }
}

/// Render RAM usage chart (0-100% fixed scale)
fn render_memory_chart(frame: &mut Frame, area: Rect, state: &SharedState) {
    if let Ok(history) = state.history.lock() {
        if history.steps.is_empty() {
            let placeholder = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default().borders(Borders::ALL).title(" RAM % "));
            frame.render_widget(placeholder, area);
            return;
        }

        let data = history_to_chart_data(&history.steps, &history.memory);
        let (x_min, x_max, _, _) = calculate_bounds(&data, 0.0);

        let dataset = Dataset::default()
            .name("RAM")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&data);

        let chart = Chart::new(vec![dataset])
            .block(Block::default().borders(Borders::ALL).title(" RAM % "))
            .x_axis(create_x_axis(x_min, x_max))
            .y_axis(create_percentage_y_axis("%"));

        frame.render_widget(chart, area);
    }
}

/// Render Column 3: GPU (50%) and VRAM (50%) - only when GPU available
fn render_gpu_resources_column(frame: &mut Frame, area: Rect, state: &SharedState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50), // GPU chart
            Constraint::Percentage(50), // VRAM chart
        ])
        .split(area);

    render_gpu_chart(frame, chunks[0], state);
    render_vram_chart(frame, chunks[1], state);
}

/// Render GPU usage chart (0-100% fixed scale)
fn render_gpu_chart(frame: &mut Frame, area: Rect, state: &SharedState) {
    if let Ok(history) = state.history.lock() {
        if history.steps.is_empty() {
            let placeholder = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default().borders(Borders::ALL).title(" GPU % "));
            frame.render_widget(placeholder, area);
            return;
        }

        let data = history_to_chart_data(&history.steps, &history.gpu);
        let (x_min, x_max, _, _) = calculate_bounds(&data, 0.0);

        let dataset = Dataset::default()
            .name("GPU")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&data);

        let chart = Chart::new(vec![dataset])
            .block(Block::default().borders(Borders::ALL).title(" GPU % "))
            .x_axis(create_x_axis(x_min, x_max))
            .y_axis(create_percentage_y_axis("%"));

        frame.render_widget(chart, area);
    }
}

/// Render VRAM usage chart (0-100% fixed scale)
fn render_vram_chart(frame: &mut Frame, area: Rect, state: &SharedState) {
    if let Ok(history) = state.history.lock() {
        if history.steps.is_empty() {
            let placeholder = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default().borders(Borders::ALL).title(" VRAM % "));
            frame.render_widget(placeholder, area);
            return;
        }

        let data = history_to_chart_data(&history.steps, &history.gpu_memory);
        let (x_min, x_max, _, _) = calculate_bounds(&data, 0.0);

        let dataset = Dataset::default()
            .name("VRAM")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Magenta))
            .data(&data);

        let chart = Chart::new(vec![dataset])
            .block(Block::default().borders(Borders::ALL).title(" VRAM % "))
            .x_axis(create_x_axis(x_min, x_max))
            .y_axis(create_percentage_y_axis("%"));

        frame.render_widget(chart, area);
    }
}

/// Help text for Chart Dashboard mode
fn render_chart_help(frame: &mut Frame, area: Rect) {
    let help_text = " [q] Quit  [p] Pause  [m] Switch Mode  [s] Save Checkpoint ";
    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(help, area);
}
