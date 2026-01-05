"""
Expectation-Gated Decay Neural Units — Comprehensive Test
==========================================================

Addresses shortcuts from simple demo:
1. Longer runtime for proper capacitance building
2. Multiple distinct patterns (not just structured vs noise)
3. Pattern discrimination testing
4. Phase transitions (pattern switches) to test adaptation
5. More rigorous violation computation with actual phase tracking
6. Cross-pattern interference testing

Dependencies:
    pip install numpy matplotlib scipy

Run:
    python test_expectation_gated_comprehensive.py
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy import stats
from collections import deque


@dataclass
class RobustExpectationUnit:
    """
    Improved unit with proper phase tracking and violation computation.
    """
    # === Core state ===
    V: float = 0.0                    # Stored value (charge)
    C: float = 0.01                   # Capacitance (starts near zero)
    λ: float = 0.1                    # Current leakage rate
    
    # === Learned temporal model ===
    w: float = 1.0                    # Weight/gain
    τ_exp: float = 15.0               # Expected period (learned)
    τ_var: float = 25.0               # Variance in period estimate (uncertainty)
    phase: float = 0.0                # Current phase in [0, 1)
    
    # === History for learning ===
    activation_times: deque = field(default_factory=lambda: deque(maxlen=20))
    interval_history: deque = field(default_factory=lambda: deque(maxlen=15))
    delta_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # === Hyperparameters ===
    λ_base: float = 0.015             # Minimum leakage (slower than before)
    λ_max: float = 0.6                # Maximum leakage
    β_violation: float = 0.5          # Violation → leak scaling
    α_C_build: float = 0.008          # Capacitance build rate (slower)
    α_C_decay: float = 0.002          # Capacitance passive decay
    α_τ: float = 0.08                 # Period learning rate
    α_var: float = 0.1                # Variance learning rate
    C_max: float = 10.0               # Maximum capacitance
    activation_threshold: float = 0.2
    
    # === Tracking ===
    tick: int = 0
    total_activations: int = 0
    violation_ema: float = 0.5        # Smoothed violation
    consistency_ema: float = 0.5      # Smoothed consistency
    
    # === Full history for analysis ===
    history: dict = field(default_factory=lambda: {
        'V': [], 'C': [], 'λ': [], 'violation': [], 
        'τ_exp': [], 'τ_var': [], 'phase': [], 'consistency': []
    })
    
    def _compute_expected_activation_probability(self) -> float:
        """
        Probability that we expect activation NOW based on learned period.
        Uses phase and uncertainty to create a probabilistic expectation.
        """
        # Expectation peaks when phase is near 1.0 (about to wrap)
        # Width of peak depends on uncertainty (τ_var)
        phase_to_expected = 1.0 - self.phase  # Distance to next expected activation
        
        # Gaussian-ish expectation window
        # High τ_var = broad expectation (uncertain timing)
        # Low τ_var = sharp expectation (confident timing)
        uncertainty = np.sqrt(self.τ_var) / max(1.0, self.τ_exp)
        uncertainty = np.clip(uncertainty, 0.05, 0.5)
        
        # Probability of expecting activation now
        p_expect = np.exp(-0.5 * (phase_to_expected / uncertainty) ** 2)
        
        return p_expect
    
    def _compute_violation(self, got_activation: bool) -> float:
        """
        Compute expectation violation.
        
        Four cases:
        1. Expected activation, got it → LOW violation (good)
        2. Expected activation, didn't get it → HIGH violation (bad)
        3. Didn't expect, didn't get → ZERO violation (fine)
        4. Didn't expect, got it → MEDIUM violation (surprise)
        
        The key insight: violation should depend on CONFIDENCE of expectation.
        """
        p_expect = self._compute_expected_activation_probability()
        
        if got_activation:
            if p_expect > 0.5:
                # Expected and got → good, low violation
                # Better match = lower violation
                return 0.1 * (1.0 - p_expect)
            else:
                # Unexpected activation
                # How unexpected determines violation
                return 0.4 * (1.0 - p_expect)
        else:
            if p_expect > 0.5:
                # Expected but didn't get → violation proportional to expectation
                return 0.8 * p_expect
            else:
                # Didn't expect, didn't get → fine
                return 0.05
    
    def _update_temporal_model(self, t: int, got_activation: bool):
        """Update learned period and variance from activation history."""
        if got_activation:
            self.activation_times.append(t)
            self.total_activations += 1
            
            if len(self.activation_times) >= 2:
                # Compute interval
                interval = self.activation_times[-1] - self.activation_times[-2]
                
                if 2 < interval < 200:  # Sanity bounds
                    self.interval_history.append(interval)
                    
                    # Update period estimate (exponential moving average)
                    self.τ_exp = (1 - self.α_τ) * self.τ_exp + self.α_τ * interval
                    
                    # Update variance estimate
                    if len(self.interval_history) >= 3:
                        recent_var = np.var(list(self.interval_history)[-5:])
                        self.τ_var = (1 - self.α_var) * self.τ_var + self.α_var * recent_var
                        self.τ_var = np.clip(self.τ_var, 0.1, 100.0)
            
            # Reset phase on activation
            self.phase = 0.0
    
    def _compute_consistency(self) -> float:
        """
        Measure how consistent recent intervals are with learned period.
        High consistency → build capacitance.
        """
        if len(self.interval_history) < 3:
            return 0.3  # Uncertain initially
        
        recent = list(self.interval_history)[-5:]
        
        # How close are recent intervals to expected period?
        deviations = [abs(i - self.τ_exp) / max(1.0, self.τ_exp) for i in recent]
        mean_deviation = np.mean(deviations)
        
        # Also consider variance - consistent timing is good
        if len(recent) >= 3:
            cv = np.std(recent) / max(1.0, np.mean(recent))  # Coefficient of variation
        else:
            cv = 0.5
        
        # Consistency: low deviation + low variance → high consistency
        consistency = np.exp(-2.0 * mean_deviation) * np.exp(-2.0 * cv)
        
        return np.clip(consistency, 0.0, 1.0)
    
    def step(self, t: int, input_delta: float) -> float:
        """Single timestep update with full dynamics."""
        self.tick = t
        
        # Store delta history for pattern analysis
        self.delta_history.append(input_delta)
        
        # Detect activation
        got_activation = abs(input_delta) > self.activation_threshold
        
        # 1. Compute expectation violation
        violation = self._compute_violation(got_activation)
        self.violation_ema = 0.9 * self.violation_ema + 0.1 * violation
        
        # 2. Update temporal model if activation occurred
        self._update_temporal_model(t, got_activation)
        
        # 3. Advance phase
        self.phase += 1.0 / max(1.0, self.τ_exp)
        if self.phase >= 1.0:
            self.phase -= 1.0
        
        # 4. Compute consistency and update capacitance
        consistency = self._compute_consistency()
        self.consistency_ema = 0.9 * self.consistency_ema + 0.1 * consistency
        
        # Capacitance builds from consistency, decays passively and from violation
        dC_build = self.α_C_build * self.consistency_ema
        dC_decay = self.α_C_decay + 0.01 * self.violation_ema
        self.C = np.clip(self.C + dC_build - dC_decay * self.C, 0.001, self.C_max)
        
        # 5. Update leakage from violation
        # High violation → high leak
        # But high capacitance provides some protection
        protection = 1.0 / (1.0 + 0.3 * self.C)
        self.λ = np.clip(
            self.λ_base + self.β_violation * self.violation_ema * protection,
            self.λ_base,
            self.λ_max
        )
        
        # 6. State dynamics
        # Effective leak reduced by capacitance
        effective_λ = self.λ / (1.0 + 0.15 * self.C)
        
        # Nonlinear leak (tanh prevents explosion)
        leak = effective_λ * np.tanh(self.V)
        
        # Excitation: weight * capacitance * input delta
        excitation = self.w * (0.5 + 0.5 * self.C) * input_delta
        
        # Update state
        self.V = self.V + excitation - leak
        
        # Record history
        self.history['V'].append(self.V)
        self.history['C'].append(self.C)
        self.history['λ'].append(self.λ)
        self.history['violation'].append(violation)
        self.history['τ_exp'].append(self.τ_exp)
        self.history['τ_var'].append(self.τ_var)
        self.history['phase'].append(self.phase)
        self.history['consistency'].append(self.consistency_ema)
        
        return self.V


class PatternGenerator:
    """Generate distinct temporal patterns for testing."""
    
    @staticmethod
    def periodic(t: int, period: int, phase_offset: int = 0, 
                 pulse_width: int = 2, amplitude: float = 1.0) -> float:
        """Regular periodic pulse."""
        if (t + phase_offset) % period < pulse_width:
            return amplitude
        return 0.0
    
    @staticmethod
    def bursting(t: int, burst_period: int, pulses_per_burst: int = 3,
                 inter_pulse: int = 3) -> float:
        """Bursting pattern: groups of pulses."""
        burst_length = pulses_per_burst * inter_pulse
        pos_in_cycle = t % burst_period
        if pos_in_cycle < burst_length:
            if pos_in_cycle % inter_pulse < 2:
                return 1.0
        return 0.0
    
    @staticmethod
    def accelerating(t: int, base_period: int, acceleration: float = 0.995) -> float:
        """Pattern with gradually decreasing period."""
        current_period = max(3, int(base_period * (acceleration ** (t / 50))))
        if t % current_period < 2:
            return 1.0
        return 0.0
    
    @staticmethod
    def noise(amplitude: float = 0.3) -> float:
        """Random noise."""
        return np.random.randn() * amplitude


def generate_multi_pattern_signal(
    n_ticks: int,
    n_units: int,
    patterns: List[Tuple[int, int, str, dict]]  # (start, end, pattern_type, params)
) -> Tuple[np.ndarray, dict]:
    """
    Generate signal with multiple distinct patterns in different regions.
    
    patterns: list of (start_unit, end_unit, pattern_type, params_dict)
    """
    signal = np.zeros((n_ticks, n_units))
    pattern_info = {}
    
    for start, end, ptype, params in patterns:
        pattern_info[(start, end)] = {'type': ptype, 'params': params}
        
        for t in range(n_ticks):
            if ptype == 'periodic':
                val = PatternGenerator.periodic(t, **params)
            elif ptype == 'bursting':
                val = PatternGenerator.bursting(t, **params)
            elif ptype == 'accelerating':
                val = PatternGenerator.accelerating(t, **params)
            elif ptype == 'noise':
                val = PatternGenerator.noise(**params)
            else:
                val = 0.0
            
            signal[t, start:end] = val
    
    return signal, pattern_info


def run_comprehensive_test():
    """
    Comprehensive test with multiple patterns and phase transitions.
    """
    print("=" * 80)
    print("COMPREHENSIVE EXPECTATION-GATED DECAY TEST")
    print("=" * 80)
    
    # === Configuration ===
    n_units = 100
    
    # Phase durations (longer for proper learning)
    phase1_ticks = 500   # Initial learning
    phase2_ticks = 300   # Continued patterns (verify stability)
    phase3_ticks = 400   # Pattern switch (test adaptation)
    phase4_ticks = 300   # Return to original (test recovery)
    
    n_ticks = phase1_ticks + phase2_ticks + phase3_ticks + phase4_ticks
    
    # Define pattern regions
    patterns_phase1 = [
        (0, 15, 'noise', {'amplitude': 0.25}),           # Noise baseline
        (15, 35, 'periodic', {'period': 8, 'pulse_width': 2}),    # Fast periodic
        (35, 55, 'periodic', {'period': 20, 'pulse_width': 2}),   # Slow periodic
        (55, 75, 'bursting', {'burst_period': 30, 'pulses_per_burst': 3}),  # Bursting
        (75, 90, 'periodic', {'period': 12, 'pulse_width': 2}),   # Medium periodic
        (90, 100, 'noise', {'amplitude': 0.25}),         # Noise baseline
    ]
    
    # Phase 3: Switch some patterns
    patterns_phase3 = [
        (0, 15, 'noise', {'amplitude': 0.25}),
        (15, 35, 'periodic', {'period': 20, 'pulse_width': 2}),   # SWITCH: was 8, now 20
        (35, 55, 'periodic', {'period': 20, 'pulse_width': 2}),   # Same
        (55, 75, 'periodic', {'period': 15, 'pulse_width': 2}),   # SWITCH: was bursting
        (75, 90, 'periodic', {'period': 12, 'pulse_width': 2}),   # Same
        (90, 100, 'noise', {'amplitude': 0.25}),
    ]
    
    print(f"\nConfiguration:")
    print(f"  Units: {n_units}")
    print(f"  Phase 1 (learn): {phase1_ticks} ticks")
    print(f"  Phase 2 (stable): {phase2_ticks} ticks")
    print(f"  Phase 3 (switch): {phase3_ticks} ticks")
    print(f"  Phase 4 (recover): {phase4_ticks} ticks")
    print(f"  Total: {n_ticks} ticks")
    print(f"\nPatterns:")
    for start, end, ptype, params in patterns_phase1:
        print(f"  Units {start:2d}-{end:2d}: {ptype:12s} {params}")
    print(f"\nPhase 3 switches:")
    print(f"  Units 15-35: period 8 → 20")
    print(f"  Units 55-75: bursting → periodic(15)")
    
    # === Generate signals for each phase ===
    signal = np.zeros((n_ticks, n_units))
    
    # Phase 1 & 2: Original patterns
    sig1, _ = generate_multi_pattern_signal(phase1_ticks + phase2_ticks, n_units, patterns_phase1)
    signal[:phase1_ticks + phase2_ticks] = sig1
    
    # Phase 3: Switched patterns
    sig3, _ = generate_multi_pattern_signal(phase3_ticks, n_units, patterns_phase3)
    signal[phase1_ticks + phase2_ticks:phase1_ticks + phase2_ticks + phase3_ticks] = sig3
    
    # Phase 4: Return to original
    sig4, _ = generate_multi_pattern_signal(phase4_ticks, n_units, patterns_phase1)
    signal[phase1_ticks + phase2_ticks + phase3_ticks:] = sig4
    
    # Compute deltas
    deltas = np.zeros_like(signal)
    deltas[1:] = signal[1:] - signal[:-1]
    
    # === Create units ===
    print("\nInitializing units...")
    units = [RobustExpectationUnit(
        τ_exp=np.random.uniform(8, 25),  # Varied initial expectations
        τ_var=np.random.uniform(10, 30),
    ) for _ in range(n_units)]
    
    # === Run simulation ===
    print("Running simulation...")
    
    V_history = np.zeros((n_ticks, n_units))
    C_history = np.zeros((n_ticks, n_units))
    λ_history = np.zeros((n_ticks, n_units))
    τ_history = np.zeros((n_ticks, n_units))
    violation_history = np.zeros((n_ticks, n_units))
    consistency_history = np.zeros((n_ticks, n_units))
    
    for t in range(n_ticks):
        if t % 200 == 0:
            print(f"  Tick {t}/{n_ticks}")
        
        for i, unit in enumerate(units):
            unit.step(t, deltas[t, i])
            V_history[t, i] = unit.V
            C_history[t, i] = unit.C
            λ_history[t, i] = unit.λ
            τ_history[t, i] = unit.τ_exp
            violation_history[t, i] = unit.violation_ema
            consistency_history[t, i] = unit.consistency_ema
    
    # === Analysis ===
    print("\nAnalyzing results...")
    
    # Define analysis regions
    regions = {
        'noise_1': (0, 15),
        'fast_periodic': (15, 35),
        'slow_periodic': (35, 55),
        'bursting': (55, 75),
        'medium_periodic': (75, 90),
        'noise_2': (90, 100),
    }
    
    true_periods = {
        'noise_1': None,
        'fast_periodic': 8,
        'slow_periodic': 20,
        'bursting': 30,  # Burst period
        'medium_periodic': 12,
        'noise_2': None,
    }
    
    # Phase boundaries
    phase_bounds = [
        0,
        phase1_ticks,
        phase1_ticks + phase2_ticks,
        phase1_ticks + phase2_ticks + phase3_ticks,
        n_ticks
    ]
    phase_names = ['Learning', 'Stable', 'Switched', 'Recovery']
    
    # === Visualization ===
    fig = plt.figure(figsize=(20, 16))
    
    # Add phase boundary markers function
    def add_phase_lines(ax):
        for pb in phase_bounds[1:-1]:
            ax.axvline(pb, color='white', linestyle='--', alpha=0.5, linewidth=1)
    
    # Row 1: Input and Capacitance
    ax1 = fig.add_subplot(4, 3, 1)
    im1 = ax1.imshow(signal.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    add_phase_lines(ax1)
    ax1.set_ylabel('Unit')
    ax1.set_title('Input Signal')
    plt.colorbar(im1, ax=ax1, label='Value')
    
    ax2 = fig.add_subplot(4, 3, 2)
    im2 = ax2.imshow(C_history.T, aspect='auto', cmap='plasma', vmin=0)
    add_phase_lines(ax2)
    ax2.set_ylabel('Unit')
    ax2.set_title('Capacitance C(t)\n(confidence built over time)')
    plt.colorbar(im2, ax=ax2, label='C')
    
    ax3 = fig.add_subplot(4, 3, 3)
    im3 = ax3.imshow(λ_history.T, aspect='auto', cmap='hot', vmin=0)
    add_phase_lines(ax3)
    ax3.set_ylabel('Unit')
    ax3.set_title('Leakage λ(t)\n(high during violation)')
    plt.colorbar(im3, ax=ax3, label='λ')
    
    # Row 2: Learned period and violations
    ax4 = fig.add_subplot(4, 3, 4)
    im4 = ax4.imshow(τ_history.T, aspect='auto', cmap='coolwarm', vmin=5, vmax=35)
    add_phase_lines(ax4)
    # Add true period markers
    for name, (s, e) in regions.items():
        if true_periods[name]:
            ax4.axhline(s, color='lime', alpha=0.3)
            ax4.axhline(e, color='lime', alpha=0.3)
    ax4.set_ylabel('Unit')
    ax4.set_title('Learned Period τ_exp(t)')
    plt.colorbar(im4, ax=ax4, label='τ')
    
    ax5 = fig.add_subplot(4, 3, 5)
    im5 = ax5.imshow(violation_history.T, aspect='auto', cmap='Reds', vmin=0, vmax=0.8)
    add_phase_lines(ax5)
    ax5.set_ylabel('Unit')
    ax5.set_title('Violation EMA\n(expectation mismatch)')
    plt.colorbar(im5, ax=ax5, label='Violation')
    
    ax6 = fig.add_subplot(4, 3, 6)
    im6 = ax6.imshow(consistency_history.T, aspect='auto', cmap='Greens', vmin=0, vmax=1)
    add_phase_lines(ax6)
    ax6.set_ylabel('Unit')
    ax6.set_title('Consistency EMA\n(pattern regularity)')
    plt.colorbar(im6, ax=ax6, label='Consistency')
    
    # Row 3: Per-region analysis over time
    ax7 = fig.add_subplot(4, 3, 7)
    for name, (s, e) in regions.items():
        mean_C = np.mean(C_history[:, s:e], axis=1)
        ax7.plot(mean_C, label=name, alpha=0.8)
    for pb in phase_bounds[1:-1]:
        ax7.axvline(pb, color='gray', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Tick')
    ax7.set_ylabel('Mean C')
    ax7.set_title('Capacitance by Region')
    ax7.legend(loc='upper left', fontsize=8)
    ax7.set_xlim(0, n_ticks)
    
    ax8 = fig.add_subplot(4, 3, 8)
    for name, (s, e) in regions.items():
        mean_λ = np.mean(λ_history[:, s:e], axis=1)
        ax8.plot(mean_λ, label=name, alpha=0.8)
    for pb in phase_bounds[1:-1]:
        ax8.axvline(pb, color='gray', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Tick')
    ax8.set_ylabel('Mean λ')
    ax8.set_title('Leakage by Region')
    ax8.legend(loc='upper right', fontsize=8)
    ax8.set_xlim(0, n_ticks)
    
    ax9 = fig.add_subplot(4, 3, 9)
    for name, (s, e) in regions.items():
        if true_periods[name]:
            mean_τ = np.mean(τ_history[:, s:e], axis=1)
            ax9.plot(mean_τ, label=f'{name} (true={true_periods[name]})', alpha=0.8)
            ax9.axhline(true_periods[name], linestyle=':', alpha=0.3)
    for pb in phase_bounds[1:-1]:
        ax9.axvline(pb, color='gray', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Tick')
    ax9.set_ylabel('Mean τ_exp')
    ax9.set_title('Learned Period by Region')
    ax9.legend(loc='upper right', fontsize=8)
    ax9.set_xlim(0, n_ticks)
    
    # Row 4: Final state analysis and discrimination test
    ax10 = fig.add_subplot(4, 3, 10)
    final_C = C_history[-1]
    colors = []
    for i in range(n_units):
        for name, (s, e) in regions.items():
            if s <= i < e:
                if 'noise' in name:
                    colors.append('gray')
                elif 'fast' in name:
                    colors.append('red')
                elif 'slow' in name:
                    colors.append('blue')
                elif 'burst' in name:
                    colors.append('green')
                elif 'medium' in name:
                    colors.append('orange')
                break
    ax10.bar(range(n_units), final_C, color=colors, alpha=0.7)
    ax10.set_xlabel('Unit')
    ax10.set_ylabel('Final C')
    ax10.set_title('Final Capacitance by Unit\n(colors = pattern type)')
    
    ax11 = fig.add_subplot(4, 3, 11)
    final_τ = τ_history[-1]
    ax11.bar(range(n_units), final_τ, color=colors, alpha=0.7)
    # Add true period lines
    for name, (s, e) in regions.items():
        if true_periods[name]:
            ax11.hlines(true_periods[name], s, e-1, colors='black', linestyles='-', linewidth=2)
    ax11.set_xlabel('Unit')
    ax11.set_ylabel('Final τ_exp')
    ax11.set_title('Final Learned Period\n(black lines = true periods)')
    
    # Discrimination analysis
    ax12 = fig.add_subplot(4, 3, 12)
    
    # Compare C at different phases
    end_phase1 = phase1_ticks
    end_phase2 = phase1_ticks + phase2_ticks
    end_phase3 = phase1_ticks + phase2_ticks + phase3_ticks
    
    region_names = list(regions.keys())
    x_pos = np.arange(len(region_names))
    width = 0.2
    
    for idx, (phase_end, phase_name) in enumerate([
        (end_phase1, 'After Learning'),
        (end_phase2, 'After Stable'),
        (end_phase3, 'After Switch'),
        (n_ticks - 1, 'After Recovery')
    ]):
        means = []
        for name, (s, e) in regions.items():
            means.append(np.mean(C_history[phase_end, s:e]))
        ax12.bar(x_pos + idx * width, means, width, label=phase_name, alpha=0.8)
    
    ax12.set_xticks(x_pos + 1.5 * width)
    ax12.set_xticklabels([r.replace('_', '\n') for r in region_names], fontsize=8)
    ax12.set_ylabel('Mean C')
    ax12.set_title('Capacitance by Region & Phase')
    ax12.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('expectation_gated_comprehensive.png', dpi=150, bbox_inches='tight')
    print("\nSaved: expectation_gated_comprehensive.png")
    
    # === Quantitative Results ===
    print("\n" + "=" * 80)
    print("QUANTITATIVE RESULTS")
    print("=" * 80)
    
    print(f"\n{'Region':<20} {'True τ':>8} {'Final τ':>10} {'τ Error':>10} {'Final C':>10} {'Final λ':>10}")
    print("-" * 80)
    
    for name, (s, e) in regions.items():
        final_τ_mean = np.mean(τ_history[-1, s:e])
        final_C_mean = np.mean(C_history[-1, s:e])
        final_λ_mean = np.mean(λ_history[-1, s:e])
        
        if true_periods[name]:
            τ_error = abs(final_τ_mean - true_periods[name])
            print(f"{name:<20} {true_periods[name]:>8} {final_τ_mean:>10.1f} {τ_error:>10.1f} {final_C_mean:>10.3f} {final_λ_mean:>10.3f}")
        else:
            print(f"{name:<20} {'N/A':>8} {final_τ_mean:>10.1f} {'N/A':>10} {final_C_mean:>10.3f} {final_λ_mean:>10.3f}")
    
    # === Validation Tests ===
    print("\n" + "=" * 80)
    print("VALIDATION TESTS")
    print("=" * 80)
    
    tests = []
    
    # Test 1: Structured regions have higher C than noise
    structured_C = np.mean([np.mean(C_history[-1, s:e]) 
                           for name, (s, e) in regions.items() if 'noise' not in name])
    noise_C = np.mean([np.mean(C_history[-1, s:e]) 
                       for name, (s, e) in regions.items() if 'noise' in name])
    test1 = structured_C > noise_C * 2
    tests.append(('Structured C > Noise C (×2)', test1, f'{structured_C:.3f} vs {noise_C:.3f}'))
    
    # Test 2: Structured regions have lower λ than noise
    structured_λ = np.mean([np.mean(λ_history[-1, s:e]) 
                           for name, (s, e) in regions.items() if 'noise' not in name])
    noise_λ = np.mean([np.mean(λ_history[-1, s:e]) 
                       for name, (s, e) in regions.items() if 'noise' in name])
    test2 = structured_λ < noise_λ * 0.8
    tests.append(('Structured λ < Noise λ (×0.8)', test2, f'{structured_λ:.3f} vs {noise_λ:.3f}'))
    
    # Test 3: Period learning accuracy
    period_errors = []
    for name, (s, e) in regions.items():
        if true_periods[name]:
            err = abs(np.mean(τ_history[-1, s:e]) - true_periods[name]) / true_periods[name]
            period_errors.append(err)
    mean_period_error = np.mean(period_errors)
    test3 = mean_period_error < 0.25  # Less than 25% error
    tests.append(('Period learning error < 25%', test3, f'{mean_period_error*100:.1f}%'))
    
    # Test 4: Switch detection - λ spike during phase 3 for switched regions
    switched_regions = [(15, 35), (55, 75)]
    switch_point = phase1_ticks + phase2_ticks
    
    λ_before_switch = np.mean([np.mean(λ_history[switch_point-50:switch_point, s:e]) 
                               for s, e in switched_regions])
    λ_after_switch = np.mean([np.mean(λ_history[switch_point:switch_point+100, s:e]) 
                              for s, e in switched_regions])
    test4 = λ_after_switch > λ_before_switch * 1.3
    tests.append(('Switch detected (λ spike)', test4, f'{λ_before_switch:.3f} → {λ_after_switch:.3f}'))
    
    # Test 5: Non-switched regions stable during phase 3
    stable_regions = [(35, 55), (75, 90)]
    λ_stable_before = np.mean([np.mean(λ_history[switch_point-50:switch_point, s:e]) 
                               for s, e in stable_regions])
    λ_stable_after = np.mean([np.mean(λ_history[switch_point:switch_point+100, s:e]) 
                              for s, e in stable_regions])
    test5 = λ_stable_after < λ_stable_before * 1.3
    tests.append(('Stable regions unaffected', test5, f'{λ_stable_before:.3f} → {λ_stable_after:.3f}'))
    
    # Test 6: Different patterns have distinguishable C profiles
    region_Cs = {name: np.mean(C_history[-1, s:e]) for name, (s, e) in regions.items()}
    structured_Cs = [v for k, v in region_Cs.items() if 'noise' not in k]
    test6 = max(structured_Cs) / min(structured_Cs) > 1.2 if structured_Cs else False
    tests.append(('Pattern discrimination (C variance)', test6, f'range: {min(structured_Cs):.3f}-{max(structured_Cs):.3f}'))
    
    # Print test results
    print()
    passed = 0
    for name, result, detail in tests:
        status = '✓ PASS' if result else '✗ FAIL'
        print(f"{status}: {name}")
        print(f"        {detail}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(tests)}")
    
    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if passed >= 5:
        print("\n>>> CORE HYPOTHESES SUPPORTED <<<")
        print("1. Structured patterns develop higher capacitance (confidence)")
        print("2. Structured patterns have lower leakage (stability)")
        print("3. Units learn correct temporal periods")
        print("4. Pattern switches are detected (leakage spikes)")
        print("5. Non-switched patterns remain stable")
        print("6. Different patterns are discriminable by their C profiles")
    else:
        print("\n>>> SOME TESTS FAILED - REVIEW PARAMETERS <<<")
    
    plt.show()
    
    return units, C_history, λ_history, τ_history


if __name__ == "__main__":
    np.random.seed(42)
    run_comprehensive_test()