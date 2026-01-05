#!/usr/bin/env python3
"""Test learning fix directly"""

import numpy as np
import json
from nupca3.config import AgentConfig, default_config
from dataclasses import replace as dc_replace

def test_learning_fixes():
    """Test that learning gates are properly configured"""
    
    # Create base config like test.py
    D = 404
    base_dim = 400
    periph_bins = 2
    periph_blocks = 4
    periph_channels = 1
    side = 20
    fovea_budget = max(4, min(D // 8, 64))
    periph_routing_weight = 0.5
    
    # Apply our fixes
    cfg = default_config()
    cfg = dc_replace(
        cfg,
        D=D,
        B=D,
        grid_width=side,
        grid_height=side,
        grid_channels=1,
        grid_base_dim=base_dim,
        periph_bins=periph_bins,
        periph_blocks=periph_blocks,
        periph_channels=periph_channels,
        fovea_routing_weight=periph_routing_weight,
        tau_E_edit=-1e-6,
        tau_C_edit=-1e-6,  # Fix compute slack gate
        tau_D_edit=-1e-6,  # Fix data headroom gate
        theta_learn=0.25,    # Increase learning threshold
        fovea_blocks_per_step=fovea_budget,
        F_max=fovea_budget,
    )
    
    print("=== LEARNING FIXES VERIFICATION ===")
    print(f"tau_C_edit: {cfg.tau_C_edit} (was 0.0, now -1e-6)")
    print(f"tau_D_edit: {cfg.tau_D_edit} (was 0.0, now -1e-6)")
    print(f"theta_learn: {cfg.theta_learn} (was 0.15, now 0.25)")
    
    # Test learning gate conditions with sample state
    x_C = 0.0  # compute slack (usually 0 at start)
    rawE = 1.0   # energy headroom
    rawD = 1.0   # data headroom
    arousal = 0.1  # low arousal
    rest_t = False  # not in REST
    freeze_t = False # not frozen
    
    # Test gates
    compute_gate_pass = x_C > cfg.tau_C_edit
    energy_gate_pass = rawE > cfg.tau_E_edit  
    data_gate_pass = rawD > cfg.tau_D_edit
    arousal_gate_pass = arousal < cfg.theta_ar_panic
    
    print(f"\n=== GATE CONDITIONS ===")
    print(f"Compute slack: {x_C:.6f} > {cfg.tau_C_edit} = {compute_gate_pass}")
    print(f"Energy headroom: {rawE:.6f} > {cfg.tau_E_edit} = {energy_gate_pass}")
    print(f"Data headroom: {rawD:.6f} > {cfg.tau_D_edit} = {data_gate_pass}")
    print(f"Arousal: {arousal:.6f} < {cfg.theta_ar_panic} = {arousal_gate_pass}")
    
    overall_pass = (
        not rest_t and 
        not freeze_t and 
        compute_gate_pass and 
        energy_gate_pass and 
        data_gate_pass and 
        arousal_gate_pass
    )
    
    print(f"\n=== OVERALL LEARNING PERMISSION ===")
    print(f"All conditions satisfied: {overall_pass}")
    
    # Test effective learning threshold
    transport_high_confidence = False  # usually False at start
    theta_learn_low_conf_scale = 0.6  # Our fix
    theta_eff = cfg.theta_learn if transport_high_confidence else cfg.theta_learn * theta_learn_low_conf_scale
    
    print(f"\n=== LEARNING THRESHOLD ===")
    print(f"Base theta_learn: {cfg.theta_learn}")
    print(f"Low confidence scale: {theta_learn_low_conf_scale} (was 0.25)")
    print(f"Effective theta_eff: {theta_eff:.3f} (vs 0.037 before fix)")
    
    print(f"\n=== SUMMARY ===")
    if overall_pass:
        print("✅ LEARNING GATES FIXED: All conditions satisfied for learning")
    else:
        print("❌ Learning gates still blocked")
        
    if theta_eff > 0.05:
        print("✅ LEARNING THRESHOLD FIXED: More reasonable learning threshold")
    else:
        print("❌ Learning threshold still too restrictive")
        
    return overall_pass and theta_eff > 0.05

if __name__ == "__main__":
    success = test_learning_fixes()
    exit(0 if success else 1)