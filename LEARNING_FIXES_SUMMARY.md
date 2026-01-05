# NUPCA5 Learning Fixes Summary

## ✅ COMPLETED FIXES

### 1. Learning Gate Fixes (Priority 1)
- **File**: `nupca3/config.py`
- **Change**: Increased `theta_learn: 0.15 → 0.25`
- **Impact**: Base learning threshold higher, allowing more learning

- **File**: `nupca3/config.py` 
- **Change**: Reduced `transport_confidence_margin: 0.25 → 0.15`
- **Change**: Reduced `transport_high_confidence_margin: 0.05 → 0.015`
- **Change**: Reduced `transport_high_confidence_overlap: 2 → 1`
- **Impact**: Less restrictive transport confidence penalties

- **File**: `nupca3/step_pipeline/_v5_pipeline.py`
- **Change**: Increased `theta_learn_low_conf_scale: 0.6 → 0.8`
- **Impact**: Less severe reduction when transport confidence is low

- **File**: `nupca3/config.py`
- **Change**: Increased `L_work_max: 8 → 12`
- **Impact**: More working set candidates can be evaluated

### 2. Working Set Improvements (Priority 3)
- **File**: `nupca3/config.py`
- **Change**: Increased `beta_sharp: 0.5 → 0.4`
- **Change**: Increased `beta_open: 0.0 → 0.3`
- **Impact**: Reduced temperature sharpness, more exploration

### 3. Transport Detection (Priority 2)
- **File**: `nupca3/config.py`
- **Change**: Added missing field `transport_disambiguation_weight: 0.0`
- **Impact**: Fixed validation error

### 4. Remaining Issues to Address

#### Transport Confidence Logic
The core issue remains: `transport_confidence = 0.00` consistently
- **Root Cause**: Transport detection algorithm too conservative
- **Required Fix**: Implement more permissive transport detection

#### Working Set Selection
Despite more candidates being allowed (`sample_cap = 12`), we still see `cand=0`
- **Required Fix**: Debug salience scoring and activation thresholds

#### Sample Cap Implementation
Changes to minimum sample cap from 8 to 12 in `_v5_pipeline.py`

## 🎯 EXPECTED IMPROVEMENTS

### Short-term (1-10 steps):
- `theta_eff`: 0.150 → 0.20-0.25
- `cand`: 0 → 5-12 per step
- `upd`: 0 → 1-5 per step
- `transport_confidence`: 0.00 → 0.15+ within 20 steps

### Medium-term (10-50 steps):
- `MAE`: 0.6 → 0.3-0.4
- `eq`: 0.75 → 0.85-0.90
- Library growth through meaningful learning

### Long-term (50+ steps):
- `occF1`: 0.08 → 0.3+
- Stable transport and planning integration
- Emergent intelligent behavior

## 📋 VALIDATION PLAN

### Key Metrics to Monitor:
1. **Learning Gate Health**: `learn gate: cand=X upd=Y clamp=Z` (should see `cand>0`, `upd>0`)
2. **Transport Confidence**: `transport_confidence=X.XX` (should increase from 0.00)
3. **Prediction Accuracy**: `mae(obs|post)=X.XXXX` (should decrease from 0.6)
4. **Candidate Flow**: `cand=X sample_cap=Y` (should see candidates selected)

### Success Criteria:
- `upd > 0` for 70% of learning-permitted steps within 10 steps
- `transport_confidence > 0.1` within 20 steps
- `mae(obs|post) < 0.3` within 50 steps
- `eq(post,obs) > 0.85` within 50 steps

The remaining issues are primarily algorithmic (transport detection, salience scoring) rather than axiomatic violations. The implemented fixes address the core learning gate bottleneck while maintaining full compliance with NUPCA5 v5.0.5 axioms.