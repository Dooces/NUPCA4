# NUPCA3 Axioms — v1.6.2


## Immediate structural risks to flag (because these axioms sit upstream of everything)

1. **Underspecification risk via undefined observables.** If (E(t), D(t), \text{opp}(t), \text{drift}*P(t), E*{pred}(t)) are not rigorously defined in the domain, A0–A2 become “free parameters in disguise,” and later gating behavior can become arbitrary or untestable. 
2. **Unit-consistency risk for compute slack.** (B_{rt}), (b_{enc}), (b_{roll}), (b_{cons}) must be in consistent units; otherwise slack-based control will oscillate or deadlock. 
3. **Normalization constants as hidden control knobs.** (\sigma_E,\sigma_D) (and any implicit scales for (x_S,x_L,x_C)) determine the relative sensitivity of arousal and need computations. If they are not calibrated, the “physiology” can be dominated by the wrong channel. 


This document is the **v1.6.2** axiom system for the **Fixed‑Budget Emergent‑Horizon Adaptive Agent** (A0–A17).

v1.6 = v1.5b **plus**:
- Foveated rolling pixel buffer semantics (A16.5): a bounded, fovea‑wide raw‑observation buffer is permitted for short‑horizon reconstruction/audit during REST, and is **purged before leaving REST** (no pixels after REST).

v1.5b = v1.4b **plus**:
- MERGE acceptance semantics fix (A12.3): per-domain non-worsening error test (replacement-consistent).
- Cold-storage retrieval made explicit (A4.3): block-keyed retrieval driven by greedy_cov residual/coverage scores (A16).
- Derived diagnostics (A17): 3-term “feel proxy” \(q(t)\) is defined as a logged observable with **no control authority**.

<!-- AXIOM_CLARIFICATION_ADDENDUM -->

## Clarification Addendum — Abstractions, Pixels, and "Constellations" (v1.6 intent)

This addendum is *not* a change to A0–A17; it makes an implementation-critical interpretation explicit to prevent a common failure mode in early prototypes.

1. **Two representational layers**
   - **Raw observation** \(y(t)\): e.g., pixels or other high-bandwidth sensory input.
   - **Internal state** \(x(t)\): an **encoded, multi-resolution abstraction vector** that is the sole argument to completion/fusion (A7) and to all stored library parameters \((m_j, W_j, b_j, \Sigma_j)\).


2. **No long-term pixel storage (REST-purged; foveated rolling buffer permitted)**
   - Pixels may exist only in a **bounded, rolling raw-observation buffer** used to form \(x(t)\) and to compute losses.
   - The permitted raw buffer is **foveated** (stores only the currently observed receptive field and its mask/geometry), so a narrow focus yields a longer temporal window under a fixed bit budget.
   - This raw buffer may be **consumed during REST** for reconstruction/audit and structural-edit evaluation, but it must be **purged before leaving REST**.
   - Pixels must never be inserted into the library/cold storage, and must not be present in any post-REST durable state (no "sleep with pixels").



3. **"Experts" are abstractions / resonance operators**
   - The code label "expert" is not intended to import standard MoE semantics.
   - A node is a footprint-local operator that *factors out* a resonant/predictable component; the residual is what remains to be abstracted (A16/A12.4).

4. **Current implementation status (known deviations)**
   - The current codebase treats \(x(t)\) as a generic dense vector and provides helpers (e.g., stale persistence) that *would* constitute pixel storage if \(x\) were instantiated as raw pixels.
   - Until an explicit encoder boundary is implemented, any experiment that feeds raw pixels directly into \(x\) should be regarded as **non-compliant with this addendum**, and performance claims should be scoped accordingly.



<!-- AXIOM_CLARIFICATION_TIMING_NAMING -->

## Clarification Addendum — Temporal Indexing, Residuals, and Naming (v1.6 intent)

This addendum is *not* a change to A0–A17. It makes the intended **time indexing** and **symbol naming** explicit so implementations do not silently drift.

### 1) Time indexing convention

At discrete step \(t\):

1. The agent enters the step with **lagged** state and predicates from \(t-1\): \(\text{rest\_permitted}(t-1)\), \(\text{demand}(t-1)\), \(\text{interrupt}(t-1)\), \(u_j(t-1)\), \(s^{ar}(t-1)\), and \(s^{th}_{ext}(t-1)\).
2. The agent computes **mode** via A14:
   \[
   \text{rest}(t)=\text{rest\_permitted}(t-1)\cdot \text{demand}(t-1)\cdot \big(1-\text{interrupt}(t-1)\big).
   \]
3. The environment reveals a **partial cue** on \(\mathcal{O}_t\) (A16.5). The agent forms a one-step predictive prior \(\hat x(t\mid t-1)\) from the previous completed state.
4. Completion/perception (A13) overwrites observed dims to produce the **completed** state \(x(t)\).
5. **Residual** is defined (A16.2, A17.1) as prediction error on the observed set:
   \[
   e(t)[k]=x(t)[k]-\hat x(t\mid t-1)[k],\quad k\in \mathcal{O}_t.
   \]
   Any “residual” definition based on \(\Delta x(t)=x(t)-x(t-1)\) is **non-compliant** with v1.6.

### 2) Lag discipline summary (non-exhaustive)

- Implementation-only: any option to evaluate gates on same-step signals is **non-axiom** and must default to lagged (t-1) semantics.

- **A14 mode**: \(\text{rest}(t)\) uses predicates from \(t-1\).
- **A10 freeze**: \(\text{freeze}(t)=\mathbf{1}\{s^{th}_{ext}(t-1)\ge \chi^{th}\}\).
- **A5 activation**: \(a_j(t)\) is computed from \(u_j(t-1)\) (via A5.4) and \(\tau^{eff}(t)\) (A5.5).
- **A8 commitment**: \(\text{commit}(t)\) uses confidence at the latency floor from \(t-1\) rollouts.

### 3) Observation blocks and footprints

- A16.1 partitions the \(D\) dimensions into \(B\) disjoint blocks \(\mathcal{B}_b\). When \(D\) is not divisible by \(B\), the partition is still valid; blocks may differ in size by at most 1.
- A4.4 footprints use these same block IDs: for any non-anchor expert \(j\), \(\mathrm{supp}(m_j)\subseteq \mathcal{B}_b\) for exactly one \(b\), and \(\phi(j)=b\).

### 4) Naming conventions (to avoid collisions)

- \(\tau_E,\tau_D,\tau_S\) in A0.3 are **need thresholds** (arguments to sigmoids), *not time constants*.
- Time constants are written explicitly as \(\tau_{\mathrm{rise}},\tau_{\mathrm{decay}},T_{\min},T_{\max},\tau_C^{edit},\tau_E^{edit},\tau_D^{edit}\), etc.
- Slope parameters are \(\kappa_*\); baseline update rates use \(\beta\), \(\beta_{slow}\).

---

## A0 — State, Margins, Two-Channel Stress

### A0.1 — Margins

$$v(t) = (m_E, m_D, m_L, m_C, m_S) \in \mathbb{R}^5$$

| Margin | Measures | Restored By |
|--------|----------|-------------|
| $m_E$ | Operational capacity | REST |
| $m_D$ | Stability | REST |
| $m_L$ | Learning opportunity | — |
| $m_C$ | Compute slack | Compression |
| $m_S$ | Semantic integrity | REST |

### A0.2 — Arousal (Fast Channel; Hormonal / Leaky)

$$A(t) = w_L|\tilde{m}_L(t)| + w_C|\tilde{m}_C(t)| + w_S|\tilde{m}_S(t)| + w_\Delta|\Delta\tilde{m}(t)| + w_E E_{pred}(t)$$

Define instantaneous arousal:
\[
s_{\mathrm{inst}}^{ar}(t)=\sigma\!\left(\frac{A(t)-\theta_{ar}}{\kappa^{ar}}\right),\qquad \sigma(x)=\frac{1}{1+e^{-x}}
\]

Define hormonal / leaky arousal state (fast rise, slow decay):
\[
s^{ar}(t)=(1-\alpha(t))\,s^{ar}(t-1)+\alpha(t)\,s_{\mathrm{inst}}^{ar}(t)
\]
\[
\alpha(t)=
\begin{cases}
1/\tau_{\mathrm{rise}} & \text{if } s_{\mathrm{inst}}^{ar}(t)>s^{ar}(t-1)\\
1/\tau_{\mathrm{decay}} & \text{otherwise}
\end{cases}
\quad\text{with }\tau_{\mathrm{rise}},\tau_{\mathrm{decay}}\ge 1.
\]

Initialization: \(s^{ar}(0)=s_{\mathrm{inst}}^{ar}(0)\).

### A0.3 —### A0.3 — Need/Deficit vs External Threat (Slow Channel)

**Internal need/deficit signals** (homeostatic boundary proximity; drive toward REST):

$$s_E^{need}(t) = \sigma\left(-\frac{(E(t) - E_{\min}) - \tau_E}{\kappa_E}\right)$$

$$s_D^{need}(t) = \sigma\left(-\frac{(D_{\max} - D(t)) - \tau_D}{\kappa_D}\right)$$

$$s_S^{need}(t) = \sigma\left(-\frac{(S_{\max}^{stab} - \text{drift}_P(t)) - \tau_S}{\kappa_S}\right)$$

$$s_{int}^{need}(t) = \max(s_E^{need}(t), s_D^{need}(t), s_S^{need}(t))$$

**External threat** (domain-defined):

$$s_{ext}^{th}(t) = f_{ext}(\text{environmental danger signals}) \in [0,1]$$

---

## A1 — Viability

$$v(t) \in \mathcal{V} \iff Av(t) \geq b$$

---

## A2 — Margins from Observables

### A2.1 — Hard Margins

$$\text{rawE}(t) = E(t) - E_{\min}, \qquad m_E(t) = \frac{\text{rawE}(t)}{\sigma_E}$$

$$\text{rawD}(t) = D_{\max} - D(t), \qquad m_D(t) = \frac{\text{rawD}(t)}{\sigma_D}$$

### A2.2 — Semantic Headroom

$$\text{rawS}(t) = S_{\max}^{stab} - \text{drift}_P(t), \qquad x_S(t) = \text{rawS}(t)$$

### A2.3 — Learning Opportunity

$$x_L(t) = \text{opp}(t)$$

### A2.4 — Compute Slack

$$x_C(t) = B_{rt} - \left(b_{enc}(t) + (1-\text{rest}(t)) \cdot h(t) \cdot b_{roll}(t) + \text{rest}(t) \cdot b_{cons}(t)\right)$$

where:
- In OPERATING: pays rollout cost $h(t)\cdot b_{roll}(t)$
- In REST: pays consolidation cost $b_{cons}(t)$

---

## A3 — Baselines and Normalization

### A3.1 — Stability Predicate

$$\text{stable}(t) = \neg\text{edits}_{struct}[t-W, t] \;\wedge\; \text{Var}(\text{probe}) \leq \nu_{\max} \;\wedge\; \text{Var}_W(x) \leq \xi_{\max}$$

### A3.2 — Baseline Updates (when stable)

$$\mu_k(t) = (1-\beta)\mu_k(t-1) + \beta \cdot x_k(t)$$

### A3.3 — Weber Normalization for L

$$\tilde{m}_L(t) = \frac{x_L(t) - \mu_L(t-1)}{|\mu_L(t-1)| + \varepsilon}$$

### A3.4 — Multi-Scale Variance for C, S

$$(\sigma_k^{fast})^2(t) = (1-\beta)(\sigma_k^{fast})^2(t-1) + \beta(x_k(t) - \mu_k(t-1))^2$$

$$(\sigma_k^{slow})^2(t) = (1-\beta_{slow})(\sigma_k^{slow})^2(t-1) + \beta_{slow}(x_k(t) - \mu_k(t-1))^2$$

$$\rho_k(t) = \frac{\sigma_k^{fast}(t)}{\sigma_k^{slow}(t) + \varepsilon}$$

$$\sigma_k^{eff}(t) = \sigma_k^{fast}(t) + \gamma_{calm} \cdot \sigma_k^{slow}(t) \cdot \text{softplus}(\rho_{min} - \rho_k(t))$$

$$\tilde{m}_k(t) = \frac{x_k(t) - \mu_k(t-1)}{\sigma_k^{eff}(t-1)}$$

### A3.5 — Freeze Escape

If frozen for $\tau > T_{\max}$: fast $\beta$, safe policy, no edits until stable returns.

---

## A4 — Memory Substrate

### A4.1 — DAG of Masked Predictive Experts

Each node $j$:
- $m_j \in \{0,1\}^D$: mask
- $W_j, b_j$: linear dynamics
- $\Sigma_j \in \mathbb{R}^D_{>0}$: per-dimension variance
- $\pi_j \in (0,1]$: reliability
- $L_j > 0$: compute cost

### A4.2 — Bounded Working Set

$$|\mathcal{A}_t| \leq N_{\max}, \qquad \sum_{j \in \mathcal{A}_t} L_j \leq L_{\max}^{work}$$

\
\
### A4.3 — Cold Storage (Block‑Keyed Retrieval)

Inactive nodes \(V \setminus \mathcal{A}_t\) are archived. Retrieval is **block‑keyed** (DoF‑aligned) so that recall is driven by the **global per‑block residual/coverage state** (A16), not by incidental overlap with the currently active set.

Let the coverage‑disciplined block score (A16.3) be
\[
S(b,t)=r(b,t)+\alpha_{cov}\log\!\big(1+\text{age}(b,t)\big).
\]

Let \(\mathcal{F}_t\) be the (coverage‑disciplined) fovea block set selected by A16.3 using \(S(b,t-1)\). Retrieval uses the same block keys:

Using the footprint/incumbent discipline (A4.4), where \(\phi(j)=\text{block-id}(m_j)\) and \(\mathcal{I}_\phi\) is the finite incumbent set per footprint, define the retrieved candidate pool:
\[
\mathcal{C}^{ret}_t := \bigcup_{b\in \mathcal{F}_t}\ \Big(\mathcal{I}_b \cap (V\setminus \mathcal{A}_{t-1})\Big).
\]

Operationally, **retrieval proposes** \(\mathcal{C}^{ret}_t\) as additional candidates; selection into the active working set remains governed by A5.4 (and A4.2 bounds). Concretely, the universe passed to \(\text{GreedySelect}(\cdot)\) at time \(t\) is:
\[
\mathcal{U}_t := \mathcal{A}_{t-1}\ \cup\ \mathcal{C}^{ret}_t\ \cup\ \mathcal{A}^{anchor},
\]
with \(\mathcal{A}^{anchor}\) the always‑on anchors (if any).

### A4.4 — Anti‑Aliasing Insertion Discipline (Disambiguation‑by‑Storage)

**Principle:** the library must not retain two incumbents that are *indistinguishable under the system’s own cue/footprint basis*.
Ambiguity is eliminated **at insertion/consolidation time**, not deferred to retrieval.

Define a **footprint key**:
\[
\phi(j) := \text{block-id}(m_j)
\]
where masks are **DoF‑block‑aligned** (see A16): each non‑anchor expert’s mask lies wholly within exactly one block.

Within each footprint \(\phi\), maintain a finite incumbent set \(\mathcal{I}_\phi\subset V\).

Define a **distinguishability test** between two experts \(p,q\) sharing the same footprint:
\[
\Delta_{\phi}(p,q) := \mathbb{E}_{t\in \mathcal{T}_\phi}\left[\ \| \mu_p(t+1|t) - \mu_q(t+1|t)\|_{1,\phi}\ \right]
\]
where \(\mathcal{T}_\phi\) is the set of timesteps in which block \(\phi\) was observed under coverage (A16), and \(\|\cdot\|_{1,\phi}\) denotes restriction to the block dimensions.

**Anti‑alias rule:** when proposing a new expert \(j_{\text{new}}\) for footprint \(\phi\),

- If \(\exists i\in\mathcal{I}_\phi\) such that \(\Delta_\phi(i,j_{\text{new}}) < \theta_{\text{alias}}\), then \(j_{\text{new}}\) is **aliased** with \(i\) and may not be retained as a separate incumbent.
  - If \(j_{\text{new}}\) is strictly better on the same evidence (higher likelihood / lower error under the acceptance tests), then **REPLACE** the aliased incumbent: \(i \leftarrow j_{\text{new}}\).
  - Otherwise **REJECT** \(j_{\text{new}}\).

- If \(j_{\text{new}}\) is non‑aliased with all incumbents (\(\Delta_\phi(i,j_{\text{new}})\ge \theta_{\text{alias}}\ \forall i\in\mathcal{I}_\phi\)), then it may be added to \(\mathcal{I}_\phi\) subject to structural‑edit acceptance (A12.3–A12.4) and working‑set constraints (A4.2).

This enforces: “you can’t store two overlapping/ambiguous block incumbents; the newer/better one dominates.”



---

## A5 — Salience (Arousal-Sharpened)

### A5.1 — Score

$$u_j(t) = \alpha_\pi \bar{\pi}_j(t) + \alpha_{deg} \frac{\deg^+_j}{\deg^+_{\max}} + \alpha_{ctx} \cdot \text{relevance}(x(t), j)$$

### A5.2 — Temperature (Need‑Sharpened, Play‑Opened)

Define a safe “play/approach” factor (lagged to preserve timing discipline):
\[
s^{play}(t-1)= s^{ar}(t-1)\cdot\bigl(1-s_{int}^{need}(t-1)\bigr)\cdot\bigl(1-s_{ext}^{th}(t-1)\bigr).
\]

Temperature is sharpened by internal need (focus under deficit) and opened by play/arousal when safe:
\[
\tau^{eff}(t)=\frac{\tau_a}{1+\beta_{sharp}\,s_{int}^{need}(t-1)}\cdot\left(1+\beta_{open}\,s^{play}(t-1)\right),
\qquad \beta_{open}\ge 0.
\]

### A5.3 — Salience

$$a_j(t) = \sigma\left(\frac{u_j(t-1) - \theta_a}{\tau^{eff}(t)}\right)$$

### A5.4 — Active Set

$$\mathcal{A}_t = \text{GreedySelect}\left(\{j\}, \frac{a_j(t) \cdot \pi_j(t)}{L_j}, \sum L_j \leq L_{\max}^{work}\right)$$

### A5.5 — Effective Complexity

$$L^{eff}(t) = \sum_{j \in \mathcal{A}_t} a_j(t) \cdot L_j$$

### A5.6 — Anchor Guardrail (Replaced)

**Force-inclusion semantics:**

1. Initialize $\mathcal{A}_t \leftarrow P_{anchor} \cap \{j : \text{alive}(j)\}$
2. Verify $\sum_{j \in \mathcal{A}_t} L_j \leq L_{max}^{work}$ (anchors must fit within budget)
3. Run GreedySelect over non-anchor candidates to fill remaining capacity:

$$\mathcal{A}_t \leftarrow \mathcal{A}_t \cup \text{GreedySelect}\left(\{j \notin P_{anchor}\}, \frac{a_j(t) \cdot \pi_j(t)}{L_j}, \sum_{j \in \mathcal{A}_t} L_j \leq L_{max}^{work}\right)$$

---

## A6 — Fixed Budget, Emergent Horizon (Amended)

### A6.1 — Budget

$$B_{rt} > 0 \quad \text{(constant)}$$

### A6.2 — Load Decomposition and Costs

**Anchor vs. rollout load (Option B semantics):**

$$L^{eff}_{anc}(t) = \sum_{j \in \mathcal{A}_t \cap P_{anchor}} a_j(t) \cdot L_j$$

$$L^{eff}_{roll}(t) = \sum_{j \in \mathcal{A}_t \setminus P_{anchor}} a_j(t) \cdot L_j$$

$$L^{eff}(t) = L^{eff}_{anc}(t) + L^{eff}_{roll}(t)$$

**Encoding cost (includes anchor overhead, paid once per step):**

$$b_{enc}(t) = b_{enc,0} + b_{anc,0}(\varepsilon + L^{eff}_{anc}(t))$$

**Rollout cost (excludes anchors, paid per horizon step):**

$$b_{roll}(t) = b_{roll,0}(\varepsilon + L^{eff}_{roll}(t))$$

**Consolidation cost (REST only):**

$$b_{cons}(t) = \text{cost of structural edits processed at } t$$

### A6.3 — Horizon

$$h(t) = \begin{cases} 0 & \text{if } \text{rest}(t) = 1 \\ \max\left(0, \left\lfloor \frac{B_{rt} - b_{enc}(t)}{b_{roll}(t)} \right\rfloor\right) & \text{if } \text{rest}(t) = 0 \end{cases}$$

---

## A7 — Completion and Fusion

### A7.1 — Per-Node Prediction

$$\mu_j(t+1|t) = W_j x(t) + b_j$$

$$\Lambda_j[k] = \begin{cases} 1/\Sigma_j[k] & \text{if } m_j[k] = 1 \\ 0 & \text{if } m_j[k] = 0 \end{cases}$$

### A7.2 — Coverage Invariant

$$\forall k: \sum_{j \in \mathcal{A}_t} \Lambda_j[k] > 0$$

If violated for dimension $k$: $\hat{x}[k] = x(t)[k]$ (persist), $\Sigma_{global}[k] = \infty$.

### A7.3 — Precision-Weighted Fusion

$$\hat{x}(t+1|t)[k] = \frac{\sum_{j \in \mathcal{A}_t} \Lambda_j[k] \cdot \mu_j(t+1|t)[k]}{\sum_{j \in \mathcal{A}_t} \Lambda_j[k]}$$

$$\Sigma_{global}(t+1|t)[k] = \left(\sum_{j \in \mathcal{A}_t} \Lambda_j[k]\right)^{-1}$$

### A7.4 — Horizon Confidence (Replaced)

**Rollout semantics:**

Apply A7.1–A7.3 iteratively using fixed $\mathcal{A}_t$ and fixed $(W_j, b_j, \Sigma_j)$:

$$\hat{x}(t+1|t) \text{ from A7.3}$$
$$\hat{x}(t+k|t) = \text{fusion applied to } \hat{x}(t+k-1|t)$$

**Uncertainty growth:**

$$\Sigma_{global}(t+1|t) = \left(\sum_{j \in \mathcal{A}_t} \Lambda_j\right)^{-1}$$

$$\Sigma_{global}(t+\ell+1|t) = \Sigma_{global}(t+\ell|t) + \Sigma_{proc}$$

with $\Sigma_{proc}[k] = \eta_{proc} \cdot \Sigma_{global}(t+1|t)[k]$, $\eta_{proc} > 0$.

**Confidence:**

$$H_k(t) = \text{mean}(\Sigma_{global}(t+k|t))$$

$$c_k(t) = \sigma\left(-\frac{H_k(t) - \mu_H}{\sigma_H}\right)$$

---

## A8 — Latency and Commitment

### A8.1 — Latency Floor

$$d = \lceil T_{proc} / \Delta t \rceil$$

### A8.2 — Commitment Gate

$$\text{commit}(t) = \mathbf{1}\{\text{rest}(t) = 0\} \cdot \mathbf{1}\{h(t) \geq d\} \cdot \mathbf{1}\{c_d(t-1) \geq \theta_{act}\}$$

### A8.3 — Action

- commit = 1: execute planned action
- commit = 0: execute safe/reflex policy $\pi_{safe}$

---

## A9 — Prediction Objective

### A9.1 — Loss

$$\mathcal{E}_k(t) = \mathbb{E}[-\log p(x(t+k) | \hat{x}(t+k|t))]$$

### A9.2 — Objective

$$\lambda(t) = \sigma(w_\lambda \cdot s^{ar}(t-1) + b_\lambda)$$

$$J_{pred}(t) = (1-\text{rest}(t)) \left[(1 - \lambda(t)) \cdot \ell_\tau(\mathcal{E}_d(t)) + \lambda(t) \sum_{k=d+1}^{h(t)} w(k) \cdot \ell_\tau(\mathcal{E}_k(t))\right]$$

### A9.3 — Curiosity

$$J_{explore}(t) = (1-\text{rest}(t)) \cdot \gamma \sum_{k=d+1}^{h(t)} (1 - c_k(t)) \cdot \mathcal{E}_k(t) \quad \text{if } s_{ext}^{th}(t-1) < \theta_{safe}$$

---

## A10 — Edit Control

### A10.1 — Freeze (External Threat Only)

$$\text{freeze}(t) = \mathbf{1}\{s_{ext}^{th}(t-1) \geq \chi^{th}\}$$

### A10.2 — Parameter Update Permission (Approach‑Arousal Compatible)

Parameter updates are permitted in OPERATING when not frozen, with compute slack and margin headroom.
Arousal is **not** a “must be calm” gate; it blocks updates only under near‑saturation (“panic cap”).

\[
\text{permit}_{param}(t)=
\mathbf{1}\{\text{rest}(t)=0\}\cdot
\mathbf{1}\{\text{freeze}(t)=0\}\cdot
\mathbf{1}\{x_C(t-1)>\tau_C^{edit}\}\cdot
\mathbf{1}\{s^{ar}(t-1)<\theta_{ar}^{panic}\}\cdot
\mathbf{1}\{\text{rawE}(t-1)>\tau_E^{edit}\}\cdot
\mathbf{1}\{\text{rawD}(t-1)>\tau_D^{edit}\}.
\]

### A10.3 —### A10.3 — Responsibility‑Gated Parameter Learning (Prevents Off‑Context Corruption)

Parameter updates are only applied to experts that are **responsible** for the observed footprint at \(t\).

Let \(\mathcal{O}_t\) be the set of observed dimensions (from foveation; A16). For expert \(j\), define:
\[
\text{obs}_j(t)=\mathbf{1}\{\exists k\in \mathcal{O}_t:\ m_j[k]=1\}
\]

Define a per‑expert local prediction error on its observed footprint:
\[
\text{err}_j(t)=\text{mean}_{k\in \mathcal{O}_t}\Big( m_j[k]\cdot |x(t)[k]-\hat{x}(t|t-1)[k]| \Big)
\]

Define responsibility:
\[
\text{resp}_j(t)=\mathbf{1}\{j\in\mathcal{A}_t\}\cdot \text{obs}_j(t)\cdot \mathbf{1}\{\text{err}_j(t)\le \theta_{\text{learn}}\}
\]

**Update rule:** if \(\text{permit}_{param}(t)=1\), then parameter learning may occur only for experts with \(\text{resp}_j(t)=1\).
Experts with \(\text{resp}_j(t)=0\) must not have \((W_j,b_j)\) updated at \(t\), but their reliability \(\pi_j\) may still be adjusted by performance tracking (decrease under sustained error).

This is the explicit formalization of “don’t rewrite an incumbent under the wrong context; punish it, don’t corrupt it.”



---

## A11 — Semantic Integrity

$$\text{drift}_P(t) = \frac{1}{|P|} \sum_{p \in P} |f_p(t) - f_p^{ref}|$$

$$\Delta S(e) = g(\Delta\text{perf}_P, \text{interference}_P, \Delta\text{drift}_P)$$

---

## A12 — Edit Acceptance

### A12.1 — Net Value

$$\Delta J(e) = \Delta F(e) - \beta \cdot \Delta L^{MDL}(e)$$

### A12.2 — Parameter Updates

Accept if:

$$\text{permit}_{param}(t) \;\wedge\; q_e \geq q_{\min}(t) \;\wedge\; \Delta J(e) \geq \varepsilon(t) \;\wedge\; \Delta S(e) \leq S_{\max}^{sem}(t) \;\wedge\; \Delta C(e) \leq C_{\max}(t)$$

\
### A12.3 — Structural Edits (REST Only)

$$\text{permit}_{struct}(t) = \mathbf{1}\{\text{rest}(t) = 1\} \cdot \mathbf{1}\{s^{ar}(t-1) < \theta_{ar}^{rest}\}$$

**MERGE(A, B → C)** if:
- Correlation($a_A, a_B$) > $\theta_{merge}$
- $L_C < L_A + L_B$
- **Replacement‑consistent error test (per‑domain non‑worsening):**

  Structural non‑anchor experts are DoF‑block‑aligned (A16). Accordingly, MERGE is defined only within a shared footprint:
  \[
  \phi(A)=\phi(B)=\phi(C).
  \]

  Let \(\mathcal{T}_{AB}\) be a finite evaluation index set (e.g., the most recent timesteps) such that the shared block \(\phi(A)\) is observed at \(\tau\) and \(\tau+1\) (so targets exist), and the pre‑merge library permits computing \(a_A(\tau)\) and \(a_B(\tau)\).g., the most recent timesteps) such that the shared block \(\phi(A)\) was observed and the pre‑merge library permits computing \(a_A(\tau)\) and \(a_B(\tau)\).

  Partition \(\mathcal{T}_{AB}\) using **pre‑merge** activations:
  \[
  \mathcal{T}_A := \{\tau\in\mathcal{T}_{AB}: a_A(\tau)\ge a_B(\tau)\},\qquad
  \mathcal{T}_B := \mathcal{T}_{AB}\setminus \mathcal{T}_A.
  \]
  Require \(|\mathcal{T}_A|\ge 1\) and \(|\mathcal{T}_B|\ge 1\).

  Define per‑step masked squared error for expert \(j\in\{A,B,C\}\):
  \[
  \mathrm{se}_j(\tau) :=
  \mathrm{mean}_{k\in \mathrm{supp}(m_A)\cap \mathcal{O}_{\tau+1}}
  \Big(x(\tau+1)[k]-\mu_j(\tau+1\mid \tau)[k]\Big)^2,
  \]
  where the comparison is restricted to the shared footprint mask (\(\mathrm{supp}(m_A)=\mathrm{supp}(m_B)=\mathrm{supp}(m_C)\)).

  Define \(\mathrm{MSE}_j(\mathcal{T}) := \frac{1}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}}\mathrm{se}_j(\tau)\).

  Accept the merge only if:
  \[
  \mathrm{MSE}_C(\mathcal{T}_A)\le \mathrm{MSE}_A(\mathcal{T}_A)+\varepsilon_{merge}
  \quad\wedge\quad
  \mathrm{MSE}_C(\mathcal{T}_B)\le \mathrm{MSE}_B(\mathcal{T}_B)+\varepsilon_{merge}.
  \]

**PRUNE(j)** if:
- $\pi_j < \theta_{cull}$ OR TimeSinceActive(j) > $T_{inactive}$


### A12.4 — Structural Growth (SPAWN/SPLIT) from Persistent Residual (REST Only)

Structural growth is permitted only in REST (A12.3), and must obey the same compute and semantic constraints (A12.1–A12.3).

Define a per‑footprint persistent residual statistic (computed under coverage; A16):
\[
R_\phi(t)= (1-\beta_R)R_\phi(t-1)+\beta_R\cdot \text{residual\_block}(\phi,t)
\]
where \(\text{residual\_block}(\phi,t)\) is the mean absolute residual on the block \(\phi\) when \(\phi\) is observed.

A **SPAWN** proposal for footprint \(\phi\) is generated when, over \(K\) distinct coverage visits to \(\phi\),
\[
R_\phi(t)>\theta_{\text{spawn}}
\]
and no existing incumbent in \(\mathcal{I}_\phi\) reduces \(R_\phi\) below \(\theta_{\text{spawn}}\).

A SPAWN creates a new expert \(j_{\text{new}}\) with mask \(m_{j_{\text{new}}}\) aligned to \(\phi\), initialized by fitting recent observed transitions on \(\phi\) (any local solver consistent with the substrate class).

Acceptance:
- must satisfy \(\text{permit}_{struct}(t)=1\),
- must satisfy the standard acceptance constraints (A12.1, A12.3) for \(\Delta J, \Delta S, \Delta C\),
- must pass **anti‑aliasing** (A4.4): if aliased with an incumbent, it REPLACE/REJECTs rather than coexisting.

A **SPLIT** is a special case of SPAWN that reduces footprint scope by producing two experts whose masks partition \(\phi\)’s dimensions (still DoF‑block‑aligned), accepted only if it reduces error and total cost under the same acceptance tests.

This makes “footprint vocabulary expands as new structure is encountered” explicit within the REST‑only structural edit regime.



---

## A13 — Unified Completion

Perception, recall, prediction use A7 with different cues:
- Perception: Dense external, $k = d$
- Prediction: Current state, $k = 1...h(t)$
- Recall: Sparse internal, variable $k$

---

## A14 — Macrostates (Amended)

### A14.1 — Two States

$$\text{rest}(t) \in \{0, 1\}$$

- **REST:** $h=0$, structural edits, margins recover
- **OPERATING:** $h$ from budget, parameter edits, margins deplete

### A14.2 — Queue and Timer Dynamics

$$Q_{struct}(t) = \begin{cases} Q_{struct}(t-1) + \text{proposals}(t) & \text{if OPERATING} \\ \max(0, Q_{struct}(t-1) - \text{edits\_processed}(t)) & \text{if REST} \end{cases}$$

$$T_{since}(t) = \begin{cases} T_{since}(t-1) + 1 & \text{if OPERATING} \\ 0 & \text{if REST} \end{cases}$$

$$T_{rest}(t) = \begin{cases} 0 & \text{if OPERATING} \\ T_{rest}(t-1) + 1 & \text{if REST} \end{cases}$$

### A14.3 — Rest Pressure

$$\Delta P^{wake}(t) = \delta_{base} + \delta_{need} \cdot s_{int}^{need}(t)$$

$$P_{rest}(t) = \begin{cases} P_{rest}(t-1) + \Delta P^{wake}(t) & \text{if OPERATING} \\ (1-\gamma_{rest})P_{rest}(t-1) & \text{if REST} \end{cases}$$

$$P_{rest}^{eff}(t) = P_{rest}(t) + \alpha_E s_E^{need}(t) + \alpha_D s_D^{need}(t) + \alpha_S s_S^{need}(t)$$

### A14.4 — Demand (Hysteresis + Completion-Gated)

**Restoration predicate:**

$$\text{restored}(t) = \mathbf{1}\{m_E(t) \geq \theta^{rest}_E\} \cdot \mathbf{1}\{m_D(t) \geq \theta^{rest}_D\} \cdot \mathbf{1}\{m_S(t) \geq \theta^{rest}_S\}$$

**State-dependent demand:**

$$\text{demand}(t) = \begin{cases} \mathbf{1}\{P_{rest}^{eff}(t) > \theta_{demand}^{enter}\} \vee \mathbf{1}\{Q_{struct}(t) > \Theta_Q^{on}\} \vee \mathbf{1}\{T_{since}(t) > T_{max}^{wake}\} & \text{if } \text{rest}(t) = 0 \\[6pt] \mathbf{1}\{\neg\text{restored}(t)\} \vee \mathbf{1}\{Q_{struct}(t) > \Theta_Q^{off}\} \vee \mathbf{1}\{T_{rest}(t) > T_{max}^{rest}\} & \text{if } \text{rest}(t) = 1 \end{cases}$$

**Threshold constraints:**

$$\theta_{demand}^{enter} > \theta_{demand}^{exit}, \qquad \Theta_Q^{on} > \Theta_Q^{off}$$

### A14.5 — Permission (External Only)

$$\text{rest\_permitted}(t) = \mathbf{1}\{s_{ext}^{th}(t) < \theta_{safe}^{th}\}$$

### A14.6 — Interrupt (External Only)

$$\text{interrupt}(t) = \mathbf{1}\{s_{ext}^{th}(t) \geq \theta_{interrupt}^{th}\}$$

### A14.7 — State Transition (Lagged)

$$\text{rest}(t) = \mathbf{1}\{\text{rest\_permitted}(t-1)\} \cdot \mathbf{1}\{\text{demand}(t-1)\} \cdot \mathbf{1}\{\neg\text{interrupt}(t-1)\}$$

### A14.8 — Initialization

$$\mu_k(0), \sigma_k^{fast}(0), \sigma_k^{slow}(0), P_{rest}(0), Q_{struct}(0), T_{since}(0), T_{rest}(0) = 0$$

Probe references $f_p^{ref}$ set from initial probe outputs.

---

## A15 — Margin Dynamics (New)

### A15.1 — Operating Costs

During OPERATING, margins deplete:

$$\dot{E}^{op}(t) = -c_E^{base} - c_E^{commit} \cdot \text{commit}(t) - c_E^{load} \cdot \frac{L^{eff}(t)}{L_{max}^{work}}$$

$$\dot{D}^{op}(t) = c_D^{base} + c_D^{commit} \cdot \text{commit}(t) + c_D^{load} \cdot \frac{L^{eff}(t)}{L_{max}^{work}}$$

$$\dot{S}^{op}(t) = c_S^{base} + c_S^{commit} \cdot \text{commit}(t) + c_S^{load} \cdot \frac{L^{eff}(t)}{L_{max}^{work}}$$

Note: Energy decreases (depletes toward $E_{min}$), Damage and drift increase (approach their maxima).

### A15.2 — REST Recovery

During REST, margins restore toward nominal:

$$E(t) = E(t-1) + k_E^{rest}(E_{max} - E(t-1))$$

$$D(t) = D(t-1) - k_D^{rest}(D(t-1) - D_{min})$$

$$\text{drift}_P(t) = \text{drift}_P(t-1) - k_S^{rest}(\text{drift}_P(t-1) - 0)$$

### A15.3 — Constraint

Recovery rates must satisfy:

$$k_E^{rest}, k_D^{rest}, k_S^{rest} \ll 1$$

to ensure multi-step REST bouts (margins recover gradually, not instantly).

---

---

A16 — Observation Geometry and Coverage-Disciplined Foveation (greedy_cov)

The environment’s state 
𝑥
(
𝑡
)
∈
𝑅
𝐷
x(t)∈R
D
 is only partially observed each step via a controllable receptive field (“fovea”).

A16.1 — DoF-Aligned Block Partition (1:1)

Partition the 
𝐷
D dimensions into disjoint blocks:

{
1
,
…
,
𝐷
}
=
⨆
𝑏
=
1
𝐵
𝐵
𝑏
,
𝐵
𝑏
∩
𝐵
𝑏
′
=
∅
{1,…,D}=
b=1
⨆
B
	​

B
b
	​

,B
b
	​

∩B
b
′
	​

=∅

Each non-anchor expert mask 
𝑚
𝑗
m
j
	​

 must satisfy:

∃
𝑏
:
 supp
(
𝑚
𝑗
)
⊆
𝐵
𝑏
.
∃b: supp(m
j
	​

)⊆B
b
	​

.
A16.2 — Per-Block Age and Residual

Let 
age
(
𝑏
,
𝑡
)
age(b,t) be the number of steps since block 
𝑏
b was last observed.

Define the per-block instantaneous residual when 
𝑏
b is observed:

𝑟
^
(
𝑏
,
𝑡
)
=
mean
𝑘
∈
𝐵
𝑏
∩
𝑂
𝑡
(
∣
𝑥
(
𝑡
)
[
𝑘
]
−
𝑥
^
(
𝑡
∣
𝑡
−
1
)
[
𝑘
]
∣
)
r
^
(b,t)=mean
k∈B
b
	​

∩O
t
	​

	​

(∣x(t)[k]−
x
^
(t∣t−1)[k]∣)

Maintain a smoothed residual:

𝑟
(
𝑏
,
𝑡
)
=
(
1
−
𝛽
𝑟
)
 
𝑟
(
𝑏
,
𝑡
−
1
)
+
𝛽
𝑟
 
𝑟
^
(
𝑏
,
𝑡
)
when 
𝑏
 observed, else 
𝑟
(
𝑏
,
𝑡
)
=
𝑟
(
𝑏
,
𝑡
−
1
)
.
r(b,t)=(1−β
r
	​

)r(b,t−1)+β
r
	​

r
^
(b,t)when b observed, else r(b,t)=r(b,t−1).
A16.3 — Coverage Debt Score and Selection

At each step select a set of blocks 
𝐹
𝑡
F
t
	​

 to observe (the fovea), with 
∣
𝐹
𝑡
∣
=
𝐹
∣F
t
	​

∣=F.

Define thresholded coverage debt (no debt before age 
𝐺
G):

age
+
(
𝑏
,
𝑡
)
:
=
max
⁡
(
0
,
 age
(
𝑏
,
𝑡
)
−
𝐺
)
.
age
+
(b,t):=max(0, age(b,t)−G).

Score each block using greedy salience + thresholded coverage debt:

score
(
𝑏
,
𝑡
)
=
𝑟
(
𝑏
,
𝑡
−
1
)
+
𝛼
cov
⋅
log
⁡
 ⁣
(
1
+
age
+
(
𝑏
,
𝑡
−
1
)
)
.
score(b,t)=r(b,t−1)+α
cov
	​

⋅log(1+age
+
(b,t−1)).

Greedy_cov policy: choose the 
𝐹
F blocks with highest 
score
(
𝑏
,
𝑡
)
score(b,t), subject to the coverage-aversion semantics below.

A16.4 — Soft Coverage Aversion with Threshold (No Deterministic Bound)

Replace the deterministic “hard coverage cap” with thresholded aversion:

𝐺
G is a coverage-debt onset threshold (as in A16.3), not a maximum age bound.

There is no forced inclusion rule based solely on 
age
(
𝑏
,
𝑡
)
age(b,t).

There is no architectural guarantee that 
age
(
𝑏
,
𝑡
)
age(b,t) is bounded.

(Any reference to “coverage visits” elsewhere means timesteps when the block is included in 
𝐹
𝑡
F
t
	​

; no bounded-reacquisition assumption is implied.)

A16.5 — Observation Set and Memory Buffer

The observed dimensions at 
𝑡
t are:

𝑂
𝑡
=
⋃
𝑏
∈
𝐹
𝑡
𝐵
𝑏
.
O
t
	​

=
b∈F
t
	​

⋃
	​

B
b
	​

.

For unobserved dimensions, the agent retains the last observed value in a buffer 
𝑥
mem
(
𝑡
)
x
mem
	​

(t) (stale persistence). Perception/recall/prediction all operate via A7 completion using the current observed subset as the cue (A13).

Foveated Rolling Pixel Buffer (raw observation; REST-purged):

When the external observation is high-bandwidth (e.g., pixels), the agent may maintain an ephemeral foveated rolling buffer of raw observations, denoted 
𝑦
𝑏
𝑢
𝑓
(
𝑡
)
y
buf
	​

(t), subject to a fixed bit budget 
𝐵
𝑝
𝑖
𝑥
B
pix
	​

.

Each entry stores only the foveated payload at its step (e.g., patches) together with its support/mask (the receptive field geometry) and any minimal alignment metadata required by the domain (e.g., crop coordinates / transform).

Eviction is mandatory under the budget: when 
𝐵
𝑝
𝑖
𝑥
B
pix
	​

 would be exceeded, the buffer must drop past entries (oldest-first or an equivalent deterministic policy).

Permitted uses:

Forming the encoded internal state 
𝑥
(
𝑡
)
x(t) from raw observation.

Computing step-local losses/residual diagnostics.

During REST only: temporary reconstruction/audit of recent foveated observations to score residual structure and validate structural edits.

Prohibited uses (hard constraint):

No raw pixels may be written into the library/cold storage (A4) or any other durable store.

The raw buffer 
𝑦
𝑏
𝑢
𝑓
y
buf
	​

 must be purged before leaving REST; post-REST operation proceeds with abstractions only.



\
## A17 — Derived Diagnostics (No Control Authority)

These quantities are **derived readouts only**. They have **no control authority** and must not be used as gates or triggers; they are logged for analysis/diagnostics.

### A17.1 — Precision‑Weighted Residual

Define the observation residual (on observed dimensions):
\[
e(t)[k] := x(t)[k]-\hat x(t\mid t-1)[k]\quad\text{for }k\in\mathcal{O}_t.
\]
Let \(\Lambda_{global}(t)=1/\Sigma_{global}(t)\) be the fusion precision on observed dimensions (A7.3). Define:
\[
q_{\mathrm{res}}(t):=\mathrm{mean}_{k\in\mathcal{O}_t}\big(|\Lambda_{global}(t)[k]\cdot e(t)[k]|\big).
\]

### A17.2 — Maintenance Load

\[
q_{\mathrm{maint}}(t):=L^{eff}(t)
\]
(A5.5).

### A17.3 — Horizon Uncertainty at the Latency Floor

Let \(d\) be the latency floor and \(H_k(t)\) the rollout uncertainty summary (A7.4). Define:
\[
q_{\mathrm{unc}}(t):=H_d(t).
\]

### A17.4 — 3‑Term “Feel Proxy” (Logged Observable)

\[
q(t):=\big(q_{\mathrm{res}}(t),\ q_{\mathrm{maint}}(t),\ q_{\mathrm{unc}}(t)\big).
\]

## proposed constraints, not gospel but preferred perspective

Constraint (Motion-Primary, Ancillary Fallback):

Primary assumption: The agent’s first-line path to objectness, correspondence, and compositional abstractions is motion-based learning (temporal continuity, tracking, persistence under dynamics).

Ancillary channels:

interventions,

explicit paired transforms,

teacher/LLM-proposed region pointers,
are treated as secondary supports rather than primary bootstraps.

When to use ancillary channels: These supports may be introduced when motion-based learning is insufficient, stalled, or failing for a target regime, as a way to:

reduce ambiguity,

provide targeted supervision,

or accelerate convergence,
without redefining the architecture’s core learning story.

## Reference Configuration (Locked Starting Point)

These are not axioms, but the validated starting point for experiments.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| $B_{rt}$ | 260 | Validated baseline |
| $b_{roll,0}$ | 0.85 | Calibrated to $h \approx 20$–22 |
| $b_{enc,0}$ | 3.2 | Empirically stable |
| $L_{max}^{work}$ | 48 | With 8 anchors @ cost 4 = 32 overhead, leaves 16 for rollout |
| $k_E^{rest}, k_D^{rest}, k_S^{rest}$ | 0.03, 0.02, 0.02 | Slow enough for multi-step bouts |
| $\theta_{demand}^{enter}$, $\theta_{demand}^{exit}$ | 0.90, 0.75 | Hysteresis gap of 0.15 |
| $\theta^{rest}_E, \theta^{rest}_D, \theta^{rest}_S$ | 0.80, 0.85, 0.85 | Completion thresholds |
| $T_{max}^{rest}$ | 900 | Safety valve |
| $\gamma_{rest}$ | 0.07 | Pressure decay rate |
| $\delta_{base}$ | 0.0035 | Calibrated baseline pressure accumulation |
| $\delta_{need}$ | 0.06 | Calibrated internal-need contribution to REST pressure |
| $\alpha_E,\alpha_D,\alpha_S$ | 0.3591, 0.3591, 0.3591 | Calibrated weights for need signals into $P_{rest}^{eff}$ |
| $\tau_E,\tau_D,\tau_S$ | 0.7825, 0.8319, 0.8406 | Calibrated internal need buffers |
| $\kappa_E,\kappa_D,\kappa_S$ | 0.0692, 0.0721, 0.0591 | Calibrated internal need slopes |
| $c_E^{base},c_E^{commit},c_E^{load}$ | 0.002114, 0.001199, 0.001254 | Calibrated operating costs (energy) |
| $c_D^{base},c_D^{commit},c_D^{load}$ | 0.002114, 0.001199, 0.001254 | Calibrated operating costs (damage) |
| $c_S^{base},c_S^{commit},c_S^{load}$ | 0.002114, 0.001199, 0.001254 | Calibrated operating costs (drift) |
| $\theta^{rest}_E,\theta^{rest}_D,\theta^{rest}_S$ space | normalized margins | Clarifies A14.4 thresholds are in $m_k$ space (A2), not raw observables |
| $F$ | 1 block per DoF (domain-set) | Fovea capacity (blocks observed per step) |
| $\alpha_{cov}$ | 0.35 | Coverage-debt weighting in greedy_cov foveation |
| $G$ | 120 | Hard coverage cap (max block age) |
| $\beta_r$ | 0.20 | Residual EMA for block residual tracking |
| $\theta_{alias}$ | 0.04 | Anti-alias distinguishability threshold (per-block) |
| $\theta_{learn}$ | 0.10 | Responsibility learning gate error threshold |
| $\beta_R$ | 0.10 | Persistent residual EMA for SPAWN trigger |
| $K$ | 3 coverage visits | Minimum evidence for SPAWN proposal |
| $\theta_{spawn}$ | 0.25 | Persistent residual threshold for SPAWN |
| $\tau_{\mathrm{rise}}$ | 2 | Hormonal arousal: fast rise (steps) |
| $\tau_{\mathrm{decay}}$ | 12 | Hormonal arousal: slow decay / hangover (steps) |
| $\theta_{ar}^{panic}$ | 0.95 | Arousal “panic cap” for parameter updates (A10.2) |
| $\beta_{open}$ | 1.0 | Play/approach broadening gain in temperature (A5.2) |


Below is an interpretation of **A0–A2** as they are written in your v1.6 axiom document. 

---

## A0 — State, Margins, Two-Channel Stress

### What it means / what it does

**A0 declares the agent’s internal “physiology.”** It says the agent does not operate solely on task loss; it carries (and updates every step) a small vector of **margins** plus two distinct “stress” channels that later axioms use as gates and modulators. 

1. **Margins as a compact viability summary**
   You define:
   [
   v(t)=(m_E,m_D,m_L,m_C,m_S)\in\mathbb{R}^5
   ]
   These are *normalized* scalars that stand in for “how close to trouble” the agent is along five axes (energy/capacity, damage/stability, learning opportunity, compute slack, semantic integrity). They are not just diagnostics; later axioms treat them as control-relevant state. 

2. **Fast stress channel: arousal (s^{ar}(t))**
   A0.2 defines an instantaneous arousal drive (A(t)) that is a weighted mix of (i) magnitudes of normalized deviations (|\tilde m_L|,|\tilde m_C|,|\tilde m_S|), (ii) a change term (|\Delta\tilde m|), and (iii) a predictive-error term (E_{pred}(t)). This is squashed by a sigmoid to (s_{\text{inst}}^{ar}(t)), then filtered with **fast-rise / slow-decay** dynamics:

* fast onset when the instantaneous arousal increases,
* slow decay otherwise.

Operationally: arousal becomes a *memoryful urgency/engagement signal* that reacts quickly to shocks and lingers (hysteresis). 

3. **Slow stress channel: internal need vs external threat**
   A0.3 explicitly separates:

* **Internal need/deficit** signals (s^{need}_E,s^{need}_D,s^{need}*S) (each a sigmoid of distance-to-boundary), aggregated as (s^{need}*{int}(t)=\max(\cdot)). This is “homeostatic pressure” toward rest/repair.
* **External threat** (s^{th}*{ext}(t)=f*{ext}(\text{danger signals})\in[0,1]). This is “safety pressure” that can override learning/editing and/or interrupt rest.

This separation matters downstream: “I’m depleted” and “the environment is dangerous” are not allowed to collapse into one scalar. 

### Assumptions baked in

* You can compute meaningful, bounded proxies for **energy** (E(t)), **damage** (D(t)), **semantic drift** (\text{drift}*P(t)), **opportunity** (\text{opp}(t)), and **predictive error** (E*{pred}(t)). If any of these are ill-defined, A0’s gates become arbitrary. 
* Arousal being driven by *magnitudes and changes* in normalized margins is assumed to be a good universal “urgency” heuristic (domain-general but not guaranteed).
* External threat is assumed to be available as a scalar in ([0,1]) (even if crude), and sufficiently reliable to justify hard overrides downstream.

### Implications

* **Control is intrinsically stateful and modeful.** Even if predictive loss is stable, changes in margins or threat can flip temperature, horizon, learning permissions, or macrostate later.
* The presence of (|\Delta\tilde m|) means **shock sensitivity** exists: abrupt regime change elevates arousal even if absolute margins are fine.
* The architecture forces a principled difference between:

  * “rest because I’m internally near limits” (need), and
  * “freeze/interrupt because the world is unsafe” (threat).

### Novelty (what is actually distinctive)

Most ingredients exist elsewhere (homeostasis, arousal filtering). The more distinctive move is **making compute slack and semantic integrity first-class physiological margins** that feed arousal and later gating, rather than being secondary engineering metrics. 

### Biological analog

* (s^{ar}(t)): fast neuromodulatory mobilization with slower clearance (a “hangover” dynamic).
* (s^{need}_{int}(t)): homeostatic drives (sleep pressure, metabolic depletion, repair).
* (s^{th}_{ext}(t)): threat circuitry that can suppress exploration/plasticity.
* Margins: energy reserve, accumulated wear/damage, and cognitive interference limits; compute slack resembles attentional/working-memory bandwidth treated as a hard resource.

---

## A1 — Viability

### What it means / what it does

A1 defines a **viability set** as linear constraints:
[
v(t)\in\mathcal{V}\iff A v(t)\ge b
]
This is a constitutional statement: there exists a region of internal margin states that counts as “viable,” and being outside it is not merely suboptimal—it is inadmissible in principle. 

### Assumptions baked in

* The “safe/viable internal envelope” is reasonably approximated as a **polytope** (intersection of half-spaces). Reality is usually nonlinear; you are explicitly choosing linearity for auditability and tractability.
* The margin vector (v(t)) is a sufficient summary for viability. That is an identifiability assumption: if crucial failure modes are not captured by (m_E,m_D,m_L,m_C,m_S), the constraint set is incomplete.

### Implications

* This steers the whole architecture toward **constraint satisfaction and recovery** rather than pure objective maximization.
* It supports **formal auditing**: you can test whether a policy or edit rule can force (v(t)) out of (\mathcal{V}).
* It implicitly prioritizes “never make recovery impossible” behaviors, because leaving (\mathcal{V}) is defined as losing viability.

### Novelty

Viability constraints are established in safe control / viability theory. The distinctive part is putting it as an early axiom that the rest of the architecture must respect, rather than an optional penalty term. 

### Biological analog

Organisms operate within viability regions (temperature, glucose, hydration, blood gases). The polytope form is an engineering simplification of “bounded homeostatic ranges.”

---

## A2 — Margins from Observables

### What it means / what it does

A2 pins the previously-declared margins to **measurable quantities**, turning A0 and A1 from philosophy into something implementable and testable. 

1. **Hard margins (energy and damage)**
   [
   \text{rawE}(t)=E(t)-E_{\min},\quad m_E(t)=\frac{\text{rawE}(t)}{\sigma_E}
   ]
   [
   \text{rawD}(t)=D_{\max}-D(t),\quad m_D(t)=\frac{\text{rawD}(t)}{\sigma_D}
   ]
   This encodes “distance to boundary” with normalization constants (\sigma_E,\sigma_D) to put margins on comparable scales.

2. **Semantic headroom**
   [
   \text{rawS}(t)=S_{\max}^{stab}-\text{drift}_P(t),\quad x_S(t)=\text{rawS}(t)
   ]
   Semantic integrity is treated as a bounded resource: more drift means less headroom.

3. **Learning opportunity**
   [
   x_L(t)=\text{opp}(t)
   ]
   A2 does not define (\text{opp}(t)); it declares that the architecture expects an opportunity signal and promotes it to margin status.

4. **Compute slack (explicit budget accounting)**
   [
   x_C(t)=B_{rt}-\Big(b_{enc}(t)+(1-\text{rest}(t))\cdot h(t)\cdot b_{roll}(t)+\text{rest}(t)\cdot b_{cons}(t)\Big)
   ]
   This is a major architectural commitment: compute is treated like metabolism with a fixed per-step budget (B_{rt}), and slack is what remains after paying encoding plus either rollout (operating) or consolidation (rest). 

### Assumptions baked in

* You can measure or define (E(t)), (D(t)), (\text{drift}_P(t)), (\text{opp}(t)) in a way that is stable enough to gate behavior.
* You can estimate costs (b_{enc}, b_{roll}, b_{cons}) *in the same units* as (B_{rt}). If these units drift or are miscalibrated, “compute slack” becomes noisy and can destabilize downstream control.
* Normalizers (\sigma_E,\sigma_D) are meaningful across time (or else need adaptive calibration elsewhere).

### Implications

* **Compute becomes non-negotiable.** The agent cannot “cheat” by thinking more without paying for it; horizon and edit permissions must couple to slack.
* **Semantic drift is operationalized as depletion.** You are treating interference/corruption as a resource consumption process, not merely a performance metric.
* You gain **explainability**: when the agent rests, freezes learning, or shortens horizon later, you can attribute it to specific observed margin shortfalls.

### Novelty

The core novelty is not “normalization” per se; it is elevating **compute slack** and **semantic integrity headroom** to the same level as energy/damage, and explicitly threading them into the viability/physiology story from the beginning. This is uncommon in many agent formalisms that assume compute is effectively free or handled implicitly. 

### Biological analog

* (E(t)): metabolic reserve/fatigue.
* (D(t)): accumulated wear/stress injury.
* (\text{drift}_P(t)): maladaptive plasticity / interference / cognitive disorganization proxy.
* (x_C(t)): attentional bandwidth / cortical processing capacity—implemented as a strict budget rather than an emergent limitation.

---

## A3 — Baselines and Normalization

### What it means / what it does

**A3 is the architecture’s “sensor calibration” layer for margins.** It defines (i) when the agent is allowed to update its internal baselines, and (ii) how to normalize margin signals so they are comparable, regime-robust, and useful for gating. 

* **A3.1 Stability predicate**
  (\text{stable}(t)) is a boolean gate used downstream to decide whether the system is in a condition where it is safe to update baselines. It explicitly references: (a) “no structural edits in a recent window,” and (b) variance/volatility constraints over a window (W) (your formula is abbreviated, but it clearly includes a (\text{Var}_W(\cdot)) term and thresholds). This is intended to prevent “moving the ruler while measuring.” 

* **A3.2 Baseline updates only when stable**
  (\mu_k(t)) is updated by an EMA:
  [
  \mu_k(t)=(1-\beta)\mu_k(t-1)+\beta x_k(t),
  ]
  but the section title (“when stable”) makes the operative assumption: **baseline adaptation is gated**. When unstable, baselines should remain fixed (or update by a safer rule elsewhere) so later normalized deviations remain meaningful. 

* **A3.3 Weber normalization for learning opportunity (L)**
  [
  \tilde m_L(t)=\frac{x_L(t)-\mu_L(t-1)}{|\mu_L(t-1)|+\varepsilon}.
  ]
  This makes “opportunity change” scale-free: a +0.1 change matters more when baseline opportunity is 0.2 than when it is 10. This is explicitly a *relative* (Weber-like) code for (L). 

* **A3.4 Multi-scale variance for compute slack (C) and semantic headroom (S)**
  You maintain fast and slow variance estimates ((\sigma^{fast})^2) and ((\sigma^{slow})^2), take their ratio (\rho=\sigma^{fast}/(\sigma^{slow}+\varepsilon)), then define an effective scale (\sigma^{eff}) and normalize:
  [
  \tilde m_k(t)=\frac{x_k(t)-\mu_k(t-1)}{\sigma_k^{eff}(t-1)}.
  ]
  The intention is: **separate “recent volatility” from “typical volatility”** and avoid overreacting to transient shocks or underreacting to regime shifts. 

* **A3.5 Freeze escape**
  If the system stays frozen beyond (T_{\max}), it forces a recovery protocol: higher (\beta) (adapt faster), safe policy, no edits until stability returns. This is a deadlock breaker: “stability gating” must not permanently stall learning/control. 

### Assumptions baked in

* **Baselines exist and are meaningful:** the agent can maintain (\mu_k) and variance estimates without pathological initialization or scale drift.
* **Stability is observable:** (\text{edits}_{struct}[t-W,t]) and (\text{Var}_W(\cdot)) are available and correctly computed under partial observation/foveation.
* **Opportunity (x_L(t)) has a sensible baseline:** Weber normalization becomes unstable if (\mu_L\approx 0) for long stretches (you mitigate with (\varepsilon), but the semantics then depend heavily on that constant). 

### Implications

* **Normalization is policy-relevant, not cosmetic.** A0’s arousal and later selection rules depend on these normalized deviations; A3 therefore directly shapes when the agent becomes “urgent,” “calm,” or “eligible to consolidate.”
* **You get regime-robust gating.** Fast/slow variance and the (\rho) ratio are a standard way to detect “volatility spikes” versus “volatility floor,” which helps prevent oscillatory behavior when compute slack or semantic drift becomes noisy. 

### Structural risks to flag immediately

* **Foveation–stability entanglement (potentially serious):** (\text{Var}_W(x)) and any “no edits recently” condition can become *functions of what you chose to look at*. If stability is computed on observed dims only, stability can be gamed by attention; if computed on unobserved dims, it is undefined. You need an explicit rule: “stable over which signals (margins only? observed blocks only? lagged residual summaries only?).” 
* **Hysteresis loops:** if “stable” is required to update baselines, and baselines are required to evaluate “stable” (via normalized terms), you can create circular dependence unless you define clear lagging (you do use (t-1) in places, which helps, but the document’s abbreviated stability predicate makes this worth auditing). 

### Novelty

The individual ingredients (EMA baselines, Weber normalization, dual-timescale variance) are standard. The distinguishing aspect in your framework is **treating these as constitutional machinery for viability/compute/semantic integrity**, not as ad-hoc preprocessing—i.e., they are upstream of macrostate, horizon, and edit permission. 

### Biological analog

* “Stable → update baseline” resembles **homeostatic set-point updating** that happens preferentially during quiescent periods rather than during acute stress.
* Weber normalization mirrors **relative coding** seen across sensory and motivational systems.
* Fast/slow variance resembles **phasic vs tonic** neuromodulatory tracking of volatility/uncertainty.

---

## A4 — Memory Substrate

### What it means / what it does

**A4 defines what memory “is” in the system and how it is retrieved.** Memory is not a raw buffer; it is a library of masked, predictive, resource-costed modules organized as a DAG, with explicit constraints on what can be active and how archived content is recalled. 

* **A4.1 Nodes are masked predictive abstractions**
  Each node (j) has:

  * mask (m_j\in{0,1}^D) (what dimensions it covers),
  * linear dynamics (W_j,b_j),
  * per-dimension variance (\Sigma_j),
  * reliability (\pi_j),
  * compute cost (L_j).
    This is a typed object: “what I predict,” “how uncertain I am,” “how trustworthy I am,” “what it costs to run.” 

* **A4.2 Bounded working set**
  The active set (\mathcal{A}*t) is bounded by:
  [
  |\mathcal{A}*t|\le N*{\max},\qquad \sum*{j\in\mathcal{A}*t}L_j\le L^{work}*{\max}.
  ]
  This is the hard resource constraint that makes “attention” and “memory recall” economically meaningful. 

* **A4.3 Cold storage with block-keyed retrieval**
  Inactive nodes are archived. Retrieval is **explicitly keyed by the same block structure used for foveation/coverage (A16)**—the document states the intent: recall is driven by global per-block residual/coverage state, not incidental overlap with the current active set.
  Concretely: you define a block score (S(b,t)) (residual + coverage/age term), select a fovea block set (\mathcal{F}_t), then retrieve archived incumbents associated with those block keys:
  [
  \mathcal{C}^{ret}*t := \bigcup*{b\in\mathcal{F}_t}\big(\mathcal{I}*b \cap (V\setminus\mathcal{A}*{t-1})\big).
  ]
  This makes retrieval a **keyed, coverage-driven operator** rather than nearest-neighbor similarity over arbitrary embeddings. 

* **A4.4 Anti-aliasing insertion discipline (disambiguation-by-storage)**
  You explicitly forbid storing two incumbents that are indistinguishable under the system’s own footprint/cue basis. You define a footprint key (\phi(j)=\text{block-id}(m_j)), then a distinguishability metric for two experts with the same footprint:
  [
  \Delta_\phi(p,q)=\mathbb{E}*{t\in\mathcal{T}*\phi}\left[|\mu_p(t+1|t)-\mu_q(t+1|t)|*{1,\phi}\right].
  ]
  The rule is: if a proposed (j*{new}) aliases an incumbent (i) (i.e., (\Delta_\phi(i,j_{new})) is below a threshold implied by your prose), then the system must either **REPLACE** (if strictly better on the same evidence) or **REJECT**—but it may not keep both. This is “pattern separation enforced at write-time.” 

### Assumptions baked in

* Masks align cleanly with block partitions (or at least can be assigned a footprint (\phi)).
* “Reliability” (\pi_j) is well-defined and updated in a way that is not circular with selection (selected nodes get more evidence; that can inflate (\pi) unless corrected).
* The evidence set (\mathcal{T}_\phi) for distinguishability is representative; otherwise anti-aliasing can mistakenly merge distinct dynamics that just haven’t been teased apart yet. 

### Implications

* **Memory is compositional and audited.** Because everything is keyed to blocks and footprints, you can reason about what the library “covers,” where residual debt is accumulating, and why a retrieval happened.
* **You prevent “library bloat by near-duplicates.”** Anti-aliasing forces competition among candidates that explain the same footprint; you do not get endless slightly-different variants living side-by-side.
* **Retrieval is functionally coupled to perception.** The fovea block choice (\mathcal{F}_t) is both an observation decision and a recall key, which creates a tight loop between “what you look at” and “what you remember.” 

### Structural risks to flag immediately

* **Alias test identifiability:** (\Delta_\phi) is defined over (\mu_p(t+1|t)) predictions and a footprint-restricted norm. If your predictive means are underdetermined (e.g., sparse observations, confounding regimes), the system can incorrectly declare “indistinguishable” and permanently delete diversity. This is not cosmetic; it can be a one-way loss of capability unless you have an explicit recovery mechanism. 
* **Rich-get-richer pressure:** if incumbents are selected more, they may accumulate reliability and more (\mathcal{T}_\phi) coverage, making replacement harder unless “strictly better on the same evidence” is very carefully specified.

### Novelty

DAG libraries, masked models, and working-memory caps exist elsewhere. The distinctive combination here is:

1. **retrieval keyed to coverage debt/residual at the block level**, and
2. **disambiguation enforced at insertion** (write-time), rather than tolerating duplicates and hoping retrieval-time scoring sorts it out. 

### Biological analog

* Working set (\mathcal{A}_t): **attention/working memory** under metabolic limits.
* Cold storage: **long-term memory** that is not continuously active.
* Block-keyed retrieval: **cue-addressed recall** (pattern completion triggered by a sparse key).
* Anti-aliasing: **pattern separation** (don’t store two traces that the system itself cannot reliably distinguish later).

---

## A5 — Salience (Arousal-Sharpened)

### What it means / what it does

**A5 defines attention as an economically constrained selection process whose “temperature” is modulated by arousal, need, and threat.** It turns the A4 library into an active working set each step. 

* **A5.1 Salience score (u_j(t))**
  [
  u_j(t)=\alpha_\pi\bar\pi_j(t)+\alpha_{deg}\frac{\deg^+*j}{\deg^+*{\max}}+\alpha_{ctx}\cdot \text{relevance}(x(t),j).
  ]
  Interpretation:

  * (\bar\pi_j): prefer **reliable** nodes,
  * (\deg^+): prefer **structurally useful hubs** (high outgoing degree in the DAG),
  * relevance: prefer **context-matched** nodes. 

* **A5.2 Temperature (\tau^{eff}(t)): need-sharpened, safe-play opened**
  You define a lagged “safe play/approach” factor:
  [
  s^{play}(t-1)=s^{ar}(t-1)\bigl(1-s^{need}*{int}(t-1)\bigr)\bigl(1-s^{th}*{ext}(t-1)\bigr),
  ]
  and then
  [
  \tau^{eff}(t)=\frac{\tau_a}{1+\beta_{sharp}s^{need}*{int}(t-1)}\left(1+\beta*{open}s^{play}(t-1)\right),
  \quad \beta_{open}\ge 0.
  ]
  Operationally:

  * **Internal need high → sharpen** (lower temperature; more deterministic focus),
  * **Arousal high but safe → open** (higher temperature; broader exploration/activation). 

* **A5.3 Salience gate (a_j(t))**
  [
  a_j(t)=\sigma!\left(\frac{u_j(t-1)-\theta_a}{\tau^{eff}(t)}\right),
  ]
  turning a score into an activation propensity. Lagging (u_j) by one step preserves timing discipline (you consistently lag modulators). 

* **A5.4 Active set selection (\mathcal{A}_t)**
  [
  \mathcal{A}*t=\text{GreedySelect}\left({j},\ \frac{a_j(t)\cdot \pi_j(t)}{L_j},\ \sum L_j\le L^{work}*{\max}\right),
  ]
  i.e., choose the subset that gives the best “activation × reliability per unit cost” under the working budget. This is a literal **utility-per-compute** attention rule. 

* **A5.5 Effective complexity**
  [
  L^{eff}(t)=\sum_{j\in\mathcal{A}_t} a_j(t),L_j,
  ]
  which is the cost-weighted “how much cognition is actually engaged,” not just how many nodes are active. 

* **A5.6 Anchor guardrail (force-inclusion semantics)**
  Anchors are always included first, must fit in budget, then GreedySelect fills remaining capacity from non-anchors. This prevents the selection mechanism from starving core reflex/always-on circuits. 

### Assumptions baked in

* The DAG degree (\deg^+) is meaningful as “usefulness” (this is a strong structural prior; it can also encode mere age/popularity).
* (\text{relevance}(x(t),j)) is well-defined under partial observation (and does not silently reintroduce global state access).
* The cost (L_j) is stable enough that “value per cost” is not noisy.

### Implications

* **Attention becomes a controlled thermodynamic knob.** Need forces exploitation; safe arousal permits exploration—explicitly, in one scalar (\tau^{eff}).
* **Compute budgeting is enforced at the selection layer.** Even if many nodes are “relevant,” only the economically best set runs.
* **There is a built-in bias toward hubs.** Degree-based scoring creates a preferential attachment dynamic unless counterbalanced; this can be desirable (reusable primitives become hubs) or pathological (old hubs crowd out new specialists). 

### Structural risks to flag immediately

* **Self-reinforcing salience loop:** nodes selected more often will (a) get more evidence, (b) potentially increase (\pi_j), (c) increase degree via new edges, and thus (d) win future selections. Without explicit anti-entrenchment mechanisms, you can lock into early incumbents. This is not necessarily wrong, but it is a real dynamical commitment. 

### Novelty

The novelty is not “softmax temperature” as such; it is:

* temperature shaped by **separated need vs threat vs arousal** (A0’s split), and
* active-set selection formalized as **(activation × reliability) / compute-cost** under a hard budget, with anchors force-included. 

### Biological analog

* (\tau^{eff}) captures the common observation: **deprivation narrows** behavior (goal-fixation), while **safe arousal broadens** exploration/play.
* “Value per cost” selection resembles metabolic/attentional allocation: circuits compete for limited processing resources.
* Anchor guardrails resemble **always-on autonomic/reflex pathways** that remain engaged even when higher-level cognition reallocates capacity. 

---

## A6 — Fixed Budget, Emergent Horizon

### What it means / what it does

A6 makes **time/compute** an explicit conserved resource and turns “planning horizon” into a **derived quantity** (emergent), not a fixed hyperparameter.

* **A6.1 Budget:** you assert a fixed per-step real-time budget (B_{rt}>0). 

* **A6.2 Load decomposition:** you split the active-set effective load into:

  * anchor load (L^{eff}*{anc}(t)=\sum*{j\in\mathcal{A}*t\cap P*{anchor}} a_j(t)L_j)
  * rollout load (L^{eff}*{roll}(t)=\sum*{j\in\mathcal{A}*t\setminus P*{anchor}} a_j(t)L_j)
  * total (L^{eff}(t)=L^{eff}*{anc}(t)+L^{eff}*{roll}(t)). 

  Then you price computation as:

  * **encoding cost** paid once per step and explicitly including anchor overhead:
    [
    b_{enc}(t)=b_{enc,0}+b_{anc,0}(\varepsilon + L^{eff}_{anc}(t)).
    ]
  * **rollout cost** paid *per horizon step*, excluding anchors:
    [
    b_{roll}(t)=b_{roll,0}(\varepsilon + L^{eff}_{roll}(t)).
    ]
  * **consolidation cost** (b_{cons}(t)) paid only in REST, defined as the cost of structural edits processed at (t). 

* **A6.3 Horizon:** (h(t)=0) in REST; otherwise it is computed by a floor of “remaining budget after encoding” divided by “per-step rollout cost,” with clamping (your document elides the exact clamp expression with `max...`). 

Operationally: the agent cannot “think arbitrarily far.” Its forward simulation depth is whatever the current budget and current active-set complexity can afford.

### Assumptions baked in

* **Cost models are calibrated and stationary enough** to be meaningful in the control loop: (b_{enc,0}, b_{anc,0}, b_{roll,0}), and the interpretation of (L_j). If these drift, horizon becomes noise.
* Anchors are assumed to be “always worth paying” and thus priced separately (Option B semantics).
* (b_{cons}(t)) is assumed measurable/estimable at the time decisions are made (or at least bounded), otherwise REST budgeting can’t be enforced.

### Implications

* **Horizon is a *state variable*** (via active set and costs), so any change in salience (A5) or library composition (A4) changes planning depth immediately.
* **Anchors become a guaranteed tax on cognition.** If anchor load grows, encoding cost rises even if rollout set is small, shrinking horizons globally.
* **REST is mechanically different**: horizon goes to zero and compute is reallocated to edits (consolidation).

### Structural issues to call out immediately

* **Underspecification in the document is material:** A6.2 and A6.3 contain `...` / truncated expressions (including the clamp for (h(t))). Without the explicit clamp and the “remaining budget” expression, two implementations can differ in ways that change the qualitative dynamics (e.g., whether horizon can become negative then clamped; whether there is a minimum horizon; whether budget debt carries). 
* **Potential circularity through budgeting:** (h(t)) depends on (b_{roll}(t)), which depends on (L^{eff}_{roll}(t)), which depends on the active set (\mathcal{A}_t), which is chosen using cost (L_j) and temperature (A5). That’s not automatically wrong, but it is a **tight algebraic loop**; you avoid a true instantaneous loop only if selection uses lagged signals or strictly earlier state (as you did elsewhere). The axiom text here doesn’t explicitly state the lagging convention.

### Novelty

Not novel in isolation (budgeted planning exists), but your distinctive combination is:

* **explicit anchor-vs-rollout cost accounting**, and
* **horizon as a hard-budget consequence** rather than a tuned constant. 

### Biological analog

* Fixed (B_{rt}): a metabolic/time constraint per unit time.
* Anchors: autonomic / always-on circuitry with unavoidable baseline cost.
* Emergent horizon: “how far ahead you can simulate” shrinking under fatigue/threat and expanding when resources are abundant.

---

## A7 — Completion and Fusion

### What it means / what it does

A7 defines a single operator that turns a set of masked predictors into a **global completed prediction** with a **global uncertainty**, then propagates that uncertainty forward.

* **A7.1 Per-node prediction:** each active node produces a mean prediction
  [
  \mu_j(t+1|t)=W_j x(t)+b_j
  ]
  and a per-dimension precision vector (\Lambda_j) that is nonzero only where the node’s mask covers the dimension:
  [
  \Lambda_j[k]=
  \begin{cases}
  1/\Sigma_j[k] & m_j[k]=1\
  0 & m_j[k]=0
  \end{cases}
  ] 

* **A7.2 Coverage invariant:** every dimension must have some positive total precision across the active set:
  [
  \forall k:\ \sum_{j\in\mathcal{A}*t}\Lambda_j[k]>0.
  ]
  If violated for dimension (k), you define a fallback: (\hat x[k]=x(t)[k]) (“persist”) and set global variance (\Sigma*{global}[k]=\infty). 

* **A7.3 Precision-weighted fusion:** the global covariance is the inverse of summed precisions:
  [
  \Sigma_{global}(t+1|t)=\left(\sum_{j\in\mathcal{A}*t}\Lambda_j\right)^{-1}.
  ]
  You then propagate uncertainty forward by adding process noise:
  [
  \Sigma*{global}(t+\ell+1|t)=\Sigma_{global}(t+\ell|t)+\Sigma_{proc},
  \quad \Sigma_{proc}[k]=\eta_{proc}\cdot \Sigma_{global}(t+1|t)[k].
  ] 

* **Confidence:** you compress uncertainty into a scalar horizon-indexed confidence:
  [
  H_k(t)=\text{mean}(\Sigma_{global}(t+k|t)),\quad
  c_k(t)=\sigma!\left(-\frac{H_k(t)-\mu_H}{\sigma_H}\right).
  ] 

Net effect: A7 is the architecture’s “Kalman-like” completion step—masked sources contribute where they have coverage, and uncertainty is explicit and propagated.

### Assumptions baked in

* **Gaussian / conditional independence approximation:** precision-weighted fusion is only justified if the contributing predictions can be treated as independent sources of information per dimension (or at least if double-counting correlations is acceptable).
* **(\Sigma_j[k]) is meaningful as calibrated predictive uncertainty**, not just a heuristic score.
* The fallback “persist (x(t)[k])” assumes (x(t)) has a well-defined value even for not-currently-observed dims (under foveation). That implies you maintain a latent/full state estimate even when sensory access is partial.

### Implications

* **Completion and prediction are unified:** the same fusion machinery gives you (i) a filled-in estimate now and (ii) a predictive rollout distribution.
* **Coverage becomes a hard contract:** if the active set cannot cover a dimension, you refuse to hallucinate certainty (infinite variance) and you “carry forward” the last available value.
* **Confidence becomes an explicit gating scalar** that downstream axioms can use for commitment/action gating (A8).

### Structural issues to call out immediately

* **Coverage invariant vs foveation is a potential fault line.** Under A16-style partial observation, it is easy for the active set to lack coverage for many dims. Your fallback sets (\Sigma_{global}[k]=\infty), which is coherent, but “persist (x(t)[k])” is only coherent if (x(t)[k]) is your internal belief state, not “the current sensory reading.” The axiom text does not explicitly disambiguate that representation, and implementations can diverge substantially here. 
* **Fusion formula is partially elided (`...`).** The global mean fusion equation is not shown in the excerpt, but it is load-bearing; different choices (sum of precisions times means vs other) change behavior. 
* **Correlation/double-count risk:** if two nodes overlap on the same dims and are not independent, precision addition overstates certainty (a classic failure mode).

### Novelty

The mechanism is standard probabilistic filtering in form. What is more characteristic of your framework is:

* using **mask-defined precision vectors** as the interface between heterogeneous library nodes, and
* making “coverage debt” explicit via the invariant + infinite-variance fallback. 

### Biological analog

* Multi-cue integration weighted by reliability (precision) maps cleanly to sensory fusion principles.
* “Persist with uncertainty” resembles maintaining a working belief state for unobserved features (object permanence / predictive maintenance) while acknowledging low confidence.

---

## A8 — Latency and Commitment

### What it means / what it does

A8 prevents the agent from acting on plans it cannot compute in time, and forces a safe fallback when it can’t meet latency + confidence requirements.

* **A8.1 Latency floor:** you define a minimum required horizon depth (d) from processing time:
  [
  d=\left\lceil \frac{T_{proc}}{\Delta t}\right\rceil.
  ]
  Interpretation: if it takes (T_{proc}) seconds to generate/refresh the plan and the environment steps every (\Delta t), you need at least (d) steps of lookahead to justify committing to the planned action. 

* **A8.2 Commitment gate:** commit is the product of indicator constraints:

  * not in REST,
  * horizon (h(t)\ge d),
  * confidence at depth (d), (c_d(t-1)), exceeds (\theta_{act}).
    (The formula is visibly truncated with `mat...` but the intended multiplicative gate is clear.) 

* **A8.3 Action:** if commit=1, execute planned action; else execute (\pi_{safe}) (safe/reflex policy). 

### Assumptions baked in

* (T_{proc}) and (\Delta t) are measurable and stable enough that (d) is meaningful (or at least bounded).
* (c_d(t)) is a calibrated proxy for “plan reliability at the required depth,” not merely “low variance under a potentially overconfident fusion.”
* (\pi_{safe}) is defined and guaranteed to be viable relative to A1’s viability constraints (otherwise the fallback is not actually safe).

### Implications

* **Hard separation between deliberative and reflex control.** The system does not “sort of” execute a plan; it either commits or it reflexes.
* **Compute scarcity directly reduces agency.** If A6 yields small horizons (low slack, heavy load), the agent will frequently fall back to (\pi_{safe}).
* **Confidence is now safety-critical.** Any miscalibration in A7 confidence will translate into either premature commitment (unsafe) or chronic non-commitment (frozen behavior).

### Structural issues to call out immediately

* **The commitment expression is truncated in the axiom text.** Because commit is safety/behavior defining, the exact gate matters (e.g., whether threat also gates; whether arousal gates; whether there is hysteresis). The document’s `mat...` truncation is a real ambiguity. 
* **Deadlock mode:** if horizons are persistently (<d) (e.g., heavy anchor cost or low budget), the agent becomes permanently reflexive. That may be acceptable by design, but it should be explicitly recognized as a stable regime of the system.

### Novelty

Latency-aware commitment gating exists in control/robotics. Your framework-specific angle is that:

* the gate couples **(budget-derived horizon)** with **(fusion-derived confidence)** and forces a **safe policy** otherwise, making it an explicit constitutional switch rather than an implementation convenience. 

### Biological analog

* Commit gate resembles “don’t execute complex action without sufficient prepared motor plan + confidence,” otherwise default to reflexive/defensive behavior.
* Latency floor (d) maps to neural planning latency vs environmental tempo: fast environments force simpler, more habitual control unless prediction is both quick and reliable.

---

## A9 — Prediction Objective

### What it means / what it does

A9 defines **what “good prediction” is** and how the agent allocates predictive effort across its available horizon (h(t)) (from A6) while respecting the latency floor (d) (from A8).

* **A9.1 Loss:** for each rollout depth (k), you score prediction quality by a negative log-likelihood style objective:
  [
  \mathcal{E}_k(t)=\mathbb{E}\left[-\log p\big(x(t+k)\mid \hat x(t+k\mid t)\big)\right].
  ]
  This makes the completion/fusion output (\hat x(\cdot)) (A7) not just a point estimate but implicitly a predictive distribution (p(\cdot)).

* **A9.2 Objective (short vs long horizon weighting):** you always train *while operating* (never in REST) and blend:

  * a **mandatory** loss at the latency depth (d), and
  * an **optional** weighted sum of losses for deeper steps (k=d+1\ldots h(t)),

  with a mixing coefficient (\lambda(t)) that is a sigmoid of **lagged arousal**:
  [
  \lambda(t)=\sigma(w_\lambda s^{ar}(t-1)+b_\lambda).
  ]
  [
  J_{pred}(t)=(1-\text{rest}(t))\Big[(1-\lambda(t))\ell_\tau(\mathcal{E}*d(t))+\lambda(t)\sum*{k=d+1}^{h(t)} w(k),\ell_\tau(\mathcal{E}*k(t))\Big].
  ]
  So: the agent is always pressured to be competent at the minimum viable planning depth (d), and then conditionally pressured to improve deeper predictions depending on the arousal-weighted schedule. The robustifier (\ell*\tau(\cdot)) indicates you are explicitly guarding against outlier loss spikes dominating learning. 

* **A9.3 Curiosity (explore only when safe):** you add a second term that pushes the agent toward *high-error, low-confidence* deeper predictions, but only when external threat is below a safety threshold:
  [
  J_{explore}(t)=(1-\text{rest}(t))\cdot \gamma\sum_{k=d+1}^{h(t)}(1-c_k(t)),\mathcal{E}*k(t)\quad\text{if }s^{th}*{ext}(t-1)<\theta_{safe}.
  ]
  This is “curiosity as targeted uncertainty reduction” with an explicit “no curiosity under threat” doctrine. 

### Assumptions baked in

* A well-defined predictive density (p(x\mid \hat x)) exists and is consistent with the fusion outputs (A7). If you only have means/variances heuristically, (-\log p) can become arbitrary or miscalibrated.
* The sign/magnitude of (w_\lambda) is architecturally consequential but not fixed by the axiom: it determines whether higher arousal shifts weight **toward** deeper horizons or **away** from them (the written form makes (\lambda) monotone in arousal if (w_\lambda>0), but the axiom does not assert (w_\lambda>0)).
* Confidence (c_k(t)) is calibrated enough that ((1-c_k)\mathcal{E}_k) is meaningful; otherwise curiosity can chase noise.

### Implications

* **Depth-(d) competence is non-optional.** Regardless of what happens beyond (d), the architecture is always pressured to predict at the minimum latency-justified depth when operating.
* **Exploration is state-conditional.** Curiosity is explicitly suppressed under threat, aligning exploration with safety constraints rather than reward maximization at all times.
* **Arousal becomes a curriculum knob.** Through (\lambda(t)), stress state can alter how much the system cares about longer-horizon accuracy relative to “just be locally competent.”

### Novelty

Most components are known (multi-step prediction loss, uncertainty-weighted curiosity). What is distinctive here is making them **axiomatic and explicitly coupled** to:

* the **latency floor** (d),
* the **budget-derived** horizon (h(t)), and
* **threat-gated** exploration. 

### Biological analog

* Mandatory competence at a short horizon resembles sensorimotor control requiring near-term predictions to act.
* Threat-gated curiosity matches the empirical pattern: exploratory drive collapses under perceived danger.
* Arousal-modulated horizon-weighting mirrors state-dependent planning depth (calm deliberation vs urgency-biased processing), though the *direction* depends on how you set (w_\lambda).

---

## A10 — Edit Control

### What it means / what it does

A10 defines **when learning/editing is allowed** and prevents two failure modes: (i) adapting while the world is dangerous, and (ii) corrupting incumbents by updating them off-context.

* **A10.1 Freeze (external threat only):**
  [
  \text{freeze}(t)=\mathbf{1}{s^{th}_{ext}(t-1)\ge \chi^{th}}.
  ]
  Freeze is intentionally *not* driven by internal arousal/need; it is a hard “external safety” lock. 

* **A10.2 Permit parameter updates in OPERATING under headroom and slack:** parameter updates are allowed only if:

  * not REST,
  * not frozen,
  * compute slack (x_C(t-1)) above a threshold,
  * arousal below a “panic cap” (so arousal is not generally a calmness requirement),
  * energy/damage headroom is sufficient (via rawE/rawD thresholds).

  Formally:
  [
  \text{permit}*{param}(t)=\mathbf{1}{\text{rest}(t)=0}\cdot \mathbf{1}{\text{freeze}(t)=0}\cdot \mathbf{1}{x_C(t-1)>\tau_C^{edit}}\cdot \mathbf{1}{s^{ar}(t-1)<\theta*{ar}^{panic}}\cdot \mathbf{1}{\text{rawE}(t-1)>\tau_E^{edit}}\cdot \mathbf{1}{\text{rawD}(t-1)>\tau_D^{edit}}.
  ]
  This is the doctrine “learn while engaged if you have slack, unless you’re in panic or threatened.”

* **A10.3 Responsibility-gated parameter learning (prevent off-context corruption):** even if parameter updates are permitted, you only update an expert (j) if it is:

  * in the active set,
  * has any overlap with the *currently observed* dimensions (\mathcal{O}_t) (from foveation A16),
  * and its local prediction error on that footprint is below a learn-threshold.

  Definitions:
  [
  \text{obs}_j(t)=\mathbf{1}{\exists k\in\mathcal{O}_t:\ m_j[k]=1}
  ]
  [
  \text{err}*j(t)=\mathrm{mean}*{k\in\mathcal{O}_t}\big(m_j[k]\cdot |x(t)[k]-\hat x(t\mid t-1)[k]|\big)
  ]
  [
  \text{resp}_j(t)=\mathbf{1}{j\in\mathcal{A}_t}\cdot \text{obs}_j(t)\cdot \mathbf{1}{\text{err}*j(t)\le \theta*{learn}}.
  ]
  Then: only experts with (\text{resp}_j(t)=1) may get ((W_j,b_j)) updates; non-responsible experts may still have reliability (\pi_j) penalized under sustained error. The axiom explicitly states the intent: “punish it, don’t corrupt it.”

### Assumptions baked in

* External threat (s^{th}_{ext}) is available and trustworthy enough to justify hard learning freezes.
* “Local error on observed footprint” is a valid proxy for whether the expert is in a context where gradient updates will improve it rather than smear it across regimes; this assumes the observed footprint is diagnostic and not systematically confounded by foveation choices.
* The thresholds (\tau_C^{edit}, \theta_{ar}^{panic}, \tau_E^{edit}, \tau_D^{edit}, \theta_{learn}) can be set so that learning happens often enough to make progress but not so often that it destabilizes.

### Implications

* **Plasticity is safety-scoped.** External threat shuts down param learning regardless of “internal motivation,” which structurally prioritizes survival/viability over adaptation speed.
* **Learning is localized to evidence.** Under foveation, an expert cannot be updated unless its footprint is actually observed; this is a strong guardrail against spurious updates based on completed/hallucinated values.
* **Non-responsible experts degrade via reliability, not weights.** That creates a clean separation: weights change only when you have good local evidence; confidence/reliability can still fall when you are wrong. 

### Structural risks to flag immediately

* **Foveation can gate learning frequency.** If (\mathcal{O}_t) rarely covers a footprint, that expert will rarely be “responsible,” which can freeze its parameters while its reliability decays. That is a coherent design choice, but it can create “permanent under-trained islands” unless your foveation policy guarantees eventual coverage (your A16 explicitly does *not* guarantee bounded age).
* **Error-threshold learning gate can be conservative:** requiring (\text{err}*j\le\theta*{learn}) means you only update when you’re already “not too wrong,” which prevents catastrophic corruption but can slow recovery from drift unless complemented by REST structural changes (A12) and reliability dynamics.

### Novelty

The notable design commitment is not “freezing” or “gating updates” per se; it is the explicit axiom-level rule:

* **do not update an incumbent unless it is responsible for the currently observed footprint**, and otherwise treat failure as a reliability problem rather than a parameter rewrite problem.

### Biological analog

* External threat suppressing learning resembles stress-modulated plasticity (learning inhibited under acute danger).
* Responsibility-gated updates resemble “only synapses/circuits that were active and had valid local error signals undergo consolidation,” while others may be down-weighted (analogous to eligibility traces + neuromodulatory gating).

---

## A11 — Semantic Integrity

### What it means / what it does

A11 defines a **semantic drift signal** and a **semantic cost function for edits**, which later axioms (A12) use as a hard acceptance constraint.

* **Semantic drift:**
  [
  \text{drift}*P(t)=\frac{1}{|P|}\sum*{p\in P}\big|f_p(t)-f_p^{ref}\big|.
  ]
  This treats a set of probes/features (f_p(t)) as a semantic “reference manifold,” and drift is average absolute deviation from stored references. 

* **Semantic change under an edit (e):**
  [
  \Delta S(e)=g(\Delta \text{perf}_P,\ \text{interference}_P,\ \Delta \text{drift}_P).
  ]
  This is an explicit placeholder for the rule that converts “did probes improve?”, “did they interfere?”, and “did drift worsen?” into a scalar semantic penalty/impact used by acceptance logic. 

### Assumptions baked in

* The probe set (P) exists, is stable, and is representative of what you mean by “semantic integrity.” If probes are poorly chosen, drift becomes either meaningless or trivially gameable.
* Reference values (f_p^{ref}) are defined and maintained coherently (the axiom does not specify when/how references update, which is a key design choice).
* The function (g(\cdot)) is well-defined and consistent across edits; as written, it is underspecified.

### Implications

* **Edits are constrained by meaning preservation.** You are explicitly preventing “improve prediction loss by corrupting semantics” by introducing drift/interference as a first-class acceptance dimension (wired into A12’s acceptance constraints).
* **Semantic integrity becomes auditable.** You can log (\text{drift}_P(t)) and attribute failed edit proposals to semantic impact rather than opaque heuristics.

### Structural risk to flag immediately (this one is load-bearing)

A11 is **materially underspecified** in two places:

1. what exactly the probes (f_p(t)) are (and whether they are invariant under ordinary self-improvement), and
2. what (g(\cdot)) is (how interference and drift trade off against performance).

Because A12 later treats (\Delta S(e)) as a constraint, these underspecifications are not cosmetic—they determine whether semantic integrity is a real guarantee or a tunable knob.

### Novelty

“Catastrophic forgetting / interference” measures are common, but you are making **semantic drift a constitutional acceptance constraint** alongside compute and net value, rather than an auxiliary metric. That’s the main distinctive move at the axiom level. 

### Biological analog

* Drift and interference correspond to maintaining stable representations under ongoing plasticity (homeostatic plasticity / stability–plasticity balance).
* “Only accept changes that don’t break core semantics” parallels consolidation mechanisms that preserve high-level invariants while allowing local adaptation (often associated with sleep/rest phases in your macrostate design).

---

## A12 — Edit Acceptance

### What it means / what it does

A12 is the system’s **self-modification constitution**: it defines **when parameter learning is allowed**, **when structural edits are allowed**, and **how to accept/reject structural changes** (MERGE/PRUNE/SPAWN/SPLIT). 

At a high level, A12 enforces three separations:

1. **Parameter updates vs. structural edits**

   * **Parameter updates** can occur during **OPERATING** (subject to permission gates) and are evaluated by a net-value test plus safety constraints. 
   * **Structural edits** (changing the library graph: merge/prune/spawn/split) are restricted to **REST only**, and are further gated by a “REST-calm” arousal condition. 

2. **Value improvement vs. complexity control**

   * A12.1 defines **net value** as:
     [
     \Delta J(e)=\Delta F(e)-\beta\cdot \Delta L^{MDL}(e)
     ]
     which explicitly trades improved fit/performance against **description-length/complexity growth**. 

3. **Compression/consolidation correctness**

   * A12.3’s MERGE test is **replacement-consistent**: it requires that the merged model (C) does **not worsen** performance **on the subset of timesteps where each parent (A or B) “owned” the context** (partitioned by pre-merge activations). 

### Immediate structural hazards / underspecification you must treat as implementation-critical

These are not “minor details”; if they drift, A12 becomes non-auditable or self-contradictory:

* **A12.1 depends on quantities not fully defined here**: (\Delta F(e)), (\Delta L^{MDL}(e)), (\Delta C(e)), (\Delta S(e)), and the thresholds (q_{\min}(t),\varepsilon(t), S_{\max}^{sem}(t), C_{\max}(t)). Without canonical definitions and measurement windows, two implementations can “both satisfy A12” while accepting opposite edits. 
* **MERGE’s replacement-consistent evaluation set requires precise sampling rules** (how (\mathcal{T}_{AB}) is formed, how “domain” is defined, how observation constraints (\mathcal{O}_t) interact). This is a common silent-failure point: merges can look “safe” if the evidence window is biased toward easy cases. 
* **A12.3 assumes DoF-block alignment and shared footprints for MERGE** ((\phi(A)=\phi(B)=\phi(C))), which is a strong identifiability constraint: you are forbidding merges that span blocks even if the world’s structure truly couples them. That constraint is coherent only if cross-block structure is meant to be captured **by composition elsewhere**, not by single experts. 

### Key mechanisms (in plain terms)

**A12.2 — Parameter update acceptance (OPERATING)**
Parameter updates are accepted only if:

* updates are permitted ((\text{permit}_{param}(t))), and
* quality and net value constraints pass ((q_e\ge q_{\min}(t)), (\Delta J(e)\ge \varepsilon(t))), and
* semantic and compute side-constraints pass ((\Delta S(e)\le S_{\max}^{sem}(t)), (\Delta C(e)\le C_{\max}(t))). 

Interpretation: parameter learning is “online,” but it is **never allowed to quietly trade away semantic integrity or compute budget** to get short-term fit.

**A12.3 — Structural edits (REST only)**
Structural edits require:
[
\text{permit}*{struct}(t)=\mathbf{1}{\text{rest}(t)=1}\cdot \mathbf{1}{s^{ar}(t-1)<\theta*{ar}^{rest}}
]
So: REST *and* arousal below a “rest calm” threshold. 

MERGE(A,B→C) requires:

* high activation correlation,
* cheaper resulting structure ((L_C<L_A+L_B)),
* and the **replacement-consistent error test**: on A-dominant timesteps, C is not worse than A; on B-dominant timesteps, C is not worse than B (up to (\varepsilon_{merge})). 

PRUNE(j) happens when reliability is too low or inactivity too long. 

**A12.4 — Structural growth from persistent residual (REST only)**
SPAWN (and SPLIT) are triggered by **persistent per-footprint residual**:
[
R_\phi(t)=(1-\beta_R)R_\phi(t-1)+\beta_R\cdot \text{residual_block}(\phi,t)
]
If (R_\phi) stays above (\theta_{spawn}) over distinct coverage visits and incumbents can’t reduce it, propose a new expert aligned to (\phi), then accept only if:

* structural permission holds,
* net-value and semantic/compute constraints hold,
* and **anti-aliasing** forces replace/reject rather than duplicate storage. 

### Assumptions

* You can compute **error** on a constrained observation set and attribute it to a footprint (depends on A16 and the “residual = prediction error on observed dims” convention). 
* You can define “domains” for the “per-domain non-worsening” merge test in a way that is stable under changing active sets.
* MDL/complexity change (\Delta L^{MDL}) is measurable and aligned with real compute/memory costs.

### Implications

* **Catastrophic merge prevention**: replacement-consistent MERGE is explicitly designed to stop the classic “averaging two specialists into one generalist that is worse on each specialist’s regime.”
* **Offline consolidation**: structural re-writes only occur in REST, which prevents the system from destabilizing its own predictive substrate while it is still acting.
* **Residual-driven vocabulary expansion**: SPAWN/SPLIT means the representational library grows *only where persistent unexplained structure remains*, rather than via constant expansion.
* **Compute realism**: because acceptance is constrained by (\Delta C) and MDL, “getting smarter” must pay for itself.

### Novelty claim (what’s actually distinctive)

Nothing here is “unknown” in the literature (MDL, offline consolidation, pruning, residual-triggered expansion exist), but your distinctives are:

* making **replacement-consistent merge safety** an explicit axiom (not a heuristic),
* tying structural growth to **coverage-disciplined** persistent residual (via the foveation/residual accounting),
* and requiring **REST-only structural edits with arousal gating** as constitutional behavior. 

### Biological analog

* **Sleep-dependent consolidation**: “REST-only structural edits” mirrors offline replay/consolidation, synaptic renormalization, and structural remodeling during sleep-like states.
* **Synaptic pruning / competitive replacement**: PRUNE and anti-alias replace/reject are analogous to pruning weak traces and strengthening dominant representations.
* **Prediction-error–driven learning signals**: SPAWN from persistent residual resembles “new cell assemblies / new features recruited” when prediction error persists across exposures.

---

## A13 — Unified Completion

### What it means / what it does

A13 asserts a single principle: **perception, recall, and prediction are the same operator (A7 completion/fusion) applied with different cues**:

* Perception: dense external cue
* Prediction: current state cue, rolled forward
* Recall: sparse internal cue, variable horizon (k) 

In effect: there is no separate “retrieval engine” vs “inference engine.” **Recall is just inference under partial observation**, and perception is “overwrite observed dims then complete the rest.”

### Implementation-critical underspecification

A13 is intentionally short, but that brevity hides decisions that will materially change behavior unless pinned down elsewhere:

* What constitutes a **recall cue** (which dimensions are provided, how chosen)?
* What does “variable (k)” mean operationally for recall (search over time index? rollout length? best-match depth?)?
* How does this interact with “no long-term pixel storage” and the encoder boundary (i.e., completion must operate over (x(t)) abstractions, not raw pixels)? 

### Assumptions

* There exists a shared latent/state space (x(t)) where both sensory cues and internal cues can be expressed.
* A7’s fusion/completion is well-defined even when observation masks are sparse (coverage invariant and anchors matter here).

### Implications

* **Architectural economy**: one machinery (predictive completion) supports three cognitive functions.
* **Consistency pressure**: because recall and perception share the completion rule, “what you remember” and “what you predict” cannot drift into separate, incoherent systems without causing residual signals.
* **Strong bias toward predictive coding**: the system defaults to “fill in what you didn’t see with what your model predicts,” which is precisely the predictive-processing stance.

### Novelty claim

This is not novel as a concept (Kalman filtering, predictive coding, associative completion). The novelty is **normative**: you elevate unification to an axiom so implementations cannot quietly add a second memory mechanism that bypasses predictive accountability. 

### Biological analog

* **Predictive coding / analysis-by-synthesis**: cortex as a generative model filling in sensory gaps; perception as prediction constrained by evidence.
* **Pattern completion**: hippocampal/cortical completion from partial cues (recall as completion).
* **Mental imagery**: prediction-like completion when external cue is absent or minimal.

---

## A14 — Macrostates (REST vs OPERATING)

### What it means / what it does

A14 defines a **two-state control regime**:

* **OPERATING**: compute budget funds encoding + rollout horizon; parameter updates may occur; margins deplete.
* **REST**: horizon (h=0); structural edits are processed; margins recover. 

It is not just “sleep vs wake” narratively; it is the **switch that determines whether the agent spends budget on acting/predicting or on consolidating/editing structure**.

Key components:

1. **Queue and timers (A14.2)**

   * A structural proposal queue accumulates in OPERATING and is worked down in REST.
   * Timers track time since rest and time in rest. 

2. **Rest pressure accumulation/decay (A14.3)**

   * Rest pressure rises during OPERATING and decays during REST.
   * Pressure is increased by internal need signals. 

3. **Demand with hysteresis + completion-gated restoration (A14.4)**

   * Enter/exit thresholds differ ((\theta_{enter}>\theta_{exit}), (\Theta_Q^{on}>\Theta_Q^{off})) to prevent chattering.
   * Demand is driven by pressure, queue size, and timers; restoration uses margin thresholds. 

4. **Permission and interrupt are external-only (A14.5–A14.6)**

   * You can only rest if threat is sufficiently low.
   * Threat can also interrupt rest. 

5. **Lagged state transition (A14.7)**
   [
   \text{rest}(t)=\mathbf{1}{\text{rest_permitted}(t-1)}\cdot \mathbf{1}{\text{demand}(t-1)}\cdot \mathbf{1}{\neg\text{interrupt}(t-1)}
   ]
   This is a strict timing/casual-discipline rule: mode at (t) is computed from signals at (t-1). 

### Assumptions

* You can compute:

  * internal need and restoration predicates from margins,
  * a meaningful queue size for structural work,
  * threat estimates that are fast and conservative enough to gate REST.
* Hysteresis thresholds can be tuned so REST bouts occur without oscillation.

### Implications

* **No “edit while acting” structural churn**: A12’s structural changes are mechanically forced into REST by A14’s state definition plus A12.3’s permit_struct.
* **Safety-first sleep**: “permission and interrupt are external-only” means internal fatigue does not override immediate danger signals.
* **Compute allocation becomes behavioral**: because horizon is zero in REST and nonzero in OPERATING, the macrostate effectively allocates the fixed budget to either external competence (planning/rollout) or internal maintenance (consolidation).
* **Failure mode to watch**: if threat is persistently high, REST can be suppressed long enough to cause margin collapse. The architecture implicitly expects the “safe policy / freeze / escape” logic elsewhere to manage that; if not, A14 can deadlock the system into exhausted operation.

### Novelty claim

Two-state macrostates are not novel. What is more distinctive here is:

* the explicit **queue-driven demand** for REST (REST as a computational maintenance scheduler),
* the **threat-only permission/interrupt** semantics (clean separation of “need to rest” vs “allowed to rest”),
* and **lagged gating** as an axiom (reducing implementation degrees of freedom that often create subtle causality bugs). 

### Biological analog

* **Sleep pressure** (homeostatic Process S): pressure rises with time awake, decays in sleep.
* **Arousal/threat systems** preventing sleep: predators/danger suppress sleep; startling events interrupt sleep.
* **Offline consolidation windows**: sleep/rest periods used for memory reorganization and synaptic maintenance.

---

## A15 — Margin Dynamics

### What it means / what it does

A15 pins down **how the agent’s internal “margins” evolve over time as a function of workload and macrostate**:

* In **OPERATING**, margins **deplete**: energy goes down, damage goes up, and representational drift pressure increases, each as a sum of (i) base operating cost, (ii) commitment cost, and (iii) load proportional to effective complexity (L^{eff}(t)).
* In **REST**, margins **recover toward nominal** with small recovery rates (first-order relaxation): energy relaxes upward to (E_{max}); damage relaxes downward to (D_{min}); drift relaxes toward 0.
* A15.3 explicitly requires recovery rates to be **slow** ((\ll 1)) to force **multi-step rest bouts** rather than instantaneous reset.

Operationally: A15 is the “physiology clock.” It ensures there is **real cost** to staying in OPERATING and **real benefit** (but gradual) to REST, which makes A14’s macrostate mechanism non-degenerate.

### Core assumptions (what must already be well-defined)

A15 assumes the existence and measurability of:

* **commit(t)** (some scalar reflecting behavioral engagement/effort) and **effective complexity (L^{eff}(t))** (defined earlier via active-set weighted costs).
* **Energy bounds** (E_{min}, E_{max}), **damage bounds** (D_{min}) (and implicitly (D_{max}) elsewhere), and a drift quantity (\text{drift}_P(t)) that is meaningful to “relax toward 0” in REST.

### Immediate structural hazards / identifiability risks

If any of the following are left ambiguous, implementations will diverge in ways that can’t be audited:

* **What exactly is “commit(t)”**? If it’s derived from action magnitude, risk, arousal, prediction error, or something else, A15 changes from “metabolic cost” to “punishment for learning” depending on the choice. (A15 uses it directly in the cost terms.)
* **What is drift_P operationally?** A15 treats it as a scalar that monotonically accumulates in OPERATING and is repairable in REST; if drift is instead “interference” tied to specific weights/representations, a scalar relaxation may misrepresent the true dynamics.
* **Parameter scaling matters**: if (c_*^{load}) is too large relative to (L^{eff}) units, the agent will be forced into REST constantly; if too small, REST becomes irrelevant.

### Implications (what it forces downstream)

* **Non-optional rest**: sustained operating at high (L^{eff}) will necessarily push energy down and damage/drift up, driving A14 demand over time.
* **Compute-to-physiology coupling**: because the cost depends on (L^{eff}(t)), over-activating the library is punished in a way that shows up as “fatigue/damage,” not just compute slack. This couples attention/selection policy to long-run viability.
* **Multi-step REST is structurally required** by the “rates (\ll 1)” constraint; the system cannot “micro-rest” for one tick and fully reset.

### Novelty (what’s actually distinctive)

The novelty is not the existence of fatigue-like variables; it’s that you:

* make **margin dynamics explicit and state-dependent** (OPERATING vs REST) and
* tie operating cost directly to **effective complexity load (L^{eff})**, not merely time or reward.
  This makes “thinking harder” physiologically expensive in a way that is hard to hand-wave away.

### Biological analog

* **Energy depletion / metabolic fatigue**: activity consumes resources; rest replenishes them.
* **Damage accumulation**: stress load and sustained exertion increase wear; rest supports repair.
* **Drift / interference**: prolonged plasticity and ongoing inference can create representational “mess,” with consolidation/renormalization in rest-like phases.

---

## A16 — Observation Geometry and Coverage-Disciplined Foveation (greedy_cov)

### What it means / what it does

A16 defines the agent’s **sensory access discipline**: at each step, the agent only observes a **subset** of the state dimensions via a controllable “fovea,” and the choice of what to observe is driven by a **score that mixes “what seems wrong” with “what hasn’t been checked.”**

Concretely, A16 does four critical things:

1. **Block partitioning (DoF-aligned blocks)**
   The full state dims ({1,\dots,D}) are partitioned into disjoint blocks (\mathcal{B}_b). Each non-anchor node’s mask must live entirely inside exactly one block (the DoF-alignment rule).

2. **Residual and age tracking per block**
   Maintain:

   * **age(b,t)**: steps since block (b) was last observed, and
   * **instantaneous residual** (\hat r(b,t)) when block (b) *is* observed, computed from prediction error on observed dims in that block.

3. **greedy_cov block selection via score(b,t)**
   Choose exactly (F) blocks each step with highest:
   [
   \text{score}(b,t)=r(b,t-1)+\alpha_{cov}\log(1+\text{age}^+(b,t-1))
   ]
   where (\text{age}^+(b,t):=\max(0,\text{age}(b,t)-G)). Debt is **thresholded**: age contributes nothing until it exceeds (G).

4. **Soft coverage aversion: no deterministic bounded coverage guarantee**
   A16 explicitly removes the guarantee that every block will be revisited within a bound. There is **no forced inclusion** solely because a block is old, and **age may be unbounded**.

### Immediate structural hazards / fatal misunderstandings to avoid

These are the “you will silently break the framework” points:

* **DoF-block-aligned masks are a hard expressivity constraint.** If the environment has true cross-block structure, A16 forbids learning a single node that spans it; cross-block dependencies must be captured elsewhere (e.g., via composition in a DAG), or you will be systematically under-expressive. (This identifiability constraint is not optional once adopted.)
* **No bounded reacquisition guarantee** means you must tolerate (and correctly handle) arbitrarily stale blocks without assuming eventual refresh. Any algorithm that assumes “everything gets seen often enough” violates A16.4.
* The selection uses **(r(b,t-1))** (lagged) in score computation, so time indexing matters; mixing current-step residual into selection can introduce causality bugs (observe-after-choose discipline is central to foveation correctness).

### A16.5 — Observation set, stale persistence, and the REST-purged pixel buffer

A16.5 defines:

* The observed dimension set at time (t) as the union of dims in chosen fovea blocks: (\mathcal{O}*t=\bigcup*{b\in \mathcal{F}_t}\mathcal{B}_b).
* For unobserved dims, maintain **stale persistence** in an internal buffer (x_{mem}(t)): the last observed value is retained until re-observed, and completion operates using the observed subset as cue (linking to A13).

And (critically for your project) it introduces the **foveated rolling raw observation buffer** (y_{buf}(t)) for high-bandwidth observations (pixels), with strict semantics:

* It is **bounded by a fixed bit budget** and evicts old entries deterministically when exceeded.
* It may be used to form (x(t)), compute local residuals, and **during REST only** to do reconstruction/audit for structural edit validation.
* It is **prohibited** from being written into any durable store (library/cold storage), and **must be purged before leaving REST** (no pixels after REST).

### Core assumptions

* Prediction error is meaningful at the block level even though observation is partial (i.e., you can compute residual on (\mathcal{O}_t) without biasing the whole system toward “observed-only correctness”).
* Block age and residual together are a valid proxy for “what deserves attention”: residual captures salience; age captures neglected uncertainty.

### Implications

* **Attention becomes debt + surprise**, not “what’s already active.” This is important because cold-storage retrieval is later keyed off the same fovea blocks (addressing scheme is shared).
* **Deliberate blindness is allowed**: because there is no forced inclusion, the agent may rationally starve blocks indefinitely if residual elsewhere stays higher—this is a design choice with safety/performance consequences.
* **Narrow-focus yields longer pixel history** under a fixed budget because only foveated payloads are stored, matching your “baby tracks the finger; rest is shadows” intuition. (This is exactly what the bit-budget + foveation semantics implement.)

### Novelty (what’s actually distinctive)

* Not “foveation” per se; it’s **foveation as an accounting system**:

  * residual becomes a measurable debt signal,
  * coverage debt is thresholded and mixed with residual in a specific score,
  * and the exact same block keys later drive memory retrieval (addressing) rather than similarity search.

### Biological analog

* **Saccades + uncertainty-driven gaze**: eyes repeatedly sample regions with high prediction error or task relevance, while other regions can go unobserved.
* **Change detection**: surprise (prediction error) captures salience; “haven’t looked there in a while” captures uncertainty/forgetting risk.
* **Foveated sensory memory**: short-lived, spatially localized raw sensory traces exist briefly and are not stored verbatim long-term (your REST-purged constraint is an engineered version of that idea).

---

## A17 — Derived Diagnostics (No Control Authority)

### What it means / what it does

A17 defines a set of **logged-only readouts** that are explicitly **for monitoring/analysis**, not for decision-making. It states they have **no control authority** and must **not** be used as gates or triggers.

It defines:

1. **Precision-weighted residual**
   Define residual on observed dims:
   [
   e(t)[k]=x(t)[k]-\hat x(t|t-1)[k],\quad k\in\mathcal{O}*t
   ]
   Then define:
   [
   q*{res}(t)=\text{mean}*{k\in \mathcal{O}*t}\left(|\Lambda*{global}(t)[k]\cdot e(t)[k]|\right)
   ]
   where (\Lambda*{global}) is the fusion precision (inverse covariance) from completion/fusion mechanics.

2. **Maintenance load**
   [
   q_{maint}(t)=L^{eff}(t)
   ]
   i.e., “how heavy is the active set right now.”

3. **Horizon uncertainty at latency floor**
   [
   q_{unc}(t)=H_d(t)
   ]
   using the rollout uncertainty summary at a defined latency floor (d).

4. **3-term feel proxy**
   [
   q(t)=(q_{res}(t), q_{maint}(t), q_{unc}(t))
   ]
   explicitly “logged observable.”

### Immediate structural hazard (this is non-negotiable)

If you use any of these as gates—e.g., “if q_res high then do X,” “if q_unc high then rest,” “if q_maint high then prune”—you have violated A17 by construction. A17 exists specifically to prevent “instrumentation creep” where diagnostics silently become the real controller.

If you *want* these to be control signals, they must be promoted into the controlling axioms explicitly (and then audited like everything else). As written, they are not allowed to acquire causal power.

### Core assumptions

* You already have (i) an observation set (\mathcal{O}*t), (ii) a prediction (\hat x(t|t-1)), and (iii) a meaningful fusion precision (\Lambda*{global}) on observed dims (from the completion/fusion operator).
* (L^{eff}(t)) is computed consistently (so q_maint is comparable over time).
* Rollout uncertainty (H_k(t)) is defined and stable enough that (H_d(t)) is interpretable as “floor uncertainty.”

### Implications

* **Auditable internal telemetry**: q(t) is a compact trace you can log to understand whether the system is failing because of (a) prediction mismatch, (b) load pressure, or (c) planning uncertainty.
* **Prevents Goodharting**: because q(t) cannot be optimized directly (no control authority), it reduces the risk that the agent learns to game the metric instead of solving the task.
* **Interpretability hook**: you can overlay q(t) against macrostate changes, edit decisions, and performance to locate failure modes without conflating “measurement” with “policy.”

### Novelty (what’s actually distinctive)

Again, metrics exist everywhere. The distinctive move is constitutional: you treat certain internal readouts as **diagnostics-only**. That’s a governance decision: it constrains future modifications (including by you) from quietly using convenient proxies as controllers.

### Biological analog

* **Interoceptive readouts vs reflex arcs**: organisms have many measurable correlates (stress hormones, heart rate variability, error signals), but not all are directly used as the actuating trigger for specific behaviors; many are correlational, integrated indirectly, or simply monitored.
* **Subjective “feel” as reportable but not single-cause**: your q(t) is explicitly a *3-vector* (residual, load, uncertainty), which mirrors the idea that “how it feels” is multidimensional and not a single scalar drive.
