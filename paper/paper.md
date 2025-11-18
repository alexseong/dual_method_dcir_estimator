# A Dual-Method Framework for DC Internal Resistance Estimation
<p align="center">
Shoban Pujari, Alex Seong<br>
November 2025
</p>

## Abstract
Accurate estimation of dynamic direct current internal resistance (DCIR) in lithium-ion cells remains one of the most critical, yet unresolved, modeling bottlenecks in industrial Battery Management Systems (BMS). Conventional voltage-drop based DCIR estimation — while computationally attractive — collapses under non–steady-state excitation, state-of-charge (SOC) dependent hysteresis, temperature-dependent transport losses, and slow relaxation modes that manifest in multi-time-constant impedance behavior. This paper introduces a **physics-informed hybrid modeling framework** that unifies:<br>
(1) a 2-RC Equivalent Circuit Model (ECM) as a physically interpretable dynamic prior,<br>
(2) a differentiable Runge–Kutta solver (Neural ODE) to propagate continuous-time electrochemical state trajectories, and<br>
(3) a temperature-aware residual neural network that learns structured unmodeled dynamics (aging, hysteresis, SEI-layer drift, localized diffusion losses).<br>

Unlike conventional ML regressors, the neural component is never tasked with learning the forward dynamics “from scratch”; instead it learns only the error manifold **orthogonal to physics**, dramatically improving generalization and identifiability.<br>
We further introduce a Bayesian Kalman filtering layer for online inference — allowing parametric DCIR to be tracked in real-time under arbitrary excitation waveforms. Synthetic campaigns and preliminary real-cell validations demonstrate that the hybrid architecture consistently outperforms voltage-drop approaches and pure RK-ECM simulation, particularly in regimes with sharp current steps, high dI/dt, thermal gradients, and low-SOC driving conditions. This establishes a generalizable methodology for cell-level DCIR estimation that scales to real pack operation and aligns directly with the future direction of industrial predictive battery control.

## Nomenclature

| Symbol | Meaning | Units |
|:---:|:---|:---:|
| $$ v_{c1}, v_{c2} $$ | Voltage across first and second RC polarization branches | V |
| $ SOC $ | State of Charge | – |
| $ T $ | Cell temperature | °C (or K) |
| $ I(t) $ | Applied current (positive = charge, negative = discharge) | A |
| $ V(t) $ | Terminal cell voltage | V |
| $ OCV(SOC,T) $ | Open-circuit voltage as function of SOC & temperature | V |
| $ R_0 $ | Instantaneous ohmic resistance | Ω |
| $ R_1, R_2 $ | Polarization resistances (slow & fast modes) | Ω |
| $ C_1, C_2 $ | Polarization capacitances (slow & fast modes) | F |
| $ Q $ | Nominal cell capacity | As or Ah |
| $ \eta $ | Coulombic efficiency | – |
| $ \dot{x} $ | Time derivative of state vector | (various) |
| $ \hat{V} $ | Model-predicted voltage | V |
| $ \Delta V_\theta $ | Neural-network residual correction | V |
| $ L(\theta) $ | Training loss (voltage domain) | V² |
| $ F_{RK4}(\cdot) $ | Runge-Kutta 4th-order time propagator | – |
| KF | Kalman Filter | – |
| DCIR | Direct Current Internal Resistance | Ω |

## 1. Introduction

Reliable estimation of a lithium-ion cell’s **DC internal resistance (DCIR)** under real operating conditions is central to modern battery management systems (BMS). DCIR governs instantaneous **power capability**, **heat generation**, and **voltage sag**, thereby affecting **driver-perceived performance** (acceleration, regen), **safety margins** (thermal run-away risk), and **state-of-health** (SOH) diagnostics. In electric vehicles and stationary storage, the estimator must remain **accurate across a wide range of currents**, **temperatures, and states of charge (SOC)**; it must be **computationally light**, **data-efficient**, and **stable** under measurement noise. Meeting all of these simultaneously is difficult because the cell’s terminal behavior couples fast interfacial phenomena (double-layer charging, SEI, contact resistances) and slow diffusion/transport effects (porous electrode and electrolyte transport), each with different time scales and temperature sensitivities. Any estimator that collapses this multiscale structure into a single lumped constant tends to be biased, especially during transients—which is precisely when the BMS needs accurate predictions. 

Industrial practice often defaults to the Voltage-Drop (current step) method for DCIR: apply a pulse $ \Delta I $ and measure the
corresponding $ \Delta V $; the ratio 
$ 𝑅_{drop} = \Delta V / \Delta I$ is simple, fast, and explainable. Yet in realistic drive cycles and grid profiles, current rarely remains piecewise constant; polarization dynamics continue to evolve well after the step, temperature may drift during the pulse, and measurement noise can corrupt small $ \Delta V $. As a result, $ 𝑅_{drop} $ becomes context-dependent: it varies with the exact timing window, pre-conditioning, and the underlying relaxation state. This leads to over-optimism (underestimating sag during a subsequent burst) or over-conservatism (excess thermal derating), both undesirable for energy and power management.

A classical remedy is to adopt Equivalent Circuit Models (ECMs) to explain transients: a series resistance $𝑅_0$ in line with one or more $𝑅𝐶$ branches that represent polarization. The 2RC structure is widely accepted as the minimum realistic representation for automotive-grade cells because it separates a fast time constant (sub-seconds to a few seconds) from a slow one (tens of seconds and beyond). In tests such as HPPC (Hybrid Pulse Power Characterization) or PRBS-like excitation, a single-RC (1RC) model typically fails to reproduce the long-tail relaxation that governs voltage recovery and heat generation, forcing downstream algorithms to “learn” unphysical corrections. In practice, we also require temperature awareness: $𝑅_0,𝑅_1,𝑅_2$ increase at low $𝑇$ (ionic mobility and conductivity degrade), while $𝐶_1, 𝐶_2$ and the open-circuit voltage $OCV(SOC)$ exhibit their own temperature and SOC dependencies. These effects are nonlinear and coupled; attempting to track them with fixed parametric laws alone (e.g., pure Arrhenius for every component) can be too rigid, while using a fully black-box neural network discards physics and harms extrapolation and interpretability.

This tension motivates hybrid modeling—marrying physics for structure with machine learning for flexibility. In recent years, Neural ODE and physics-informed learning have matured into a practical recipe for such problems: put the known differential equations (the ECM) in the forward pass, integrate them with a differentiable solver (e.g., RK4 at BMS sampling rates), and let a small neural network learn only the residual—the part that the physics cannot explain well (hysteresis, aging drift, path dependence, parasitic leakage). This preserves causality and units, keeps parameters positive (through constrained activations), and still grants the estimator enough capacity to fit complex data. Crucially, gradients flow through the integrator and the ECM states, enabling end-to-end training on raw $(𝐼,𝑉,𝑇)$ trajectories rather than on hand-crafted features.

Beyond modeling fidelity, a production-grade estimator must satisfy operational constraints:

* **Computational economy**: BMS controllers operate on limited hardware at 1–10 Hz (sometimes higher in sub-modules). An explicit **RK4** step is stable and accurate at these rates for the ECM ODEs, avoiding the cost and implementation complexity of fully adaptive/implicit solvers unless the dynamics are truly stiff.

* **Identifiability under realistic excitation**: On-road or in-field data are not textbook pulses. The estimator should remain well-posed for compound profiles (PRBS, WLTC-like driving, grid cycles) and modest sensor noise. Hybrid models help because physics encodes useful priors; the learning component refines rather than invents dynamics.

* **Robustness to temperature and SOC coverage gaps**: Seasonal climate, pack thermal gradients, and nonuniform SOC distributions imply that training data are imbalanced. Physics-informed structure regularizes the model in underrepresented regions and guards against pathological extrapolation.

* **Traceability**: System engineers and safety teams require explainable read-outs—for instance, a time-varying $\hat{R}_0(t)$ that correlates with Voltage-Drop trends and a decomposition of polarization into fast/slow branches. A residual head $ \Delta V_\theta $ should be small and structured, not an opaque correction that dominates the signal.

These realities explain why neither extreme—purely heuristic DCIR formulas nor purely black-box deep nets—offers a satisfying solution. The former are not transient-correct; the latter are hard to certify and brittle when the duty cycle or temperature range changes. The middle path is to keep the ECM as the backbone and empower it with a **temperature-aware parameterization** and a **residual neural head** trained jointly with the **RK4 integrator**. In this setting:

* The **2RC ECM** enforces physically meaningful states $[v_{c1},v_{c2},SOC]$ and the causal relationship between current and voltage.
* The **parameter head** $g_\theta(SOC, T)$ maps operating conditions to $[\hat{R}_0,\hat{R}_1,\hat{C}_1,\hat{R}_2,\hat{C}_2]$ with positivity constraints, capturing smooth thermal/SOC dependencies that are difficult to hand-encode uniformly across chemistries and aging stages.
* The **residual head** $h_\theta(\cdot)$ outputs $\Delta V_\theta$ to absorb residual nonlinearities (minor hysteresis, sensor bias, pack wiring artifacts) while remaining small in magnitude; this preserves interpretability and transfer across similar cells.
* The **Voltage-Drop** method remains in the loop as a baseline and diagnostic: its estimates provide sanity checks and simple field monitors; departures between $R_{drop}$ and learned $\hat{R}_0$ highlight **transient bias** or **temperature-SOC confounding**.

Finally, there is a practical question of data logistics and reproducibility. Many labs and production test benches produce AVL-style CSV logs containing synchronized current, voltage, per-cell voltages, temperature, and metadata. Our pipeline consumes these logs directly, applies consistent sign conventions (charge-positive in the file vs discharge-positive in modeling), infers or accepts the sampling period $\Delta t$, and executes a single training loop that backpropagates through RK4 and both neural heads. The same codebase generates publication-quality plots (voltage fits, parameter trajectories, SOC evolution, error histograms) and enables ablation studies (1RC vs 2RC, residual ON/OFF, analytic OCV vs lookup). This end-to-end path reduces friction between academic modeling and deployable BMS algorithms.

In summary, today’s deployment constraints and accuracy requirements argue strongly for a dual-method DCIR framework: keep the Voltage-Drop method for its speed and transparency, and augment it with a Neural ODE estimator—a temperature-aware 2RC ECM integrated with RK4 and a physics-informed residual neural network—to recover transient-correct behavior with interpretable, physically constrained parameters. This hybrid approach respects the realities of embedded systems while matching or exceeding the accuracy of far heavier identification procedures. The rest of this paper develops the method in detail, establishes its identifiability and stability properties, and demonstrates its superiority on both synthetic stress tests and real AVL datasets spanning large current excursions, wide SOC ranges, and meaningful temperature variations.

## 2. Literature Review / Related Work
Estimation of DC internal resistance (DCIR) in contemporary lithium-ion cells has gradually evolved into a core topic at the intersection of electrochemistry, control engineering, automotive BMS deployment, diagnostic AI, and scientific machine learning. Although the concept of DC resistance has existed since the earliest days of battery instrumentation, the way in which the community defines, measures, and operationalizes DCIR has changed notably over the past decade due to the convergence of three forces: (i) the electrification of transportation and the dramatic increase in dynamic loading severity, (ii) the rising operational demands placed on BMS logic for safety and performance, and (iii) the shift from laboratory-grade stationary pulse measurements toward in-field measurements taken under uncontrolled vehicle operating conditions. Consequently, the literature landscape has become considerably fragmented across multiple methodological families, each solving slightly different aspects of the same estimation problem, yet none of them individually fully satisfying the requirements for accuracy, robustness, physical interpretability, and real-time deployability in production EVs.

Historically, the industry standard has adopted the voltage-step or voltage-drop DCIR measurement paradigm, in which a commanded current step is applied and the DCIR value is computed directly via the ratio of instantaneous $\Delta V$ over $\Delta I$. This procedure remains institutionalized in HPPC protocols, end-of-line manufacturing tests, and cell grading pipelines because it is fast, intuitive, and computationally trivial. However, numerous studies have shown that voltage-drop DCIR is inherently window-dependent, algorithmically sensitive to transient reaction kinetics, and thermally biased. It does not represent a single intrinsic material resistance, but rather a superposition of multiple time-constant dependent polarization phenomena. Modern cells exhibit dual-scale relaxation: a fast interfacial component often settling within seconds, and a slower diffusion-driven component that may continue evolving for tens of seconds or longer. These two exponentially separated mechanisms directly imply that the DCIR measured through $\Delta V$/$\Delta I$ is fundamentally a functional of the measurement protocol rather than a stable physical property. This problem becomes pronounced at low SOC, cold temperature exposure, and aging — precisely the corners where diagnosis is most safety-critical.

To address the window-dependence and protocol sensitivity, the community progressively shifted toward physics-structured estimation based on Equivalent Circuit Models (ECM). Early ECM methods used a single RC branch (1RC) primarily because it was analytically simple and identifiability was manageable. Yet over the past ten years, it has become universally accepted — particularly in the automotive literature — that at least 2RC is required to reproduce the long-tail relaxation observed in real cells. The multi-time-constant structure embedded in 2RC models more faithfully captures the superposed polarization modes, and thereby yields more stable DCIR estimation when driven by dynamic excitation profiles rather than dedicated pulse windows. ECM research has produced a range of identification algorithms, from recursive least squares to extended Kalman filtering to sliding-mode observers. However, these conventional approaches still rely on globally parametric, fixed-form parametric maps for resistance and capacitance versus temperature and SOC — usually linear or Arrhenius-type curve fits. These fixed forms are not expressive enough to handle real, high-dimensional, vehicle-grade operating envelopes where thermal gradients, SOC inhomogeneity, load hysteresis, and surface-film dynamics interact in non-trivial ways. The result is that classical ECM remains interpretable and fast, but cannot adapt to complex dynamic environments without over-regularizing the physics or over-fitting the parametric maps.

On the opposite end of the spectrum from ECM are P2D-style electrochemical models. These models theoretically possess the most physical fidelity, as they explicitly resolve lithium intercalation kinetics, electrolyte concentration gradients, and solid diffusion. Theoretically, DCIR computed from such PDE-based frameworks should be closer to the “true” internal resistance. However, the computational cost, parameter non-identifiability, requirement for implicit PDE solvers, and lack of robust real-time realizability make these models incompatible with embedded BMS targets — particularly when required update rates exceed 10 Hz for multi-module EV packs. Even reduced-order versions such as the Single Particle Model (SPM) cannot be readily deployed for real-time DCIR inference in production EV platforms without model-order reduction or offline trained surrogates.

This tension between physical accuracy and deployment feasibility led to the emergence of ML-based battery models. The early ML literature primarily framed DCIR estimation as a regression task using engineered features extracted from pulse responses, relaxation slopes, or domain-expert-crafted thermal descriptors. While these approaches achieved improved empirical fitting, they remained purely black-box, fundamentally lacked causal constraints, and yield non-physical extrapolation outside the training domain — a serious problem because batteries age continuously, environment varies seasonally, and BMS logic must be robust not only to the conditions seen during training, but to unseen corner cases. Certification and interpretability therefore remain open issues.

The most recent wave of literature — and the direction this paper follows — is the scientific machine learning paradigm: Physics-Informed Neural Networks, Neural ODE frameworks, and hybrid models in which neural networks do not replace the physics but augment it by learning residual dynamics. This class of approaches has reframed the role of machine learning in battery modeling: rather than replacing the ECM entirely, the ECM differential equations themselves become part of the neural forward model. In this framework, 2RC ECM dynamics are integrated explicitly within the forward pass using numerical ODE solvers such as Runge–Kutta. The neural components are then trained to learn only the parts of the system that the physics cannot express — for example nonlinear temperature dependence, state-of-aging effects, and fine-scale deviations not captured by the nominal ECM structure. In effect, the physically-derived ODEs provide structural priors, while neural residual maps fill only the gaps. This hybrid approach has emerged as a synthesis that resolves the long-standing triad of contradictions: it preserves interpretability, controls generalization, and enables deployment. Thus, this scientific machine learning direction — neural ODEs + physics-structured ECM + residual networks — defines the methodological frontier and provides the conceptual foundation upon which the present study is built.

## 3. Methodology

This section presents the proposed hybrid DCIR estimation framework in full detail. The model is built around a temperature-aware, two-branch equivalent circuit representation (2RC ECM) embedded in a differentiable ODE integrator (RK4), with two neural submodules: a parameter head that maps operating conditions to physically constrained ECM parameters, and a residual head that corrects remaining voltage mismatch stemming from hysteresis, aging, or unmodeled pack effects. The overall design preserves physical interpretability, supports end-to-end learning from raw current–voltage–temperature sequences, and yields a DCIR read-out that is both transient-aware and aligned with industrial Voltage-Drop practices.

### 3.1 Overall formulation and problem statement
We denote the measured input–output sequence as $\{I_k, V_k, T_k\}_{k=0}^N$ sampled with period $\Delta t$. We adopt the **discharge-positive** modeling convention: positive $I$ denotes discharge power flow into the external load (note that many datasets log charge-positive; those are sign-flipped at ingestion). The goal is to learn a predictor $M_\Theta$ such that the simulated terminal voltage $V_k^{pred}$ tracks $V_k$ under realistic cycles while yielding interpretable, temperature/SOC-dependent parameters whose ohmic component constitutes a model-implied DCIR:<br>
$$
\hat{R}(k) = (\text {ohmic series resistance output by the parameter head at step } k).
$$
We concurrently retain a classic Voltage-Drop estimator $R_{drop}$ computed on screened pulse windows as a diagnostic baseline; agreement between $\hat{R}_0$ and $R_{drop}$ on quasi-steady pulses, and their divergence during transients, is a central validation axis.

The hybrid model consists of:

1. a **2RC ECM** that encodes the causal dynamics between current and voltage,
2. a **parameter head** $g_\theta(SOC, T)$ that outputs $(R_0,R_1,C_1,R_2,C_2)$ with positivity guarantees,
3. a **residual head** $h_\phi(\cdot)$ that produces a small voltage correction $\Delta V_\phi$ from local states and inputs,
4. a **differentiable RK4 integrator** that advances the ECM states in time.

### 3.2 Two-RC equivalent circuit model (state-space form)
**States and output**. We model two polarization branches and the bulk charge inventory:
$$
x = \begin{bmatrix} vc1 \\ vc2 \\ SOC \end{bmatrix} \text{, \hspace{1cm}}   V = OCV(SOC) - R_0I - v_{c1} - v_{c1} + \Delta V_\phi.
$$

**Continuous-time dynamics**.
$$
\dot{v}_{c1} = -\frac{v_{c1}}{R_1C_1} + \frac{I}{C_1} \text{, \hspace{1cm}} \dot{v}_{c2} = -\frac{v_{c2}}{R_2C_2} + \frac{I}{C_2} \text{, \hspace{1cm}} \dot{SOC} = \frac{\eta}{Q}I.
$$

Here $R_0,R_1,R_2>0, C_1,C_2>0, Q>0$(As), and $0\lt\eta\leq1$. The OCV–SOC map is assumed smooth and monotone on $[0,1]$, ealized either by an analytic surrogate or a calibrated lookup with interpolation. This **minimum 2RC structure** separates a fast interfacial time constant and a slow diffusion/transport time constant; Section 6 ablates 1RC vs 2RC to show why 2RC is the minimal physically faithful choice for automotive-class cells.

**Discrete time via integration**. Let $x_k\approx x(k\Delta t)$. For a given current sample $I_k$ and parameters at $k$, we advance $x_k\mapsto x_{k+1}$ with RK4 (Sec. 3.5). The terminal voltage at step $k$ is then:
$$
V^{pred}_{k} = OCV(SOC_k) - R_{0,k}I_k - v_{c1,k} - v_{c2,k} + \Delta V_{\phi k}.
$$

### 3.3 Temperature-aware parameterization (neural parameter head)
The ECM parameters are not constants; they vary with operating condition in ways that are difficult to capture with global parametric laws alone. We therefore define a **small, structured regressor**:
$$
(R_{0,k},R_{1,k},C_{1,k},R_{2,k},C_{2,k}) = g_\theta(SOC_k, T_k),
$$
with the following design constraints.

**Positivity and scale enforcement**. To guarantee $R_i, C_i > 0$ and discourage degenerate scales we use shifted Softplus outputs:
$$
R_i = \epsilon_R + \alpha_R \, \text{softplus}(z_i)\text{, \hspace{1cm}}C_i = \epsilon_C + \alpha_C \, \text{softplus}(w_i), 
$$
with small $\epsilon_{R,C} > 0$ and calibration coefficients $\alpha_{R,C}$ chosen relative to expected magnitudes (e.g., $m\Omega$ for $R_0$, $\Omega$ for $R_{1,2}$, $10^2 - 10^4$ F for $C_{1,2}$ depending on cell format). This avoids negative/unphysical parameters and stabilizes gradients.

**Context inputs**. We use $(SOC,T)$ as primary predictors. If available, additional slow-varying covariates (estimated state-of-health, cycle count) can be appended without altering the rest of the formulation. The mapping capacity is deliberately small (e.g., two hidden layers with 32–64 units, SiLU/Tanh activations) to reduce overfitting and to retain smooth, low-variance parameter trajectories.

### 3.4 Residual voltage correction (neural residual head)
Even with temperature-aware parameters, there remain effects that the canonical ECM cannot express: minor hysteresis, sensor biases, wiring drops, local heating, age-dependent offsets. We capture these with a **residual head**:
$$
\Delta V_{\phi,k} = h_\phi(v_{c1, k}, \, v_{c2, k}, \, \text{SOC}_k, \, I_k, \, T_k),
$$
again as a small MLP (e.g., widths 32–64) with zero-mean initialization to bias early training toward the pure ECM. We constrain $\Delta V_\phi$ implicitly through the loss (Sec. 3.6) and, optionally, through a magnitude regularizer to keep the residual **small** and **structured**; the intent is **physics-informed augmentation**, not replacement of the ECM.

### 3.5 Differentiable RK4 integration
We integrate the continuous dynamics using classical **Runge–Kutta 4th order** at the BMS sampling rate. Let $f(x, I, Q_{p,k})$ denote the right-hand side where $\theta_{p,k} = (R_{1,k}, (C_{1,k},(R_{2,k}), (C_{2,k}, Q, \eta)$. For step $k$:
$$
\begin{align*}
k_1 &= f(x_k, I_k; Q_{p, k}), \\
k_2 &= f(x_k + \tfrac{\Delta t}{2}k_1, I_k; Q_{p, k}), \\
k_3 &= f(x_k + \tfrac{\Delta t}{2}k_2, I_k; Q_{p, k}), \\
k_4 &= f(x_k + \Delta t \, k_3, I_k; Q_{p, k}), \\
x_{k+1} &= x_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k4).
\end{align*}
$$

All operations are differentiable; in PyTorch/JAX, automatic differentiation propagates through $g_\theta, h_\phi, $ and the RK4 updates, enabling end-to-end training on time series. For typical sampling (1–10 Hz) and ECM time scales, explicit RK4 is stable; if strong stiffness is observed (e.g., very small $R_iC_i$ at low $T$, one can swap RK4 for a semi-implicit variant without changing the rest of the learning machinery.

### 3.6 Training objective and regularization
**Primary fit**. On a sequence of length $N$, we minimize mean-squared voltage error:
$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{k=0}^{N-1}(V_k^{\text{pred}} - V_k)^2.
$$
**Temporal smoothness of parameters**. To discourage implausible jitter in $R_i, C_i$, we add a discrete Tikhonov regularizer:
$$
\mathcal{L}_{\text{smooth}} = \frac{1}{N-1}\sum_{k=0}^{N-2}\left\lVert \theta_{p, k+1} - \theta_{p, k}\right\rVert_2^2\text{, \hspace{1cm}} \theta_{p,k} = [R_{0,k}, R_{1,k}, C_{1,k}, R_{2,k}, C_{2,k}]^\text{T}.
$$
**Residual discipline (optional)**. To keep $\Delta V_\phi$ from dominating:
$$
\mathcal{L}_{\text{res}} = \frac{1}{N}\sum_{k=0}^{N-1}(\Delta V_{\phi, k})^2.
$$
This term is weighted lightly to avoid suppressing legitimate corrections.

**Weight decay and gradient control**. We employ AdamW with small weight decay and clip global gradient norm (e.g., $\left\lVert \nabla \right\rVert \le 1$ to stabilize long unrolls.

**Total loss.**
$$
\mathcal{L} = \mathcal{L}_\text{MSE} + \lambda_{\text{sm}}\,\mathcal{L}_\text{smooth} + \lambda_{\text{res}}\,\mathcal{L}_\text{res} + \lambda_{\text{wd}} \left\lVert \Theta \right\rVert_2^2,
$$
with $\Theta$ denoting all trainable weights. We tune $\lambda_{\text{sm}}, \lambda_{\text{res}}, \lambda_{\text{wd}}$ on validation sequences to balance fidelity and physical plausibility.

### 3.7 OCV modeling

The OCV–SOC relation strongly influences identifiability. We support two realizations:

1. **Calibrated table + interpolation** (preferred for accuracy): a monotone cubic interpolation of lab-measured OCV vs SOC at multiple temperatures; if only a single temperature is available, we use it across all $T$ and allow residuals/parameters to absorb small thermal shifts.

2. **Analytic surrogate** (useful for ablations and synthetic data): a smooth, S-shaped function that captures the plateau and sharp knees near low/high SOC. Parameters of the surrogate are fixed during training to avoid entangling OCV shape with resistance learning.

In both cases, clipping ensures $\text{SOC} \in [0, 1]$; extrapolation outside this interval is prevented.

### 3.8 Sign conventions, units, and preprocessing

**Current direction**. Many bench logs are charge-positive; we convert to **discharge-positive** ($I_{\text{model}} = -I_\text{log}$) so that $R_0\,I$ contributes a positive sag during discharge.

**Sampling period**. We infer $\Delta_t$ from timestamps or header metadata and resample if necessary to achieve uniform spacing compatible with RK4.

**Voltage source**. If per-cell voltages are present, we average them for a pack-level terminal voltage; otherwise we use the top-level measured terminal voltage. Minor sensor noise can be attenuated with a mild Savitzky–Golay filter on visualization only (the training target should not be aggressively prefiltered to avoid bias).

**Temperature**. We use the most representative temperature channel (surface thermistor, coolant, or estimated core temperature) and convert to Kelvin. If multiple probes exist, a simple average or a learned linear combination can be used; our experiments show robustness to reasonable choices.

### 3.9 Identifiability and stability considerations

**Why 2RC (identifiability).** Under PRBS/pulse excitation, the voltage response exhibits at least two distinct exponential relaxations. A 1RC model structurally cannot represent the long tail; as a result, the residual head will be forced to emulate missing physics, harming extrapolation and interpretability. Using 2RC confines the residual to genuinely unmodeled effects and yields smoother, more physical parameter trajectories.

**Role of OCV.** Mis-specified OCV slopes can be partially compensated by $R_0$ and the residual, leading to parameter leakage. A measured OCV table (or a well-tuned surrogate) alleviates this. We avoid learning OCV jointly with resistances in the baseline configuration; if desired, one can learn a small correction to OCV with strong smoothness penalties.

**Temperature coupling.** $R_0,\,R_1,\,R_2$ typically increase at low $T$, while $C_1, \, C_2$ and OCV may shift nonlinearly. Feeding 
$T$(and $SOC$) to the parameter head improves identifiability by explaining systematic trends without forcing the residual to absorb them.

**Numerical stability.** Explicit RK4 is stable at 1–10 Hz for typical ECM time constants; if step sizes become large relative to 
$R_i\,C_i$, reduce $\Delta t$ or adopt a semi-implicit update (e.g., trapezoidal for the RC states) — the learning objective and parameterization remain unchanged.

### 3.10 Implementation details (for reproducibility)

**Networks.** Parameter head: MLP with inputs $(\text{SOC}, T)$ hidden layers of 32–64 units, SiLU activations, Softplus-constrained outputs for $(R_0,\,R_1,\,C_1,\,R_2,\,C_2)$. Residual head: MLP on $(v_{c1},\, v_{c2},\, \text{SOC},\, I,\, T)$  with similar width, last-layer bias initialized to zero.

**Initialization.** States start at $v_{c1} = v_{c2} = 0,\, \text{SOC}_0$ from metadata or a reasonable prior. Parameter head biases are initialized to nominal values (e.g., $R_0$ a few $\text{m}\Omega, \, R_{1,2}$ tens of $\text{m}\Omega\text{--}\Omega, C_{1,2}\,10^2 - 10^4\,\mathrm{\small F}$) to speed convergence.

**Optimization.** AdamW (lr $10^{-3}-2\cdot10^{-3}$), weight decay $10^{-6}$, gradient clip 1.0, 40–60 epochs for single-cell datasets; mini-batches are contiguous windows to preserve temporal coherence.

**Regularization.** $\lambda_\text{sm}$ on the order of $10^{-4}-10^{-3}$ stabilizes parameter time series without oversmoothing; $\lambda_\text{res}$ small ($\le10^{-3}$) to keep residuals modest.

**Ablations.** Residual ON/OFF, 1RC vs 2RC, analytic vs table OCV, temperature-aware vs temperature-ignorant parameter head; we report voltage RMSE, parameter smoothness, and alignment with $R_\text{drop}$.

### 3.11 Algorithmic summary (training loop)

Given a sequence $\{I_k,\,V_k,\,T_k\}_{k=0}^N$:
1. Initialize $x_0 = [0, 0, \text{SOC}_0]^\text{T}.$
2. For $k = 0$ to $N-1$:
    * a) Compute $(R_{0, k}, \, R_{1, k}, \, C_{1, k}, \, R_{2, k}, \, C_{2, k}) = g_\theta(\text{SOC}_k, T_k)$.
	* b) Form $V_k^\text{ECM} = \text{OCV}(\text{SOC}_k) - R_{0,k}I_k-v_{c1,k}-v_{c2,k}.$ 
	* c) Compute $\Delta V_{\phi, k} = h_\phi(v_{c1,k}, v_{c2,k}, \text{SOC}_k, I_k, T_k).$
	* d) Set $V_k^\text{pred} = V_k^\text{ECM} + \Delta V_{\phi, k}.$
	* e) Advance states $\,x_{k+1} = \text{(RK4)}(x_k,\, I_k,\, \theta_{p,k};\, \Delta t)$ 
3. Accumulate $\mathcal{L} = \mathcal{L}_\text{MSE} + \lambda_{\text{sm}}\,\mathcal{L}_\text{smooth} + \lambda_{\text{res}}\,\mathcal{L}_\text{res} + \lambda_{\text{wd}} \left\lVert \Theta \right\rVert_2^2.$
4. Backpropagate through the entire unrolled trajectory; update $(\theta,\, \phi)$ with AdamW.
5. Repeat over sequences; select hyperparameters using a held-out validation set; report metrics on test sequences.

### 3.12 DCIR read-outs and interpretation

For **operational traceability**, we report:

* **Model-implied ohmic DCIR** $\hat{R}_0(k)$ from the parameter head, interpretable as the instantaneous series resistance at $(\text{SOC}_k, \, T_k).$ 
* **Voltage-Drop baseline** $R_\text{drop}$ computed on screened pulses (with outlier rejection and window standardization).

Under quasi-steady conditions with small polarization evolution, $\hat{R}_0$ and $R_\text{drop}$ should align within measurement noise. During strong transients or temperature drift, $R_\text{drop}$ will be biased by evolving $v_{c1}, v_{c2} $, while $\hat{R}_0$
 remains stable—precisely the value of a physics-informed, transient-aware estimator.

**Summary**. The proposed **temperature-aware 2RC ECM + RK4 + residual neural network** constitutes a Neural ODE estimator: physics provides the backbone; neural heads provide flexible, smooth parameter mappings and a disciplined residual correction. The design achieves transient-correct voltage prediction, interpretable DCIR trajectories, and practical computational cost suitable for BMS deployment, while retaining compatibility with standard Voltage-Drop diagnostics.

## 4. Data, Preprocessing, and Sign Conventions

The present work employs real automotive-grade lithium-ion cell measurement data obtained from a production-class high-precision battery cycler (AVL–class bench). The dataset contains time-resolved current, terminal voltage, temperature, and cell-level telemetry at 1 Hz sampling frequency, representing multiple operating segments such as rest, low-power conditioning phases, large pulsed discharge/charge steps, and mixed transient regimes. The dataset therefore contains both quasi-steady DC voltage plateaus and rich first-order and second-order relaxation responses; the latter are exactly the type of transient fading behavior that separate the 2RC ECM family from the traditional 1RC OCV–R model.

Before training, several preprocessing steps are required — not to artificially beautify the signal, but to formalize the physical consistency of the neural ODE problem, prevent incorrect parameter inference, and enforce sign and frame conventions that are unfortunately inconsistent across cycler vendors, automotive test lines, and academic research. Unlike naive ML battery prediction papers, these steps are not simply “data cleaning.” In physics-informed inverse modeling, a single violated sign convention can entirely invalidate ODE inference (for example, the same +current could mean discharge on one bench and charge on another). Therefore this section explicitly fixes conventions and makes them non-ambiguous for reproducibility.

### 4.1 Data Fields and Sampling Geometry

Each measurement record consists of:
* timestamp (wall time)
* sample index (implicit “RecordingTime” column)
* measured terminal voltage (pack-level or cell-averaged, depending on configuration)
* measured signed current (A)
* temperature channel ($^\circ \text{C}$) from pack thermistor or chamber sensor
* auxiliary columns (internal cell voltages, counter registers, time-in-test, etc.)

Only the tuple {Voltage, Current, Temperature, Time} is used to estimate dynamic DCIR in this paper. Other fields are retained to allow future structural modeling (SOH progression, cell balancing dynamics, inter-cell variance, etc.) but are not used here for parameter learning.

### 4.2 Fixed Sign Convention (Critical)

Different vendor SW uses divergent sign conventions.
For this work we enforce the following strict global rule:

* **Discharge current is defined as positive**
* **Charge current is defined as negative**

This is aligned with physical energy flow into an external load.

If the raw dataset uses the opposite sign (e.g., PEC_Measured_Current > 0 meaning charging), then the sign is flipped at ingestion. All ECM differential equations, RK4 steps, residual network inputs, smoothness losses, and DCIR outputs assume this convention. If this rule is broken, the neural ODE will learn inverted causal slopes, generating physically impossible $R_0$ posteriors.

Therefore this step is non-negotiable for proper scientific reproducibility.

### 4.3 Voltage Reference Normalization

Voltage is not normalized, but is used in absolute terminal units.
ECM and OCV functions assume absolute voltage in volts.

No min-max scaling is applied to voltage because doing so destroys OCV curvature information; this curvature is one of the few observables with direct electrochemical meaning.

### 4.4 Temperature Treatment

Temperature is retained as an explicit input channel and fed into the parameter-head network. Temperature is not merely metadata: in automotive traction Li-ion, $R_0$, $R_1$, $R_2$, as well as relaxation spectra are strongly temperature-dependent. Temperature is also slow-varying relative to transient current steps, which means it should not be KL-smoothed or differentiated in the time domain.

We do not normalize temperature to [0,1]; instead we standardize (mean-std) over the dataset.
The model consumes it as a physical predictor, not a hidden latent variable.

### 4.5 Removing Non-Dynamic Segments

Segments of zero-current rest conditions are preserved but annotated.
They are not discarded, because:
* they encode OCV
* they define SOC drift constraints
* they help parameter identifiability

However, when we compute evaluation metrics for DCIR $drop - based$ comparison, very long zero-current plateaus are excluded (e.g., overnight stabilization) so that classical $\Delta V/\Delta I$ comparisons remain meaningful.

### 4.6 No Filtering of Dynamics

We explicitly **do not** apply Savitzky–Golay smoothing or low-pass filtering on the voltage trace. Filtering would suppress exactly the time-constant structure that differentiates 2RC from 1RC. Prior works that apply smoothing inadvertently erase the physics they intend to estimate.

Noise is handled only through the residual network.

### 4.7 Data Windowing and Training Batches

Training is performed on contiguous sliding windows extracted from long sequences, preserving causal structure. We do not shuffle time. Unlike regular ML forecasting tasks, neural ODE models are not permutation-invariant — time ordering is essential.

### 4.8 Why This Section Exists

Most papers treat “data preprocessing” as a housekeeping detail.
Here, the data section is central — because DCIR estimation is not purely supervised regression. DCIR estimation is a partial identification problem governed by:

* a known nonlinear OCV–SOC relation
* two first-order RC relaxation poles
* temperature-dependent ohmic term
* sign-constrained energy-flow convention

If any of these are violated at the data interface stage, the entire downstream neural ODE inference becomes physically meaningless.

Thus, this section defines the preconditions needed so that the neural ODE can learn real resistance, not a statistical surrogate.

## 5. Experiment Protocol

This section specifies the full experimental pipeline used to train, validate, and test the proposed temperature-aware 2RC Neural-ODE DCIR estimator. We define data splitting, windowing, baseline construction, optimization schedules, ablations, and statistical reporting. Every choice is made to ensure **reproducibility**, **physical validity**, and **fair comparison** to the classical Voltage-Drop DCIR method.

### 5.1 Data partitions and coverage guarantees

**Sequences and sampling.** Each experiment uses contiguous time series drawn from bench logs at 1 Hz (or uniformly resampled to $\Delta\text{t}$ if the native cadence differs). We require that all three channels are present: terminal voltage $V$, current $I$, and temperature $T$. If multiple temperature probes exist, we select a representative channel (surface/coolant/core estimate) consistently across partitions.

**Stratified splits.** To prevent train/test leakage of quasi-identical conditions and to guarantee coverage across operating regimes, we split by **(SOC band, temperature band, excitation type)**:
* SOC bands: $[0.05,0.3), [0.3,0.7), [0.7,0.95]$
* Temperature bands ($^\circ \text{C}$): $[−20,−5), [−5,10), [10,25), [25,45], (45,60]$
* Excitation: rest/OCV holds, step pulses (HPPC-like), PRBS/mixed drive (WLTC-like)

Within each stratum we assign 60% of segments to **train**, 20% to **validation**, and 20% to **test**, ensuring that a stratum present in test is also represented in train (possibly with different exact traces). When data is scarce in extreme corners (e.g., very cold), we down-weight those strata in training but **do not** exclude them from test; this penalizes models that cannot extrapolate physics.

**Randomization and seeds.** We report results averaged over three stratified seeds (s=11,17,23). For each seed, the stratified sampler draws disjoint segment IDs before windowing (Sec. 5.3). All model initializations, minibatch orders, and bootstrap resamples (Sec. 5.8) use the same seed to enable exact reproduction.