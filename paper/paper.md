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