# PPO Control of a Real Inverted Pendulum (Sim-to-Real)

Master's project at the National University of Singapore (2025–2026).
A neural-network policy trained with continuous-action PPO replaces the LQR of a commercial
Acrome cart–pole pendulum and runs on the real hardware (Raspberry Pi, 250 Hz, Simulink + Embedded Coder).

<!-- Video:https://github.com/user-attachments/assets/20ebd535-814c-48e8-a4e5-8f570861dfc7-->

## Results on the hardware

Last 45 s of steady balancing, injection segments removed.

| Controller | θ std | θ peak | PWM std |
|---|---|---|---|
| PPO policy trained from scratch (S8) | 0.98° | **4.1°** | 114 |
| Network clone of the LQR, gains × 0.85 (best linear controller tested) | 0.99° | 6.1° | 119 |
| Manufacturer's LQR block (recorded on another day) | 1.39° | 6.1° | 132 |

- The policy trained from scratch reached the level of the best linear controller tested, with the lowest peak angle.
  Recorded on the same day as the manufacturer's LQR, it balanced about equally well and had a lower peak sensitivity.
- A policy retrained in a calibrated simulator (S12) kept a larger stability margin on the hardware, as predicted
  before the test: closed-loop sensitivity |S| at 2.5 Hz was 3.3–3.4, against 5.6–5.7 for S8.
- A slow integral outer loop removed the cart offset caused by the drifting angle-sensor zero (−124 mm → −3 mm).

## What was hard

- **Policies that balanced perfectly in simulation failed on the hardware.** A network trained to imitate the LQR,
  run through the same deployment path, showed that deployment worked and the simulator did not.
- **Two interface errors in the velocity signals** (wrong sign on the cart velocity, wrong unit scaling on the angular
  velocity), and the bandwidth of the velocity estimates turned out to decide whether a policy could run at all.
- **Closed-loop system identification** with a stepped-sine probe showed the simulator lacked 40–90° of phase lag.
  With friction and inertia measured in open-loop tests, the calibrated simulator predicted the manufacturer's LQR
  sensitivity at eight frequencies within 20 %, before the data were recorded.
- **Hinge constraints on the actor loss** encode the properties exposed by each failed hardware test (gain at the
  oscillation frequency, far-field thrust, channel signs, velocity gain), because they are properties of the network
  function and cannot be written as per-step rewards.
- **Pre-flight checks before every hardware run**, evaluated on the states the policy actually visits.

## Repository layout

| Folder | Contents |
|---|---|
| `training/` | PPO training: `train_rl_scratch_S8.m` (best hardware policy), `train_rl_S12.m` (calibrated simulator + domain randomisation), `train_lqr_clone.m`, `train_rl_ABL.m` (reward ablation), `train_rl_EST.m` (estimator bandwidth test) |
| `calibrated_sim/` | Calibrated simulator (`v1_*.m` for MATLAB training, `sim_model_v1.py` for Python), plant identification from probe data |
| `deployment/` | Simulink model `PendulumQuickStart_RL.slx` and the controller functions pasted into its `RL_Controller` block (`fcn_S8S12_ABAB.m`: S8/S12 A/B test with outer loop and probe; `fcn_S8_OL.m`: outer-loop centring; `fcn_LQR_static.m`) |
| `preflight/` | Checks run before any network goes on the hardware |
| `analysis/` | Hardware data analysis, LQR weight scan, velocity-chain check |
| `prototypes/` | My early hand-written RL code: REINFORCE → A2C → n-step A2C → PPO → continuous PPO with squashed Gaussian, plus a SAC cross-check |
| `models/` | Trained networks (`actor_net_S8_deploy.mat`, `actor_net_S12_deploy.mat`, LQR clone) |

Policy: actor 4→64→64→2, critic 4→64→64→1, squashed Gaussian; GAE (γ 0.99, λ 0.95), clip 0.2.
State `[x, v/5, α, ω/10]`, action `pwm = 1000·tanh(μ)`.

## Notes

- Raw hardware recordings are not included, so the analysis scripts show the method but will not run as is.
- To deploy: open the model, paste one of the `deployment/fcn_*.m` functions into `RL_Controller`, put the network
  `.mat` next to the model, and set your own Raspberry Pi address and login in Hardware Settings
  (they have been cleared from this copy). The block currently holds the code of the last experiment.
- `PendulumQuickStart_RL.slx` is derived from the manufacturer's example model for the Acrome pendulum.
- Requires MATLAB R2025a with Deep Learning Toolbox, Simulink, Embedded Coder and the Raspberry Pi support package;
  Python scripts need NumPy, SciPy, pandas and Matplotlib.
