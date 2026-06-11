# Autonomous UAV Fire Detection — Weekly Project Timeline
## October 2025 – May 2026

---

## OCTOBER 2025

**October — Week 1 (Oct 1–7)**
Problem scoping and initial research. Reviewed wildfire detection literature, explored satellite and UAV-based fire monitoring limitations. Identified the core research gap: existing systems cannot detect under-canopy early-stage fires. Decided on simulation-based approach.

**October — Week 2 (Oct 8–14)**
Literature survey continued. Read key papers on DRL-based UAV navigation, GPS-denied flight, and AirSim simulation. Shortlisted references for the CENG400 design proposal. Drafted the initial system concept: AI Tower → Bearing → PPO hybrid architecture.

**October — Week 3 (Oct 15–21)**
Wrote the CENG400 design proposal. Defined the three-layer architecture (Strategic, Tactical, Perception). Outlined functional and non-functional requirements. Submitted first draft for supervisor review.

**October — Week 4 (Oct 22–31)**
Set up the development environment. Installed Unreal Engine, AirSim, Python venv, and required libraries (airsim, opencv-python, numpy, stable-baselines3, gymnasium). Tested AirSim connection with a default multirotor scene.

---

## NOVEMBER 2025

**November — Week 1 (Nov 1–7)**
AirSim + Unreal Engine integration. Configured the Unreal forest scene with tree meshes and terrain. Verified that the multirotor could take off, hover, and land via AirSim Python API. Confirmed RGB camera and depth camera data streams.

**November — Week 2 (Nov 8–14)**
Sensor configuration and physics validation. Tested IMU data, confirmed collision detection events, and verified depth camera range and resolution. Wrote `connect_test.py` and `camera_test.py` as diagnostic scripts.

**November — Week 3 (Nov 15–21)**
Started the AI Tower Monitor. Implemented baseline frame capture logic — the tower saves a clean reference image of the forest scene from a fixed high viewpoint.

**November — Week 4 (Nov 22–30)**
Implemented grayscale frame differencing in `ai_tower_monitor.py`. Tuned the pixel change threshold to avoid noise-triggered alarms. Saved the first difference mask images.

---

## DECEMBER 2025

**December — Week 1 (Dec 1–7)**
Added HSV fire/smoke masking to the tower monitor. Defined the HSV color ranges to detect orange, red, and smoky-grey tones. Applied the mask on top of the frame difference result to filter out lighting artifacts.

**December — Week 2 (Dec 8–14)**
Implemented contour detection and centroid extraction on the masked change region. Tested with a manually placed fire actor in the Unreal scene. Successfully localized the fire centroid in pixel coordinates.

**December — Week 3 (Dec 15–21)**
Implemented the bearing formula: converts centroid pixel position to a compass bearing using camera field-of-view and tower yaw angle. First alarm image with annotated centroid and bearing text generated.

**December — Week 4 (Dec 22–31)**
Started `bearing_navigator.py`. Implemented the basic UAV dispatch: arm, take off, fly in the direction of the calculated bearing. Validated that the drone moves toward the correct quadrant of the scene.

---

## JANUARY 2026

**January — Week 1 (Jan 1–7)**
Added cross-track error (CTE) correction to the bearing navigator. The UAV now corrects lateral drift from wind or physics jitter to stay close to the bearing ray. Flight frame images started being saved during navigation.

**January — Week 2 (Jan 8–14)**
Implemented altitude hold logic in the bearing navigator. Tuned the target cruise altitude to maintain under-canopy level flight without clipping the terrain or hitting lower branches.

**January — Week 3 (Jan 15–21)**
Added smoke density monitoring to the bearing navigator. The UAV's front RGB camera is analyzed each step using HSV masking. Smoke density percentage is logged.

**January — Week 4 (Jan 22–31)**
Implemented the phase handoff trigger: when smoke density exceeds the threshold (20%), the system transitions from bearing navigation to the close-range PPO phase. First end-to-end handoff achieved in test run.

---

## FEBRUARY 2026

**February — Week 1 (Feb 1–7)**
Designed the PPO environment (`ppo_environment.py`) as a Gymnasium-style custom environment. Defined observation space: flattened depth camera image + target direction angle.

**February — Week 2 (Feb 8–14)**
Defined the action space (continuous velocity commands: vx, vy, vz, yaw rate) and reward function. Initial reward components: distance reduction bonus, collision penalty, survival bonus.

**February — Week 3 (Feb 15–21)**
First PPO training runs with Stable-Baselines3. Observed early instability: agent was too aggressive and collided within a few steps. Started reward shaping iteration.

**February — Week 4 (Feb 22–28)**
Adjusted reward function weights. Added a per-step survival reward to encourage the agent to stay alive longer. Collision penalty increased. Training runs began showing improved episode lengths.

---

## MARCH 2026

**March — Week 1 (Mar 1–7)**
Continued PPO training and hyperparameter search. Tuned learning rate, clip range, and n_steps. Observed that shorter depth image resolution (downsized) sped up training without losing navigational performance.

**March — Week 2 (Mar 8–14)**
Implemented checkpoint saving in `train_ppo.py`. The model saves intermediate checkpoints every N timesteps so training can be resumed if interrupted. Reward rate adjustment commit finalized.

**March — Week 3 (Mar 15–21)**
PPO agent showed consistent obstacle avoidance behavior in open areas. Started testing inside the dense forest section. Identified that altitude descent near the fire target caused instability.

**March — Week 4 (Mar 22–31)**
Tuned DESCENT_FLOOR_Z = 65 parameter to prevent the drone from descending too aggressively. Stabilized the altitude control during the homing transition. Began work on the 360-degree scan module.

---

## APRIL 2026

**April — Week 1 (Apr 1–7)**
Implemented 360-degree scan logic: the UAV yaws through North, East, South, West, and Downward positions, capturing an RGB image at each angle. HSV fire pixel count is logged for each direction.

**April — Week 2 (Apr 8–14)**
Fire confirmation event finalized: if total fire pixels across scan images exceed the threshold, a FIRE CONFIRMED event is logged with distance and coordinates. Scan images saved to `output/mission/scan_*.png`.

**April — Week 3 (Apr 15–21)**
Built `map_generator.py`: seed-based random placement of tree actors and fire actor within the Unreal scene. Minimum tree-to-tree distance and bounds enforcement implemented. Supports repeatable experiment seeds.

**April — Week 4 (Apr 22–30)**
Wrote `test_map_generator.py`: unit-style tests verifying actor placement, seed repeatability, boundary compliance, and minimum distance between trees. All tests passed.

---

## MAY 2026

**May — Week 1 (May 1–7)**
Developed `dashboard_server.py`: a local HTTP server providing a mission status JSON API, a photo gallery API, and a Server-Sent Events (SSE) endpoint for real-time log streaming. Integrated with the mission output folder watcher.

**May — Week 2 (May 8–14)**
Developed the dashboard front-end: `dashboard.js`, `index.html`, `index.css`. Implemented state machine for mission phases, live console log display, mission timer, and a scrollable photo gallery. Dashboard displays Sector C map layout.

Implemented coordinate privacy masking: during the active surveillance phase, the mission console hides exact fire GPS coordinates in the logs. Coordinates are only revealed once the mission status is confirmed as COMPLETE.

**May — Week 3 (May 15–18)**
Ran full end-to-end mission recording. Recorded the complete pipeline: tower detection → 50,742 px alarm → 207.38° bearing → bearing navigation → 20.09% smoke handoff → PPO close-range → 4.3 m confirmation → 360-degree scan saved. Total mission duration: ~5 min 46 s.

Final report writing: compiled `CENG401_Detailed_Project_Report.md` with UML diagrams, integration validation table, literature review, and full project management section. Defense preparation completed.

---

## Summary

| Month | Primary Focus |
|---|---|
| October 2025 | Problem scoping, literature review, CENG400 proposal, dev environment setup |
| November 2025 | AirSim/Unreal integration, sensor validation, tower baseline capture |
| December 2025 | HSV masking, contour detection, bearing formula, initial UAV dispatch |
| January 2026 | Cross-track correction, altitude hold, smoke density monitoring, handoff logic |
| February 2026 | PPO environment design, observation/action/reward definition, first training runs |
| March 2026 | PPO hyperparameter tuning, checkpoint management, altitude stabilization |
| April 2026 | 360-degree scan, fire confirmation, map generator, test suite |
| May 2026 | Dashboard (server + UI + SSE), privacy masking, end-to-end mission run, final report |
