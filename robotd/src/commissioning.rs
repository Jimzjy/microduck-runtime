//! The calibration bench: what `--commissioning` and `robot.busCommand` actually do.
//!
//! Two commands move or watch the robot, and both are answered only once the thing they promise
//! has happened — `hold-home` after the pose is reached and settled, `verify-imu` after a
//! nose-down gesture has been seen. So neither is a socket handler: both are state machines the
//! control loop advances one tick at a time, because the loop is the only owner of the wire and
//! the only place a target can be written.
//!
//! Time is accumulated from the `dt` the loop passes in, never read from a clock. That keeps the
//! whole thing deterministic under test, where a twenty-second gesture happens in twenty
//! microseconds of wall time.

use std::time::Duration;

use duck_control::{DEFAULT_POSITION, NUM_JOINTS};

/// How far a joint may end up from the home pose before `hold-home` calls it a failure.
///
/// Five degrees is not a tolerance this module invented: the bench refuses to sample a coarse
/// calibration on a pose it cannot trust to five degrees, so a hold that misses by more than
/// that is a hold that would silently poison the numbers it is about to measure.
pub const HOLD_ERROR_TOLERANCE_DEG: f64 = 5.0;

/// The ramp must exist for at least this long, however little the client asked for.
///
/// A zero-duration hold is a step change to fifteen servo goals, which is exactly the command a
/// robot jumps at.
pub const HOLD_MIN_MS: u32 = 100;

/// How long to keep holding after the ramp before answering, so the answer describes a settled
/// pose rather than the last frame of a move.
pub const HOLD_SETTLE: Duration = Duration::from_millis(400);

/// A robot counts as upright inside this.
pub const UPRIGHT_TOLERANCE_DEG: f64 = 5.0;

/// The nose-down window the bench asks for: far enough to be unambiguous, not so far that the
/// robot is falling over.
pub const NOSE_DOWN_MIN_DEG: f64 = 15.0;
pub const NOSE_DOWN_MAX_DEG: f64 = 30.0;

/// Body-frame `x` (forward) must have at least this much down-gravity in it for a tilt inside the
/// window to be nose-*down* rather than nose-up.
///
/// 0.2 is about 11.5°, so the window's lower edge clears it comfortably while a nose-up tip of
/// the same angle fails it outright — which is the mistake this check exists to catch.
pub const NOSE_DOWN_FORWARD: f64 = 0.2;

/// How long `verify-imu` watches for the gesture. Studio's own timeout is 35 s, so 20 s plus the
/// settle and the write leaves room to answer before it gives up.
pub const VERIFY_WINDOW: Duration = Duration::from_secs(20);

/// Grace period for the robot to be stood upright before `verify-imu` complains.
pub const UPRIGHT_GRACE: Duration = Duration::from_secs(3);

/// The ramp length a `hold-home` without an explicit duration gets.
pub const DEFAULT_HOLD_MS: u32 = 3_000;

/// Clamp a requested hold duration into something a servo will survive.
pub fn hold_duration(ms: Option<u32>) -> Duration {
    Duration::from_millis(u64::from(ms.unwrap_or(DEFAULT_HOLD_MS).max(HOLD_MIN_MS)))
}

/// Largest joint error, in degrees, between where the robot is and where it was told to be.
fn max_error_deg(positions: &[f64; NUM_JOINTS], target: &[f64; NUM_JOINTS]) -> f64 {
    let mut worst: f64 = 0.0;
    for (at, want) in positions.iter().zip(target.iter()) {
        if !at.is_finite() {
            // A joint that answered with nonsense is not a joint at its target.
            return f64::INFINITY;
        }
        worst = worst.max((at - want).abs());
    }
    worst.to_degrees()
}

/// `hold-home`: ramp fifteen joints to the home pose, then hold them there and say how well.
pub struct Hold {
    from: [f64; NUM_JOINTS],
    target: [f64; NUM_JOINTS],
    ramp: Duration,
    elapsed: Duration,
    settled_for: Duration,
    /// Worst error seen since the ramp finished. Not since the command arrived: during the ramp
    /// every joint is by definition far from the target, and reporting that would fail a hold
    /// that arrived perfectly.
    max_error_deg: f64,
    holding: bool,
}

impl Hold {
    /// Begin a hold from wherever the robot is now.
    pub fn begin(from: [f64; NUM_JOINTS], ramp: Duration) -> Self {
        Self {
            from,
            target: DEFAULT_POSITION,
            ramp,
            elapsed: Duration::ZERO,
            settled_for: Duration::ZERO,
            max_error_deg: 0.0,
            holding: false,
        }
    }

    /// Advance one tick and return the targets to command.
    pub fn step(&mut self, dt: Duration, positions: &[f64; NUM_JOINTS]) -> [f64; NUM_JOINTS] {
        self.elapsed += dt;
        if !self.holding && self.elapsed >= self.ramp {
            self.holding = true;
        }
        if self.holding {
            self.settled_for += dt;
            let error = max_error_deg(positions, &self.target);
            self.max_error_deg = self.max_error_deg.max(error);
            return self.target;
        }

        let progress = (self.elapsed.as_secs_f64() / self.ramp.as_secs_f64()).clamp(0.0, 1.0);
        let mut targets = self.from;
        for (at, goal) in targets.iter_mut().zip(self.target.iter()) {
            *at += (goal - *at) * progress;
        }
        targets
    }

    /// `ramping` or `holding`, in Studio's words.
    pub fn stage(&self) -> &'static str {
        if self.holding { "holding" } else { "ramping" }
    }

    /// Worst joint error since the ramp finished, in degrees. Meaningless while ramping.
    pub fn max_error_deg(&self) -> f64 {
        self.max_error_deg
    }

    /// Whether the answer is now worth sending: the ramp is done and the pose has settled.
    pub fn settled(&self) -> bool {
        self.holding && self.settled_for >= HOLD_SETTLE
    }
}

/// What `verify-imu`'s state machine has to say after a tick.
#[derive(Debug, PartialEq)]
pub enum VerifyStep {
    /// Nothing conclusive yet; keep watching.
    Waiting,
    /// The gesture was seen. The mount to store, normalised.
    Done([f64; 4]),
    /// The robot is not doing what the gesture needs. The text is the explanation, and Studio
    /// matches keywords in it to turn a failure into advice.
    Failed(String),
}

/// `verify-imu`: watch for the robot to be stood upright and then tipped nose-down.
///
/// The gesture is what confirms the mount: a robot whose IMU is mounted some other way reports a
/// different direction for the same tip, so seeing the tip arrive where the passed mount predicts
/// it is the whole test. The measurement the two agree on is not recomputed here — Studio reads
/// back exactly the quaternion it supplied, so storing that (normalised) is what makes the readback
/// agree to 1e-8 instead of to however many digits a fit would produce.
pub struct Verify {
    mount: [f64; 4],
    elapsed: Duration,
    seen_upright: bool,
    last_tilt_deg: f64,
}

impl Verify {
    /// Begin watching. `mount` is the quaternion the caller believes is in use.
    pub fn begin(mount: [f64; 4]) -> Self {
        Self {
            mount: normalize(mount),
            elapsed: Duration::ZERO,
            seen_upright: false,
            last_tilt_deg: 0.0,
        }
    }

    pub fn step(&mut self, dt: Duration, tilt_deg: f64, gravity: [f64; 3]) -> VerifyStep {
        self.elapsed += dt;
        self.last_tilt_deg = tilt_deg;

        if !self.seen_upright {
            if tilt_deg <= UPRIGHT_TOLERANCE_DEG {
                self.seen_upright = true;
            } else if self.elapsed >= UPRIGHT_GRACE {
                return VerifyStep::Failed(format!(
                    "the robot is {tilt_deg:.1}° from upright and has to start upright: stand the \
                     robot upright (within {UPRIGHT_TOLERANCE_DEG:.0}°) and try again. Nothing saved."
                ));
            }
            return VerifyStep::Waiting;
        }

        if (NOSE_DOWN_MIN_DEG..=NOSE_DOWN_MAX_DEG).contains(&tilt_deg) {
            if gravity[0] >= NOSE_DOWN_FORWARD {
                return VerifyStep::Done(self.mount);
            }
            // Same angle, wrong way — the one mistake that looks like success to a tilt-only
            // check, and the one that would store a mount 180° out.
            return VerifyStep::Failed(format!(
                "that is a tip of {tilt_deg:.1}° away from nose-down: tip the head forward, \
                 not backward, about the vertical axis. Nothing saved."
            ));
        }

        if self.elapsed >= VERIFY_WINDOW {
            return VerifyStep::Failed(format!(
                "no nose-down tip of {NOSE_DOWN_MIN_DEG:.0}–{NOSE_DOWN_MAX_DEG:.0}° within \
                 {:.0} s (the last reading was {:.1}° from upright). Nothing saved.",
                VERIFY_WINDOW.as_secs_f64(),
                self.last_tilt_deg
            ));
        }

        VerifyStep::Waiting
    }
}

/// A normalised copy of `q`, or `None` when it is not a rotation.
///
/// The check is a norm, not unit-ness: a caller may hand over a quaternion in any scaling, and
/// what makes it unusable is a length of zero or a NaN.
pub fn normalised(q: [f64; 4]) -> Option<[f64; 4]> {
    let norm = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-6 {
        return None;
    }
    let mut out = q;
    for c in out.iter_mut() {
        *c /= norm;
    }
    Some(out)
}

/// Normalise a quaternion, falling back to identity on anything that is not a rotation.
fn normalize(q: [f64; 4]) -> [f64; 4] {
    normalised(q).unwrap_or([1.0, 0.0, 0.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    const TICK: Duration = Duration::from_millis(20);

    fn away() -> [f64; NUM_JOINTS] {
        // Every joint a quarter turn from home: the ramp has real work to do.
        let mut p = DEFAULT_POSITION;
        for at in p.iter_mut() {
            *at += std::f64::consts::FRAC_PI_2;
        }
        p
    }

    #[test]
    fn hold_ramps_to_home_and_then_holds_it() {
        let mut hold = Hold::begin(away(), Duration::from_millis(300));
        let mut positions = away();
        // Halfway through the ramp.
        for _ in 0..7 {
            positions = hold.step(TICK, &positions);
        }
        assert_eq!(hold.stage(), "ramping");
        // Partway from where it started to home, having left neither.
        assert!(
            positions[0] < away()[0] && positions[0] > DEFAULT_POSITION[0],
            "{:?}",
            positions[0]
        );

        // The rest of the ramp, then long enough to settle.
        for _ in 0..30 {
            positions = hold.step(TICK, &positions);
        }
        assert_eq!(hold.stage(), "holding");
        assert_eq!(positions, DEFAULT_POSITION);
        assert!(hold.settled());
    }

    #[test]
    fn hold_measures_error_only_after_the_ramp() {
        let mut hold = Hold::begin(DEFAULT_POSITION, Duration::from_millis(100));
        // The robot never moves, so every joint stays a quarter turn out.
        let stuck = away();
        for _ in 0..10 {
            hold.step(TICK, &stuck);
        }
        // Ramps are excluded, so the error is the stalled pose, not the 90° it started at…
        assert!((hold.max_error_deg() - 90.0).abs() < 0.1, "{:?}", hold.max_error_deg());

        // …and a robot that arrived reports essentially nothing.
        let mut hold = Hold::begin(DEFAULT_POSITION, Duration::from_millis(100));
        for _ in 0..10 {
            hold.step(TICK, &DEFAULT_POSITION);
        }
        assert!(hold.max_error_deg() < 0.001);
    }

    #[test]
    fn hold_clamps_a_zero_duration_ramp() {
        assert_eq!(hold_duration(Some(0)), Duration::from_millis(100));
        assert_eq!(hold_duration(None), Duration::from_secs(3));
        assert_eq!(hold_duration(Some(500)), Duration::from_millis(500));
    }

    #[test]
    fn verify_refuses_to_start_on_a_robot_that_is_not_upright() {
        let mut verify = Verify::begin([1.0, 0.0, 0.0, 0.0]);
        let mut step = VerifyStep::Waiting;
        for _ in 0..200 {
            step = verify.step(TICK, 30.0, [0.5, 0.0, -0.86]);
            if matches!(step, VerifyStep::Failed(_)) {
                break;
            }
        }
        let VerifyStep::Failed(text) = step else {
            panic!("expected a failure, got {step:?}");
        };
        assert!(text.contains("from upright"), "{text}");
        assert!(text.contains("stand the robot upright"), "{text}");
        assert!(text.contains("Nothing saved"), "{text}");
    }

    #[test]
    fn verify_accepts_a_forward_tip() {
        let mut verify = Verify::begin([1.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            verify.step(TICK, 1.0, [0.0, 0.0, -1.0]),
            VerifyStep::Waiting
        );
        // 20° nose down: gravity gains forward-x.
        assert_eq!(
            verify.step(TICK, 20.0, [0.34, 0.0, -0.94]),
            VerifyStep::Done([1.0, 0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn verify_rejects_the_same_tip_backwards() {
        let mut verify = Verify::begin([1.0, 0.0, 0.0, 0.0]);
        verify.step(TICK, 0.0, [0.0, 0.0, -1.0]);
        let step = verify.step(TICK, 20.0, [-0.34, 0.0, -0.94]);
        let VerifyStep::Failed(text) = step else {
            panic!("expected a failure, got {step:?}");
        };
        assert!(text.contains("away from nose-down"), "{text}");
        assert!(text.contains("Nothing saved"), "{text}");
    }

    #[test]
    fn verify_gives_up_after_the_window() {
        let mut verify = Verify::begin([1.0, 0.0, 0.0, 0.0]);
        let mut step = VerifyStep::Waiting;
        // Five hundred ticks is ten seconds: still watching.
        for _ in 0..500 {
            step = verify.step(TICK, 0.0, [0.0, 0.0, -1.0]);
        }
        assert_eq!(step, VerifyStep::Waiting);
        for _ in 0..600 {
            step = verify.step(TICK, 0.0, [0.0, 0.0, -1.0]);
        }
        let VerifyStep::Failed(text) = step else {
            panic!("expected a failure, got {step:?}");
        };
        assert!(text.contains("no nose-down tip"), "{text}");
        assert!(text.contains("within 20 s"), "{text}");
    }

    #[test]
    fn verify_stores_the_mount_it_was_given_normalised() {
        let mut verify = Verify::begin([2.0, 0.0, 0.0, 0.0]);
        verify.step(TICK, 0.0, [0.0, 0.0, -1.0]);
        assert_eq!(
            verify.step(TICK, 20.0, [0.34, 0.0, -0.94]),
            VerifyStep::Done([1.0, 0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn a_nonsense_tilt_is_never_a_success() {
        let mut verify = Verify::begin([1.0, 0.0, 0.0, 0.0]);
        verify.step(TICK, 0.0, [0.0, 0.0, -1.0]);
        assert_eq!(verify.step(TICK, f64::NAN, [f64::NAN; 3]), VerifyStep::Waiting);
    }
}
