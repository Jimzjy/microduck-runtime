//! The Feetech bus — STS/SMS-protocol servos (HD-1910) in place of the XL330s.
//!
//! The same shape as [`crate::bus`]: one `sync_read` per tick, one goal pass per tick, and
//! the same startup census, register check and motor-swap path. What differs is
//! the wire — Feetech's own framing (`FF FF id len instr … ~sum`), which `rustypot` carries as
//! its "protocol v1" together with the STS register table — and four facts about the servo
//! that change what the code around the wire has to do:
//!
//! - **One block holds everything.** Position, speed, load, voltage, temperature, status and
//!   current sit contiguously at 56–70. The tick's read already carries the supply voltage and
//!   the case temperature, so [`RobotIo::slow_sensors`] costs no transaction of its own here.
//! - **The gains live in EEPROM.** P/D/I are at 21–23, inside the EEPROM area. The servo's
//!   write lock (register 55) ships *closed*, and a write to an EEPROM address while it is
//!   closed is accepted but not persisted — which is exactly the RAM semantics the XL330's
//!   gains have, and what [`RobotIo::set_gain`] relies on. Nothing here opens the lock for a
//!   gain write, so a limp fall does not cost an EEPROM cycle.
//! - **The sign of a joint is not a given.** On the XL330 robot the horn orientation fixes the
//!   positive direction and it matches the model. With different servos in remodelled parts
//!   that has to be measured per joint, so it is a parameter: [`FeetechConfig::directions`].
//! - **The IMU board answers here too, at ID 200, and last.** The `imu_to_dxl` board's
//!   firmware carries the Feetech register table as well as Dynamixel Protocol 2, and serves
//!   the same 15-byte fast block at 56 — so the one sync read that collects the servos also
//!   collects orientation, no second transaction. It must be requested **after** the servos:
//!   measured on the bench, a non-servo block ahead of a servo makes the servo's answer run a
//!   byte early and the board's checksum then eats the next block's first byte. Whether the
//!   board is expected at all is [`FeetechConfig::imu_on_bus`].
//! - **A goal write re-powers the servo, and SYNC_WRITE is dropped.** Measured on the
//!   HD-1910: any WRITE of `goal_position` — unicast or REG_WRITE — sets `torque_enable`
//!   (register 40) back to 1 whatever it said before, and SYNC_WRITE (0x23) is ignored
//!   outright, goals included. So goals go out as one REG_WRITE per joint plus one broadcast
//!   ACTION ([`FeetechIo::write`]), and while the robot is limp no goal write is sent at all
//!   — a limp that wrote goals would be a limp that re-enables its own torque. A servo that
//!   still creeps back on under read-only traffic is caught by the watchdog in
//!   [`FeetechIo::read`].
//!
//! Register addresses and semantics are from Feetech's *磁编码 STS 内存表* v1.1 (2026-08-27) and
//! *SCS 通信协议* v1.0 (2026-06-09); `rustypot`'s `sts3215` table agrees with them.

use std::f64::consts::PI;
use std::io::Write as _;
use std::time::Duration;

use rustypot::servo::feetech::sts3215::Sts3215Controller;

use crate::bus::{StaleImuTracker, STALE_RUN_WARN};
use crate::imu::{ImuData, SflpDecoder, IMU_BLOCK_LEN};
use crate::io::{ImuStale, IoError, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::model::{
    BAUD_RATE, FACTORY_ID, FEETECH_BAUD_REGISTER, FEETECH_EXPECTED_MODE, IMU_DXL_ID, JOINT_IDS,
    JOINT_NAMES, NUM_JOINTS,
};

/// What `rustypot` returns. Its own alias is private, so it is spelled out here.
type RpResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Start of the feedback block: `present_position` (56, 2) · `present_speed` (58, 2) ·
/// `present_load` (60, 2) · `present_voltage` (62) · `present_temperature` (63) · async-write
/// flag (64) · `status` (65) · `moving` (66) · goal readback (67, 2) · `present_current` (69, 2).
const READ_ADDR: u8 = 56;
const READ_LEN: u8 = 15;

/// `goal_position`, two bytes, little-endian. Written raw rather than through rustypot's
/// `AnglePosition` so the per-joint direction and the 0–4095 clamp are applied in one place.
const GOAL_POSITION_ADDR: u8 = 42;

/// `torque_enable`, one byte. What a limp must keep an eye on — see the watchdog in
/// [`FeetechIo::read`].
const TORQUE_ENABLE_ADDR: u8 = 40;

/// REG_WRITE (0x04) and ACTION (0x05): how a goal reaches an HD-1910. Its firmware drops
/// SYNC_WRITE (0x23) outright — measured with correct packets and checksums, nothing happens
/// — so each tick's goals become one deferred register write per joint, latched together by
/// a single broadcast ACTION. Neither instruction is answered, which is why this path costs
/// a third of what fifteen acknowledged unicast writes would.
const REG_WRITE_INSTR: u8 = 0x04;
const ACTION_INSTR: u8 = 0x05;
const BROADCAST_ID: u8 = 0xFE;

/// Version block: firmware major/minor, the endianness flag, servo major/minor.
const VERSION_ADDR: u8 = 0;
const VERSION_LEN: u8 = 5;

/// Encoder steps per revolution, and the register value that means 0 rad.
const STEPS_PER_REV: f64 = 4096.0;
const RAW_MAX: i32 = 4095;

/// The middle of a turn: the zero a servo that has never been calibrated is read against.
///
/// The same 2048 the shipped calibration file centres every joint on, and the same constant
/// `robotd-params::calibration::FACTORY_ZERO_RAW` names — kept as a literal here so the bus does
/// not depend on the params crate for one number.
const FACTORY_ZERO_RAW: f64 = 2048.0;

/// `present_current` counts 6.5 mA each (STS table, register 69).
const MILLIAMPS_PER_COUNT: f64 = 6.5;

/// `present_voltage` counts 0.1 V each.
const VOLTS_PER_COUNT: f64 = 0.1;

/// Same bound as the Dynamixel bus, for the same reason: a missing device must cost a bounded
/// hiccup, not the serial driver's default.
const READ_TIMEOUT: Duration = Duration::from_millis(30);

/// The protocol document gives "about 800 ms" for the REBOOT instruction (0x08). The XL330
/// path waits 500 ms; that would ping a Feetech servo while it is still down and fail an
/// adoption that worked.
const REBOOT_SETTLE: Duration = Duration::from_millis(1100);

/// Pause after each EEPROM write, as on the other bus.
const EEPROM_SETTLE: Duration = Duration::from_millis(20);

/// `phase` (register 18) bit 2: speed registers count 0.0146 rpm — one encoder step per second —
/// when set, and 0.732 rpm — fifty steps per second — when clear.
const PHASE_SPEED_UNIT_FINE: u8 = 1 << 2;

/// What this backend needs to know that the bus cannot tell it.
#[derive(Debug, Clone, PartialEq)]
pub struct FeetechConfig {
    /// Feetech `P` per unit of the XL330 `position_p_gain` the rest of the daemon thinks in.
    ///
    /// The control loop's gains (200 running, 160 standing, 50 limp) are XL330 register
    /// values on a 0–16383 scale; the STS `P` is one byte. The factory HD-1910 value is 32, and
    /// 32 / 200 = 0.16 is the starting point — it makes "running gain" mean "factory stiffness".
    /// Calibrate against a step response on the bench; see `software/飞特适配架构.md` §3.3.
    pub p_scale: f64,
    /// +1 or −1 per joint, indexed as [`JOINT_NAMES`]: whether the servo's increasing count is
    /// the model's positive rotation. Measured on the robot, not derived.
    pub directions: [f64; NUM_JOINTS],
    /// The register count that means 0 rad for each joint, indexed as [`JOINT_NAMES`].
    ///
    /// 2048 is the middle of a turn and what a servo that has never been calibrated answers at
    /// whatever pose it happens to be in, so a robot with no calibration file still moves — it
    /// just does not know that its "zero" is the pose the bench wants. The calibration file is
    /// what makes this a measurement; `robotd-params::calibration` carries it.
    pub zeros: [f64; NUM_JOINTS],
    /// Travel allowed on each joint, in model radians, indexed as [`JOINT_NAMES`].
    ///
    /// Enforced when a target is written, not when a pose is computed: a policy may ask for
    /// anything, and what reaches the servo is clamped to what the bench allowed. `None` means
    /// the file left that joint unbounded and only the single-turn register clamp applies.
    pub limits: [Option<(f64, f64)>; NUM_JOINTS],
    /// Whether the IMU board answers on this bus, at [`IMU_DXL_ID`], after the servos.
    ///
    /// True for a robot with the board fitted; false makes the read one block shorter and
    /// leaves [`RobotIo::imu_ready`] false, which is what the daemon already does when it
    /// starts on a bus with no IMU. Not a compile-time choice: the same binary drives a robot
    /// with the board and one without.
    pub imu_on_bus: bool,
}

impl Default for FeetechConfig {
    fn default() -> Self {
        Self {
            p_scale: 0.16,
            directions: [1.0; NUM_JOINTS],
            zeros: [FACTORY_ZERO_RAW; NUM_JOINTS],
            limits: [None; NUM_JOINTS],
            imu_on_bus: true,
        }
    }
}

impl FeetechConfig {
    /// Build from the parameter file's plain values, refusing anything that is not ±1 — a
    /// direction of 0 would pin a joint at its zero, and 2 would double every angle.
    ///
    /// `zeros` and `limits` come from the calibration file when there is one, and are the
    /// factory centre and unbounded otherwise. Both are checked here rather than trusted: a zero
    /// outside a turn, or a limit pair upside down, would turn into motion that looks plausible
    /// until the duck falls over.
    pub fn new(
        p_scale: f64,
        directions: &[i8],
        zeros: [f64; NUM_JOINTS],
        limits: [Option<(f64, f64)>; NUM_JOINTS],
        imu_on_bus: bool,
    ) -> std::result::Result<Self, String> {
        if directions.len() != NUM_JOINTS {
            return Err(format!(
                "bus.feetech.directions has {} entries, expected {NUM_JOINTS}",
                directions.len()
            ));
        }
        if !(p_scale.is_finite() && p_scale > 0.0) {
            return Err(format!(
                "bus.feetech.p_scale must be positive, got {p_scale}"
            ));
        }
        let mut out = [1.0; NUM_JOINTS];
        for (i, &d) in directions.iter().enumerate() {
            out[i] = match d {
                1 => 1.0,
                -1 => -1.0,
                other => {
                    return Err(format!(
                        "bus.feetech.directions[{i}] ({}) must be 1 or -1, got {other}",
                        JOINT_NAMES[i]
                    ));
                }
            };
        }
        for (i, &zero) in zeros.iter().enumerate() {
            if !(zero.is_finite() && (0.0..=f64::from(RAW_MAX)).contains(&zero)) {
                return Err(format!(
                    "{} has zero_raw {zero}; a turn is 0..={RAW_MAX}",
                    JOINT_NAMES[i]
                ));
            }
        }
        for (i, limit) in limits.iter().enumerate() {
            let Some((lo, hi)) = limit else { continue };
            if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
                return Err(format!(
                    "{} has limits {lo}..{hi}; the low end must not be above the high end",
                    JOINT_NAMES[i]
                ));
            }
        }
        Ok(Self {
            p_scale,
            directions: out,
            zeros,
            limits,
            imu_on_bus,
        })
    }

    /// `rad`, clamped to what this joint is allowed to travel. The register clamp in
    /// [`rad_to_raw`] is the last resort; this is the bench's limit, and the one a policy
    /// should never be able to step over.
    fn clamp(&self, joint: usize, rad: f64) -> f64 {
        match self.limits[joint] {
            Some((lo, hi)) => rad.clamp(lo, hi),
            None => rad,
        }
    }
}

pub struct FeetechIo {
    controller: Sts3215Controller,
    config: FeetechConfig,
    /// Encoder steps per second for one count of a speed register — 1 or 50, from `phase`.
    /// Read at [`Self::check_registers`]; 1 until then, which is what the servos ship with.
    speed_steps_per_count: f64,
    /// Voltage and temperatures from the last tick's block, so `slow_sensors` is free.
    last_slow: Option<SlowSensors>,
    /// The IDs one sync read asks for, in the order the blocks come back: the servos in
    /// [`JOINT_IDS`] order, then the IMU board. The board must be last — see the module doc.
    ids: Vec<u8>,
    /// Orientation, from the board's SFLP table. The mount is a property of how the board is
    /// screwed to the body, so it arrives after `open`.
    imu: SflpDecoder,
    /// Counts reads in which the board answered without refreshing.
    stale_imu: StaleImuTracker,
    /// Whether the daemon last commanded torque on. Gates the goal pass — see
    /// [`FeetechIo::write`] — and arms the limp watchdog — see [`FeetechIo::read`].
    torque_on: bool,
    /// Tick counter for the limp watchdog.
    limp_ticks: u32,
    /// A second handle on the same port, for the goal writes. `rustypot` owns its serial port
    /// privately, and its protocol-1.0 vocabulary has no REG_WRITE instruction — so the
    /// per-joint REG_WRITE and the broadcast ACTION are written by hand through this handle.
    /// All bus access is sequential on one thread, so the two handles never talk at once.
    raw: Box<dyn serialport::SerialPort>,
}

impl FeetechIo {
    pub fn open(port: &str, config: FeetechConfig) -> Result<Self> {
        let ids = read_ids(config.imu_on_bus);
        // One port, two handles: `serialport` takes an exclusive flock on every open, so a
        // second open of the same device fails outright. The controller gets a `try_clone` —
        // a dup of the same open file description, which shares the lock instead of
        // fighting for it — and this struct keeps the original for the hand-built
        // REG_WRITE/ACTION goal packets. All bus access is sequential on one thread.
        let raw = open_raw_port(port, BAUD_RATE)?;
        let controller = open_controller(raw
            .try_clone()
            .map_err(|e| IoError::Bus(format!("clone the motor serial port: {e}")))?);
        Ok(Self {
            controller: controller?,
            config,
            speed_steps_per_count: 1.0,
            last_slow: None,
            ids,
            imu: SflpDecoder::new(SflpDecoder::DEFAULT_MOUNT),
            stale_imu: StaleImuTracker::default(),
            torque_on: false,
            limp_ticks: 0,
            raw,
        })
    }

    /// Which way the IMU board hangs off the bus, relative to the model's frame.
    ///
    /// The board reports the raw chip frame; the daemon hands in the mounting rotation once at
    /// startup ([`crate::imu::SflpDecoder::DEFAULT_MOUNT`] for a Zero 3W). Called whether or not
    /// the board is expected, so the mounting stays a one-line fact in one place.
    pub fn set_imu_mount(&mut self, mount: [f64; 4]) {
        self.imu = SflpDecoder::new(mount);
    }

    /// Does the IMU board answer a ping?
    ///
    /// `true` when the parameter file says there is no board: there is nothing to wait for, and
    /// startup must not stall on a device the robot does not have. [`RobotIo::imu_ready`] is
    /// what keeps the policy and fall detection off in that case, so the daemon comes up and
    /// reports, rather than waiting forever for a board that was never fitted.
    ///
    /// When a board *is* expected, the ping is what tells an absent board from one that has not
    /// been asked yet — the answer decides whether startup waits.
    pub fn imu_answers(&mut self) -> Result<bool> {
        if !self.config.imu_on_bus {
            return Ok(true);
        }
        self.controller
            .ping(IMU_DXL_ID)
            .map(|_| true)
            .map_err(|e| IoError::Bus(format!("ping imu board {IMU_DXL_ID}: {e}")))
    }

    /// Assert the registers the control loop depends on, and correct the one that is safe to.
    ///
    /// Returns how many were corrected. Only the baud rate is ever written: a servo at another
    /// speed would not have answered the census, so in practice this finds nothing, and it is
    /// here for the replacement path. The rest is reported and left alone on purpose —
    ///
    /// - `mode`: the HD-1910 ships in mode 4, "pure position PD", which is what the policy is
    ///   to be trained against. A servo in another mode still answers and still moves, just
    ///   differently; rewriting fifteen EEPROMs on a guess is worse than saying so.
    /// - `response_status_level`: must be 1. At 0 the servo acknowledges nothing but reads, and
    ///   every `rustypot` write then costs a 30 ms timeout and reports failure. That cannot be
    ///   worked around from here, so it is an error rather than a warning.
    ///
    /// Also reads `phase` once, which decides the unit of every speed register.
    pub fn check_registers(&mut self) -> Result<usize> {
        let mut fixed = 0;
        for &id in &JOINT_IDS {
            fixed += self.check_registers_of(id)?;
        }
        self.read_speed_unit()?;
        Ok(fixed)
    }

    fn check_registers_of(&mut self, id: u8) -> Result<usize> {
        let version = self
            .controller
            .read_raw_data(id, VERSION_ADDR, VERSION_LEN)
            .map_err(|e| IoError::Bus(format!("read version block on {id}: {e}")))?;
        if version.len() != VERSION_LEN as usize {
            return Err(IoError::ShortRead {
                what: "version block",
                expected: VERSION_LEN as usize,
                got: version.len(),
            });
        }
        // Register 2 is the endianness flag; 0 is little-endian, which every decode below
        // assumes. The potentiometer SCS family is big-endian and shares the instruction set.
        if version[2] != 0 {
            return Err(IoError::Bus(format!(
                "servo {id} reports big-endian registers (END = {}); this backend is for the \
                 magnetic-encoder STS/SMS family",
                version[2]
            )));
        }

        let level = first(
            self.controller.read_response_status_level(id),
            "response_status_level",
            id,
        )?;
        if level != 1 {
            return Err(IoError::Bus(format!(
                "servo {id} has response_status_level = {level}; it must be 1 or no write is \
                 ever acknowledged — set it with the Feetech debug tool"
            )));
        }

        let mode = first(self.controller.read_mode(id), "mode", id)?;
        if mode != FEETECH_EXPECTED_MODE {
            tracing::warn!(
                id,
                mode,
                want = FEETECH_EXPECTED_MODE,
                firmware = %format!("{}.{}", version[0], version[1]),
                "servo is not in the expected operating mode; left as it is"
            );
        }

        let mut fixed = 0;
        let baud = first(self.controller.read_baudrate(id), "baudrate", id)?;
        if baud != FEETECH_BAUD_REGISTER {
            tracing::warn!(
                id,
                got = baud,
                want = FEETECH_BAUD_REGISTER,
                "correcting motor baud rate register"
            );
            self.with_eeprom_unlocked(id, |c| c.write_baudrate(id, FEETECH_BAUD_REGISTER))?;
            fixed += 1;
        }
        Ok(fixed)
    }

    /// `phase` bit 2 decides whether a speed count is one step per second or fifty.
    fn read_speed_unit(&mut self) -> Result<()> {
        let id = JOINT_IDS[0];
        let phase = first(self.controller.read_phase(id), "phase", id)?;
        self.speed_steps_per_count = if phase & PHASE_SPEED_UNIT_FINE != 0 {
            1.0
        } else {
            50.0
        };
        tracing::info!(
            phase,
            steps_per_count = self.speed_steps_per_count,
            "servo speed unit"
        );
        Ok(())
    }

    /// Run `write` with the EEPROM lock open, and close it again whatever happened.
    fn with_eeprom_unlocked(
        &mut self,
        id: u8,
        write: impl FnOnce(&mut Sts3215Controller) -> RpResult<()>,
    ) -> Result<()> {
        self.controller
            .write_lock(id, false)
            .map_err(|e| IoError::Bus(format!("unlock eeprom on {id}: {e}")))?;
        let result = write(&mut self.controller);
        std::thread::sleep(EEPROM_SETTLE);
        let relock = self.controller.write_lock(id, true);
        result.map_err(|e| IoError::Bus(format!("eeprom write on {id}: {e}")))?;
        relock.map_err(|e| IoError::Bus(format!("relock eeprom on {id}: {e}")))
    }

    /// The expected servo IDs that do not answer a ping, in [`JOINT_IDS`] order.
    pub fn missing_servos(&mut self) -> Result<Vec<u8>> {
        let mut missing = Vec::new();
        for &id in &JOINT_IDS {
            let answered = self
                .controller
                .ping(id)
                .map_err(|e| IoError::Bus(format!("ping {id}: {e}")))?;
            if !answered {
                missing.push(id);
            }
        }
        Ok(missing)
    }

    /// Flash a factory-fresh servo so it takes the place of the one that is missing.
    ///
    /// A new Feetech servo answers as ID 1 and already at 1 Mbps, so unlike the XL330 path there
    /// is no second baud rate to look at: unlock, write the ID, lock, check, reboot.
    ///
    /// The reboot is kept, for a different reason than on the XL330. There it clears an alert;
    /// here it is the proof that the new ID reached EEPROM — an ID written with the lock still
    /// closed answers until the next power cycle and then is 1 again, and this is the moment to
    /// find that out rather than the next morning.
    pub fn adopt_replacement(&mut self, id: u8) -> Result<bool> {
        let name = JOINT_IDS
            .iter()
            .position(|&j| j == id)
            .map(|i| JOINT_NAMES[i])
            .ok_or_else(|| IoError::Bus(format!("{id} is not a joint id")))?;

        let fresh = self
            .controller
            .ping(FACTORY_ID)
            .map_err(|e| IoError::Bus(format!("ping factory id {FACTORY_ID}: {e}")))?;
        if !fresh {
            return Ok(false);
        }
        tracing::warn!(
            id,
            joint = name,
            "factory-fresh servo on the bus; flashing it as the missing joint"
        );

        // The lock is per servo and the servo changes address half way through: open it under
        // the old ID, close it under the new one.
        self.controller
            .write_lock(FACTORY_ID, false)
            .map_err(|e| IoError::Bus(format!("unlock eeprom on {FACTORY_ID}: {e}")))?;
        self.controller
            .write_id(FACTORY_ID, id)
            .map_err(|e| IoError::Bus(format!("write id {id} on {FACTORY_ID}: {e}")))?;
        std::thread::sleep(EEPROM_SETTLE);
        self.controller
            .write_lock(id, true)
            .map_err(|e| IoError::Bus(format!("relock eeprom on {id}: {e}")))?;

        let fixed = self.check_registers_of(id)?;

        RobotIo::reboot(self, id)?;
        std::thread::sleep(REBOOT_SETTLE);
        let back = self
            .controller
            .ping(id)
            .map_err(|e| IoError::Bus(format!("ping {id} after reboot: {e}")))?;
        if !back {
            return Err(IoError::Bus(format!(
                "servo {id} ({name}) was flashed but did not come back from its reboot — \
                 the id did not reach eeprom, or the firmware has no REBOOT instruction"
            )));
        }
        let status = first(self.controller.read_status(id), "status", id)?;
        if status != 0 {
            tracing::error!(
                id,
                joint = name,
                status,
                "replacement servo still reports a fault after its reboot"
            );
        }
        tracing::warn!(
            id,
            joint = name,
            registers_fixed = fixed,
            "replacement servo adopted"
        );
        Ok(true)
    }

    /// Present positions only, in model radians.
    pub fn present_positions(&mut self) -> Result<[f64; NUM_JOINTS]> {
        let blocks = self
            .controller
            .sync_read_raw_data(&JOINT_IDS, READ_ADDR, 2)
            .map_err(|e| IoError::Bus(format!("read present positions: {e}")))?;
        if blocks.len() != NUM_JOINTS {
            return Err(IoError::ShortRead {
                what: "present positions",
                expected: NUM_JOINTS,
                got: blocks.len(),
            });
        }
        let mut out = [0.0; NUM_JOINTS];
        for (joint, block) in blocks.iter().enumerate() {
            if block.len() != 2 {
                return Err(IoError::ShortRead {
                    what: "present position",
                    expected: 2,
                    got: block.len(),
                });
            }
            let raw = u16::from_le_bytes([block[0], block[1]]);
            out[joint] = raw_to_rad(
                raw,
                self.config.directions[joint],
                self.config.zeros[joint],
            );
        }
        Ok(out)
    }

    /// Torque on every servo. Every servo is written whatever the others said — see
    /// [`crate::bus::DynamixelIo::set_torque`] for the fall this guards against.
    pub fn set_torque(&mut self, on: bool) -> Result<()> {
        let mut failed = Vec::new();
        for &id in &JOINT_IDS {
            if let Err(e) = self.controller.write_torque_enable(id, on) {
                failed.push(format!("torque {on} on {id}: {e}"));
            }
        }
        // Recorded whether or not every servo answered: a servo that missed its torque-off
        // write is caught by the limp watchdog, and a missed torque-on shows up as a joint
        // that does not hold. Either way the gate below must follow what was asked for, not
        // what the bus managed.
        self.torque_on = on;
        if failed.is_empty() {
            Ok(())
        } else {
            Err(IoError::Bus(failed.join("; ")))
        }
    }

    /// Ramp every joint from where it is now to `target`, linearly. `robotd init` only.
    pub fn interpolate_to(
        &mut self,
        target: &[f64; NUM_JOINTS],
        duration: Duration,
        step: Duration,
    ) -> Result<()> {
        let start = self.present_positions()?;
        let steps = (duration.as_secs_f64() / step.as_secs_f64())
            .ceil()
            .max(1.0) as u32;
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let mut next = [0.0; NUM_JOINTS];
            for j in 0..NUM_JOINTS {
                next[j] = start[j] + (target[j] - start[j]) * t;
            }
            self.write(&JointTargets::new(next))?;
            std::thread::sleep(step);
        }
        Ok(())
    }
}

/// The serial port at `baud`, wrapped in a Feetech-protocol controller.
fn open_controller(serial: Box<dyn serialport::SerialPort>) -> Result<Sts3215Controller> {
    Ok(Sts3215Controller::new()
        .with_protocol_v1()
        .with_serial_port(serial))
}

/// A plain serial handle at `baud` — the port both bus halves use: `rustypot` wraps a
/// `try_clone` of it, and the hand-built REG_WRITE/ACTION goal packets go out through the
/// original.
fn open_raw_port(port: &str, baud: u32) -> Result<Box<dyn serialport::SerialPort>> {
    serialport::new(port, baud)
        .timeout(READ_TIMEOUT)
        .open()
        .map_err(|e| IoError::Port {
            path: port.to_owned(),
            source: std::io::Error::other(e),
        })
}

/// A protocol-1.0 instruction packet: `FF FF id len instr params ~sum`, where `len` counts
/// the instruction, the parameters and the checksum, and the checksum inverts the byte sum
/// from `id` onwards.
fn packet(id: u8, instruction: u8, params: &[u8]) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(params.len() + 8);
    pkt.extend_from_slice(&[0xFF, 0xFF, id, (params.len() + 2) as u8, instruction]);
    pkt.extend_from_slice(params);
    let sum = pkt[2..].iter().fold(0u8, |acc, b| acc.wrapping_add(*b));
    pkt.push(!sum);
    pkt
}

/// rustypot returns a `Vec` even for a single-id read; an empty one means the servo did not
/// answer, which must not be read as "register is fine".
fn first<T: Copy>(read: RpResult<Vec<T>>, name: &'static str, id: u8) -> Result<T> {
    read.map_err(|e| IoError::Bus(format!("read {name} on {id}: {e}")))?
        .first()
        .copied()
        .ok_or(IoError::ShortRead {
            what: "register read",
            expected: 1,
            got: 0,
        })
}

/// Sign-magnitude, bit 15 the sign: how the STS reports position, speed and current.
fn sign_magnitude(raw: u16) -> i32 {
    let magnitude = i32::from(raw & 0x7FFF);
    if raw & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Register count to model radians: `zero_raw` is 0 rad, and `direction` says which way is
/// positive.
///
/// The same mapping as the XL330's — 4096 steps a turn — so [`crate::model::DEFAULT_POSITION`]
/// means the same pose on both robots once each joint's zero has been calibrated to the
/// straight-leg pose. Before calibration `zero_raw` is the middle of the turn, which is what the
/// Dynamixel path assumes too.
fn raw_to_rad(raw: u16, direction: f64, zero_raw: f64) -> f64 {
    let centred = f64::from(sign_magnitude(raw)) - zero_raw;
    direction * centred * (2.0 * PI / STEPS_PER_REV)
}

/// Model radians to a register count, clamped to one turn. A target outside it would be read by
/// the servo as a multi-turn move.
fn rad_to_raw(rad: f64, direction: f64, zero_raw: f64) -> u16 {
    let raw = (direction * rad * (STEPS_PER_REV / (2.0 * PI)) + zero_raw).round();
    raw.clamp(0.0, f64::from(RAW_MAX)) as u16
}

/// XL330-scale gain to the STS `P` byte. Never 0: a zero P is a servo that does not hold, which
/// is not what any caller of `set_gain` means — limp is a low gain, torque off is `set_torque`.
fn scale_gain(kp: u16, p_scale: f64) -> u8 {
    (f64::from(kp) * p_scale).round().clamp(1.0, 254.0) as u8
}

/// One servo's feedback block, decoded.
struct Feedback {
    position: f64,
    velocity: f64,
    current_ma: f64,
    volts: f64,
    temp_c: f64,
}

fn decode_block(
    block: &[u8],
    direction: f64,
    zero_raw: f64,
    speed_steps_per_count: f64,
) -> Feedback {
    let position = u16::from_le_bytes([block[0], block[1]]);
    let speed = u16::from_le_bytes([block[2], block[3]]);
    let current = u16::from_le_bytes([block[13], block[14]]);
    Feedback {
        position: raw_to_rad(position, direction, zero_raw),
        velocity: direction
            * f64::from(sign_magnitude(speed))
            * speed_steps_per_count
            * (2.0 * PI / STEPS_PER_REV),
        current_ma: f64::from(sign_magnitude(current)).abs() * MILLIAMPS_PER_COUNT,
        volts: f64::from(block[6]) * VOLTS_PER_COUNT,
        temp_c: f64::from(block[7]),
    }
}

/// The IDs one sync read asks for: the servos in [`JOINT_IDS`] order, then the IMU board.
///
/// The board being last is a requirement of its firmware, not a preference — see the module
/// doc for what a non-servo block ahead of a servo does to the answer.
fn read_ids(imu_on_bus: bool) -> Vec<u8> {
    let mut ids = JOINT_IDS.to_vec();
    if imu_on_bus {
        ids.push(IMU_DXL_ID);
    }
    ids
}

impl RobotIo for FeetechIo {
    fn read(&mut self) -> Result<Sensors> {
        // Limp watchdog: with no goal WRITE ever sent while torque is off (see `write`), the
        // remaining way a joint re-powers itself is noise on the wire — under continuous
        // sync traffic a servo has been measured flipping its torque register back on, a few
        // seconds after a relax, with nothing in this daemon asking for it. Once a second,
        // read every joint's torque register and write it back off where it crept up: a
        // locked joint is exactly what `relax` exists to prevent, and the log line is the
        // evidence trail for whatever hardware investigation follows.
        self.limp_ticks = self.limp_ticks.wrapping_add(1);
        if !self.torque_on && self.limp_ticks.is_multiple_of(50) {
            match self.controller.sync_read_raw_data(&JOINT_IDS, TORQUE_ENABLE_ADDR, 1) {
                Ok(blocks) => {
                    let flipped: Vec<u8> = JOINT_IDS
                        .iter()
                        .zip(blocks)
                        .filter(|(_, block)| block.first().is_some_and(|&v| v != 0))
                        .map(|(id, _)| *id)
                        .collect();
                    if !flipped.is_empty() {
                        tracing::warn!(
                            ids = ?flipped,
                            "servos re-enabled torque on their own; writing it back off"
                        );
                        let _ = self.set_torque(false);
                    }
                }
                Err(e) => tracing::debug!(error = %e, "limp torque check skipped"),
            }
        }
        let blocks = self
            .controller
            .sync_read_raw_data(&self.ids, READ_ADDR, READ_LEN)
            .map_err(|e| IoError::Bus(format!("motor sync_read: {e}")))?;
        if blocks.len() != self.ids.len() {
            return Err(IoError::ShortRead {
                what: "sync_read blocks",
                expected: self.ids.len(),
                got: blocks.len(),
            });
        }

        let mut sensors = Sensors {
            imu: ImuData::default(),
            ..Sensors::default()
        };

        // The servo blocks come first, the IMU board last, in request order.
        let (servo_blocks, imu_block) = if self.config.imu_on_bus {
            let (servos, board) = blocks.split_at(NUM_JOINTS);
            (servos, Some(&board[0]))
        } else {
            (&blocks[..], None)
        };

        if let Some(block) = imu_block {
            // The board's fast block is 15 bytes — gyro, quaternion x/y/z, sample counter,
            // status, reserved — and the decoder wants the first 12, exactly the 12 a
            // Dynamixel read of the same board returns (w is derived from x/y/z either way).
            if block.len() < IMU_BLOCK_LEN {
                return Err(IoError::ShortRead {
                    what: "imu block",
                    expected: IMU_BLOCK_LEN,
                    got: block.len(),
                });
            }
            let mut raw = [0u8; IMU_BLOCK_LEN];
            raw.copy_from_slice(&block[..IMU_BLOCK_LEN]);
            // Say so, or the counters are numbers nobody ever reads — but only once the run is
            // long enough to mean something, and rate-limited past that because a board that has
            // stopped refreshing produces one of these every tick.
            let run = self.stale_imu.observe(&raw);
            if run == STALE_RUN_WARN || (run > STALE_RUN_WARN && run.is_multiple_of(500)) {
                tracing::warn!(
                    consecutive = run,
                    total = self.stale_imu.stale.total,
                    "imu board has returned the same sample {run} reads running — orientation is frozen"
                );
            }
            sensors.imu = self.imu.decode(&raw);
        }

        let mut temps_c = [0.0; NUM_JOINTS];
        let mut joint_volts = [0.0; NUM_JOINTS];
        let mut volts = Vec::with_capacity(NUM_JOINTS);
        for (joint, block) in servo_blocks.iter().enumerate() {
            if block.len() != READ_LEN as usize {
                return Err(IoError::ShortRead {
                    what: "motor block",
                    expected: READ_LEN as usize,
                    got: block.len(),
                });
            }
            let fb = decode_block(
                block,
                self.config.directions[joint],
                self.config.zeros[joint],
                self.speed_steps_per_count,
            );
            sensors.positions[joint] = fb.position;
            sensors.velocities[joint] = fb.velocity;
            sensors.currents_ma[joint] = fb.current_ma;
            temps_c[joint] = fb.temp_c;
            joint_volts[joint] = fb.volts;
            // A servo answering 0 V is answering nonsense; it must not drag the mean down as if
            // the pack were half flat. The per-joint array keeps the zero as "did not say".
            if fb.volts > 0.0 {
                volts.push(fb.volts);
            }
        }
        if !volts.is_empty() {
            self.last_slow = Some(SlowSensors {
                volts: volts.iter().sum::<f64>() / volts.len() as f64,
                temps_c,
                joint_volts,
            });
        }
        Ok(sensors)
    }

    fn write(&mut self, targets: &JointTargets) -> Result<()> {
        // An HD-1910 turns its own torque on for any goal WRITE — reg40 goes to 1 whatever it
        // said before (measured). So while the robot is limp this function must not touch the
        // goal register at all: the only writes it has to send are goal writes, and a goal
        // write is a torque write. `set_torque(true)` is what reopens this path.
        if !self.torque_on {
            return Ok(());
        }
        for (joint, &rad) in targets.positions.iter().enumerate() {
            // The bench's limit applies before the register clamp: what the file allows is a
            // fact about this body, and a policy asking past it is a policy that will fall.
            let raw = rad_to_raw(
                self.config.clamp(joint, rad),
                self.config.directions[joint],
                self.config.zeros[joint],
            )
            .to_le_bytes();
            let pkt = packet(
                JOINT_IDS[joint],
                REG_WRITE_INSTR,
                &[GOAL_POSITION_ADDR, raw[0], raw[1]],
            );
            self.raw.write_all(&pkt).map_err(|e| {
                IoError::Bus(format!("goal reg_write on {}: {e}", JOINT_IDS[joint]))
            })?;
        }
        // One broadcast ACTION latches the whole tick's register writes at once. Neither
        // instruction is answered, so this costs ~3 ms of bus time per tick where the old
        // per-joint unicast path would have cost fifteen round trips.
        self.raw
            .write_all(&packet(BROADCAST_ID, ACTION_INSTR, &[]))
            .map_err(|e| IoError::Bus(format!("goal ACTION broadcast: {e}")))
    }

    fn set_torque(&mut self, on: bool) -> Result<()> {
        FeetechIo::set_torque(self, on)
    }

    fn torque_confirmed(&self) -> Option<bool> {
        Some(self.torque_on)
    }

    fn reboot(&mut self, id: u8) -> Result<()> {
        // The protocol document asks for torque off before a REBOOT. Best effort: the servo
        // being rebooted is, by definition, one that may not be answering properly.
        if let Err(e) = self.controller.write_torque_enable(id, false) {
            tracing::warn!(id, error = %e, "could not switch torque off before the reboot");
        }
        // The servo sends no status packet for 0x08; rustypot reports that as `Ok(false)`, so
        // only a failure to send is an error. No waiting here — `Safety::reboot_motors` calls
        // this once per servo from inside the control tick.
        self.controller
            .reboot(id)
            .map(|_| ())
            .map_err(|e| IoError::Bus(format!("reboot {id}: {e}")))
    }

    fn set_gain(&mut self, kp: u16) -> Result<()> {
        // P only, and with the EEPROM lock left closed: the value takes effect and is not
        // persisted, so a reboot or a power cycle restores whatever the servo was tuned to.
        // D and I are deliberately not zeroed as they are on the XL330 — there the factory D
        // is an accident the prototype removed; here it is part of how the vendor tuned a
        // 320:1 gearbox not to ring.
        let p = scale_gain(kp, self.config.p_scale);
        for &id in &JOINT_IDS {
            self.controller
                .write_p_coefficient(id, p)
                .map_err(|e| IoError::Bus(format!("p_coefficient {p} (kp {kp}) on {id}: {e}")))?;
        }
        Ok(())
    }

    /// Supply voltage and case temperatures — from the last tick's block, no transaction.
    ///
    /// Before the first successful read there is nothing cached, so one read is made; that is
    /// the `robotd init` case, which asks before any tick has run.
    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        if self.last_slow.is_none() {
            self.read()?;
        }
        self.last_slow.ok_or(IoError::ShortRead {
            what: "input voltage",
            expected: NUM_JOINTS,
            got: 0,
        })
    }

    /// Whether the orientation in [`Sensors`] is a measurement.
    ///
    /// Two conditions, and both matter: the board has to be on the bus at all, and it has to
    /// have sent enough samples for the filter to have settled (the same 25 the Dynamixel bus
    /// waits for). Until then the policy and fall detection stay off — which is why a bus with
    /// no board does not walk.
    fn imu_ready(&self) -> bool {
        self.config.imu_on_bus && self.imu.ready()
    }

    /// Reads in which the board answered with a block identical to the one before.
    fn imu_stale(&self) -> ImuStale {
        self.stale_imu.stale
    }

    fn cached_slow(&self) -> Option<SlowSensors> {
        self.last_slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2048 is 0 rad and a quarter turn is 1024 counts — the XL330's mapping, which is what
    /// lets `DEFAULT_POSITION` be shared between the two robots.
    #[test]
    fn position_mapping_matches_the_dynamixel_bus() {
        assert_eq!(raw_to_rad(2048, 1.0, FACTORY_ZERO_RAW), 0.0);
        assert!((raw_to_rad(3072, 1.0, FACTORY_ZERO_RAW) - PI / 2.0).abs() < 1e-12);
        assert!((raw_to_rad(1024, 1.0, FACTORY_ZERO_RAW) + PI / 2.0).abs() < 1e-12);
        for raw in [0u16, 1, 1024, 2047, 2048, 2049, 3072, 4095] {
            assert_eq!(rad_to_raw(raw_to_rad(raw, 1.0, FACTORY_ZERO_RAW), 1.0, FACTORY_ZERO_RAW), raw);
            assert_eq!(rad_to_raw(raw_to_rad(raw, -1.0, FACTORY_ZERO_RAW), -1.0, FACTORY_ZERO_RAW), raw);
        }
    }

    /// A reversed joint mirrors about 2048 — it must not mirror about 0, which would send a
    /// knee asked for +10° to the far end of its travel.
    #[test]
    fn a_reversed_joint_mirrors_about_the_centre() {
        let ten_degrees = 10.0_f64.to_radians();
        let forward = rad_to_raw(ten_degrees, 1.0, FACTORY_ZERO_RAW);
        let reversed = rad_to_raw(ten_degrees, -1.0, FACTORY_ZERO_RAW);
        assert_eq!(i32::from(forward) - 2048, 2048 - i32::from(reversed));
        assert!((raw_to_rad(reversed, -1.0, FACTORY_ZERO_RAW) - ten_degrees).abs() < 2.0 * PI / STEPS_PER_REV);
    }

    /// Targets past one turn are clamped rather than wrapped: the servo would take a wrapped
    /// value as a request to go the long way round.
    #[test]
    fn targets_outside_one_turn_are_clamped() {
        assert_eq!(rad_to_raw(10.0, 1.0, FACTORY_ZERO_RAW), 4095);
        assert_eq!(rad_to_raw(-10.0, 1.0, FACTORY_ZERO_RAW), 0);
    }

    /// Bit 15 is a sign, not a value bit. Read as plain `u16`, a slow reverse speed of 5
    /// counts would arrive as 32773 — a joint doing thirty radians a second.
    #[test]
    fn bit_fifteen_is_the_sign() {
        assert_eq!(sign_magnitude(5), 5);
        assert_eq!(sign_magnitude(0x8005), -5);
        assert_eq!(sign_magnitude(0x8000), 0);
    }

    fn block(position: u16, speed: u16, volts: u8, temp: u8, current: u16) -> Vec<u8> {
        let mut b = vec![0u8; READ_LEN as usize];
        b[0..2].copy_from_slice(&position.to_le_bytes());
        b[2..4].copy_from_slice(&speed.to_le_bytes());
        b[6] = volts;
        b[7] = temp;
        b[13..15].copy_from_slice(&current.to_le_bytes());
        b
    }

    /// Every field of the block lands where the register table says, in SI units.
    #[test]
    fn a_feedback_block_decodes_into_si_units() {
        let fb = decode_block(&block(3072, 0x8000 | 100, 74, 36, 20), 1.0, FACTORY_ZERO_RAW, 1.0);
        assert!((fb.position - PI / 2.0).abs() < 1e-12);
        assert!((fb.velocity + 100.0 * 2.0 * PI / 4096.0).abs() < 1e-12);
        assert!((fb.current_ma - 130.0).abs() < 1e-9);
        assert!((fb.volts - 7.4).abs() < 1e-9);
        assert_eq!(fb.temp_c, 36.0);
    }

    /// Direction flips velocity along with position; one without the other is a joint whose
    /// observed velocity opposes its observed motion, which a policy reads as a disturbance.
    #[test]
    fn direction_flips_velocity_with_position() {
        let forward = decode_block(&block(2148, 100, 74, 30, 0), 1.0, FACTORY_ZERO_RAW, 1.0);
        let reversed = decode_block(&block(2148, 100, 74, 30, 0), -1.0, FACTORY_ZERO_RAW, 1.0);
        assert_eq!(forward.position, -reversed.position);
        assert_eq!(forward.velocity, -reversed.velocity);
    }

    /// With `phase` bit 2 clear a speed count is fifty steps a second, not one.
    #[test]
    fn the_coarse_speed_unit_is_fifty_times_the_fine_one() {
        let fine = decode_block(&block(2048, 3, 74, 30, 0), 1.0, FACTORY_ZERO_RAW, 1.0);
        let coarse = decode_block(&block(2048, 3, 74, 30, 0), 1.0, FACTORY_ZERO_RAW, 50.0);
        assert!((coarse.velocity - 50.0 * fine.velocity).abs() < 1e-12);
    }

    /// The daemon's three gains at the default scale: running 200 is the factory 32, and the
    /// limp gain is soft but not zero.
    #[test]
    fn gains_scale_onto_one_byte_and_never_reach_zero() {
        assert_eq!(scale_gain(200, 0.16), 32);
        assert_eq!(scale_gain(160, 0.16), 26);
        assert_eq!(scale_gain(50, 0.16), 8);
        assert_eq!(scale_gain(0, 0.16), 1);
        assert_eq!(scale_gain(u16::MAX, 0.16), 254);
    }

    /// The middle of a turn is where an uncalibrated joint reads 0 rad; a calibrated one reads
    /// 0 rad at the count the bench measured, which is what makes `zero_raw` worth recording.
    #[test]
    fn a_calibrated_zero_moves_the_mapping() {
        let zero = 2000.0;
        assert_eq!(raw_to_rad(2000, 1.0, zero), 0.0);
        assert!((raw_to_rad(3024, 1.0, zero) - PI / 2.0).abs() < 1e-12);
        assert!((raw_to_rad(976, 1.0, zero) + PI / 2.0).abs() < 1e-12);
        // And it round-trips, whatever the zero.
        for raw in [0u16, 1, 976, 2000, 3024, 4095] {
            assert_eq!(rad_to_raw(raw_to_rad(raw, 1.0, zero), 1.0, zero), raw);
        }
    }

    #[test]
    fn config_refuses_directions_that_are_not_plus_or_minus_one() {
        let zeros = [FACTORY_ZERO_RAW; NUM_JOINTS];
        let limits = [None; NUM_JOINTS];
        let mut d = [1i8; NUM_JOINTS];
        assert!(FeetechConfig::new(0.16, &d, zeros, limits, true).is_ok());
        d[3] = -1;
        assert_eq!(
            FeetechConfig::new(0.16, &d, zeros, limits, true)
                .unwrap()
                .directions[3],
            -1.0
        );
        d[3] = 0;
        assert!(FeetechConfig::new(0.16, &d, zeros, limits, true).is_err());
        assert!(FeetechConfig::new(0.16, &d[..14], zeros, limits, true).is_err());
        assert!(FeetechConfig::new(0.0, &[1i8; NUM_JOINTS], zeros, limits, true).is_err());
        // The presence of the IMU board is a parameter, not a validation: both settings build.
        assert!(!FeetechConfig::new(0.16, &[1i8; NUM_JOINTS], zeros, limits, false)
            .unwrap()
            .imu_on_bus);
    }

    #[test]
    fn config_refuses_a_zero_outside_a_turn_or_an_upside_down_limit() {
        let mut zeros = [FACTORY_ZERO_RAW; NUM_JOINTS];
        let limits = [None; NUM_JOINTS];
        assert!(FeetechConfig::new(0.16, &[1i8; NUM_JOINTS], zeros, limits, true).is_ok());
        zeros[5] = 5000.0;
        let e = FeetechConfig::new(0.16, &[1i8; NUM_JOINTS], zeros, limits, true)
            .expect_err("zero past a turn");
        assert!(e.contains("zero_raw"), "{e}");
        zeros[5] = FACTORY_ZERO_RAW;
        let mut bad = [None; NUM_JOINTS];
        bad[7] = Some((1.0, -1.0));
        let e = FeetechConfig::new(0.16, &[1i8; NUM_JOINTS], zeros, bad, true)
            .expect_err("low above high");
        assert!(e.contains("limits"), "{e}");
    }

    /// The bench's limit is a fact about this body, so it applies before the register clamp:
    /// a policy asking for 1 rad on a joint allowed ±0.1 gets 0.1, not the end of the turn.
    #[test]
    fn the_benchs_limit_clamps_before_the_register_clamp() {
        let mut limits = [None; NUM_JOINTS];
        limits[0] = Some((-0.1, 0.1));
        let config = FeetechConfig::new(
            0.16,
            &[1i8; NUM_JOINTS],
            [FACTORY_ZERO_RAW; NUM_JOINTS],
            limits,
            true,
        )
        .expect("valid");
        assert_eq!(config.clamp(0, 1.0), 0.1);
        assert_eq!(config.clamp(0, -1.0), -0.1);
        assert_eq!(config.clamp(0, 0.05), 0.05);
        // An unbounded joint is untouched.
        assert_eq!(config.clamp(1, 3.0), 3.0);
    }

    /// The board answers under Feetech framing only when it is *last* in the request — a
    /// non-servo block ahead of a servo makes the servo's answer run a byte early (measured on
    /// the bench). So the ID list is servo order, then 200, and this test is the guard on it.
    #[test]
    fn the_imu_board_is_read_after_the_servos() {
        let with = read_ids(true);
        assert_eq!(with.len(), NUM_JOINTS + 1);
        assert_eq!(&with[..NUM_JOINTS], &JOINT_IDS[..]);
        assert_eq!(with[NUM_JOINTS], IMU_DXL_ID);

        let without = read_ids(false);
        assert_eq!(without.len(), NUM_JOINTS);
        assert_eq!(&without[..], &JOINT_IDS[..]);
        assert!(!without.contains(&IMU_DXL_ID));
    }

    /// The board's 15-byte fast block puts the SFLP quaternion (x/y/z, half precision, w
    /// derived) in bytes 6–11 — the same 12 bytes a Dynamixel read of the same board returns,
    /// which is why the decoder is shared rather than written twice.
    #[test]
    fn the_feetech_imu_block_decodes_as_the_shared_sflp_block() {
        let mut block = [0u8; IMU_BLOCK_LEN];
        // q = (0, 0, 1) as three half-precision values: 0.0, 0.0, 1.0 (0x3C00).
        block[6..12].copy_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x3C]);
        let mut imu = SflpDecoder::new(SflpDecoder::DEFAULT_MOUNT);
        assert!(!imu.ready(), "one sample is not a settled orientation");
        let mut data = ImuData::default();
        for _ in 0..25 {
            data = imu.decode(&block);
        }
        assert!(imu.ready());
        assert!(data.quat.iter().all(|v| v.is_finite()));
        let norm = data.quat.iter().map(|v| v * v).sum::<f64>();
        assert!((norm - 1.0).abs() < 1e-9, "quaternion must stay unit length");
    }

    /// A board that answers without refreshing is a frozen orientation, not a quiet robot, and
    /// it is the one fault nothing else reports. The tracker is the same one the Dynamixel bus
    /// uses, so this only checks that the Feetech path feeds it.
    #[test]
    fn a_repeating_block_is_counted_as_stale() {
        let mut tracker = StaleImuTracker::default();
        let block = [7u8; IMU_BLOCK_LEN];
        assert_eq!(tracker.observe(&block), 0, "the first block cannot be stale");
        assert_eq!(tracker.observe(&block), 1);
        assert_eq!(tracker.observe(&block), 2);
        let mut other = block;
        other[0] = 8;
        assert_eq!(tracker.observe(&other), 0, "a fresh block ends the run");
        assert_eq!(tracker.stale.total, 2, "the total never resets");
        assert_eq!(tracker.stale.run, 0);
    }
}
