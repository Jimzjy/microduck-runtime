//! The calibration file a bench session leaves behind.
//!
//! One file holds everything the bench measured or decided about this particular body: where
//! each joint's zero is, which way it counts positive, how far it may travel, which way the IMU
//! board is mounted, and whether motion is allowed at all. `robotd` starts from it and Studio's
//! root helper writes it (`deploy/microduck-helper`, `calib-commit`), so its layout is a wire
//! format between two programs and not a private convenience: [`Calibration::render`] writes
//! exactly what the helper writes, and [`Calibration::parse`] reads exactly that back.
//!
//! Three of these fields are gates rather than data:
//!
//! - `motion_enabled` — false means the body has never been through commissioning, so the wheels
//!   may not drive: a policy output sent to uncalibrated joints is not a pose, it is a guess at
//!   one, and the duck falls over.
//! - `imu_mount_verified` — false means nobody has looked at which way the board is screwed on.
//!   The quaternion is still honoured (the default is what the shipped file carries), but the
//!   state stream says it is unverified so the interfaces can ask for the check.
//! - per-joint `calibrated` — false means the zero was never measured; motion stays off until
//!   every joint says true.
//!
//! Nothing here touches hardware. The file is read once at startup, and the mapping it describes
//! is handed to the bus as plain numbers; see `duck_control::bus_feetech::FeetechConfig`.

use std::collections::BTreeMap;
use std::path::Path;

use duck_ipc_proto::{JOINT_IDS, JOINT_NAMES};
use serde::{Deserialize, Serialize};

/// The registers a turn is worth, and the count the shipped file centres every joint on. Both
/// are properties of the STS protocol, not of this file; they are here because `zero_raw` is
/// expressed in those counts and a reader has to know what they mean.
pub const COUNTS_PER_TURN: f64 = 4096.0;
pub const FACTORY_ZERO_RAW: u16 = 2048;

/// Where the helper puts the file on a duck. The default the params file points at, so a bench
/// session and a daemon agree without either naming the path twice.
pub const DEFAULT_PATH: &str = "/etc/robot/feetech-ft5/hd1910-calibration.toml";

/// The IMU chips whose mount can be recorded beside the bus board's.
pub const IMU_CHIPS: [&str; 2] = ["bno08x", "lsm6dsv16x"];

/// The mount the shipped file carries, and the one `SflpDecoder` starts with: +90° about Y.
pub const DEFAULT_BUS_QUAT: [f64; 4] = [
    std::f64::consts::FRAC_1_SQRT_2,
    0.0,
    std::f64::consts::FRAC_1_SQRT_2,
    0.0,
];

/// One joint's line in the file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Joint {
    /// One of [`JOINT_NAMES`]. The file is looked up by name, never by position, so a file
    /// written for one joint order still applies to another.
    pub name: String,
    /// The servo ID this joint answers on. Checked against [`JOINT_IDS`] — a file that says the
    /// left knee is 14 is a file about a different robot.
    pub id: u8,
    /// The register count that means 0 rad for this joint.
    pub zero_raw: u16,
    /// Which way is positive: 1 or -1.
    pub direction: f64,
    /// Whether the zero was actually measured. Motion waits for all fifteen.
    pub calibrated: bool,
    /// Travel limits in radians, if the file has them. A limit is enforced when a target is
    /// written, so a policy cannot ask a servo past what the bench allowed.
    #[serde(default)]
    pub min_rad: Option<f64>,
    #[serde(default)]
    pub max_rad: Option<f64>,
}

impl Joint {
    /// The limits as a pair, if both ends are present and sane.
    pub fn limits(&self) -> Option<(f64, f64)> {
        match (self.min_rad, self.max_rad) {
            (Some(lo), Some(hi)) if lo.is_finite() && hi.is_finite() && lo <= hi => Some((lo, hi)),
            _ => None,
        }
    }
}

/// A secondary IMU's mount, as the helper records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImuMount {
    pub quat: [f64; 4],
    pub verified: bool,
}

/// The whole file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub motion_enabled: bool,
    pub imu_mount_verified: bool,
    pub imu_mount_quat: [f64; 4],
    #[serde(default)]
    pub imu_bus_protocol: Option<String>,
    #[serde(default)]
    pub current_ma_per_count: f64,
    #[serde(default)]
    pub units_verified: bool,
    /// `[imu_mounts.bno08x]` and friends. Absent when those chips were never seen.
    #[serde(default)]
    pub imu_mounts: BTreeMap<String, ImuMount>,
    pub joints: Vec<Joint>,
}

/// Why a file was not usable. The text goes to a person: it names the joint or the field.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationError(pub String);

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CalibrationError {}

fn err<T>(what: impl Into<String>) -> Result<T, CalibrationError> {
    Err(CalibrationError(what.into()))
}

impl Calibration {
    /// Parse and validate a file's text.
    pub fn parse(text: &str) -> Result<Self, CalibrationError> {
        let cal: Self = toml::from_str(text).map_err(|e| CalibrationError(format!("{e}")))?;
        cal.validate()?;
        Ok(cal)
    }

    /// Read and validate the file at `path`.
    pub fn load(path: &Path) -> Result<Self, CalibrationError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| CalibrationError(format!("{}: {e}", path.display())))?;
        Self::parse(&text)
    }

    /// What a body looks like before anyone has been near it: every joint on the factory zero,
    /// centred, no direction chosen, motion off. Used when the file is absent, so a fresh board
    /// can still report its own state instead of failing to start.
    pub fn factory() -> Self {
        let joints = JOINT_NAMES
            .iter()
            .zip(JOINT_IDS.iter())
            .map(|(name, &id)| Joint {
                name: (*name).to_owned(),
                id,
                zero_raw: FACTORY_ZERO_RAW,
                direction: -1.0,
                calibrated: false,
                min_rad: None,
                max_rad: None,
            })
            .collect();
        Self {
            motion_enabled: false,
            imu_mount_verified: false,
            imu_mount_quat: DEFAULT_BUS_QUAT,
            imu_bus_protocol: None,
            current_ma_per_count: 0.0,
            units_verified: false,
            imu_mounts: BTreeMap::new(),
            joints,
        }
    }

    /// The file's own checks. Everything `robotd` would otherwise discover while driving.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        check_quat(self.imu_mount_quat, "imu_mount_quat")?;
        for (chip, mount) in &self.imu_mounts {
            if !IMU_CHIPS.contains(&chip.as_str()) {
                return err(format!("unknown imu_mounts entry: {chip}"));
            }
            check_quat(mount.quat, &format!("imu_mounts.{chip}.quat"))?;
        }
        if let Some(protocol) = &self.imu_bus_protocol {
            if protocol != "feetech" && protocol != "dxl2" {
                return err(format!(
                    "imu_bus_protocol is {protocol:?}; only \"feetech\" or \"dxl2\" is one"
                ));
            }
        }
        if self.joints.len() != JOINT_NAMES.len() {
            return err(format!(
                "the file has {} joints, a duck has {}",
                self.joints.len(),
                JOINT_NAMES.len()
            ));
        }
        let mut seen: Vec<&str> = Vec::with_capacity(self.joints.len());
        for joint in &self.joints {
            let index = JOINT_NAMES
                .iter()
                .position(|name| *name == joint.name)
                .ok_or_else(|| {
                    CalibrationError(format!("unknown joint in the file: {}", joint.name))
                })?;
            if JOINT_IDS[index] != joint.id {
                return err(format!(
                    "{} is servo {}, not {}",
                    joint.name, JOINT_IDS[index], joint.id
                ));
            }
            if joint.direction != 1.0 && joint.direction != -1.0 {
                return err(format!(
                    "{} has direction {}; it is 1 or -1",
                    joint.name, joint.direction
                ));
            }
            if joint.zero_raw >= COUNTS_PER_TURN as u16 {
                return err(format!(
                    "{} has zero_raw {}; a turn is 0..={}",
                    joint.name,
                    joint.zero_raw,
                    COUNTS_PER_TURN as u16 - 1
                ));
            }
            if let (Some(lo), Some(hi)) = (joint.min_rad, joint.max_rad) {
                if !lo.is_finite() || !hi.is_finite() || lo > hi {
                    return err(format!(
                        "{} has min_rad {lo} above max_rad {hi}",
                        joint.name
                    ));
                }
            }
            seen.push(&joint.name);
        }
        for name in JOINT_NAMES {
            if !seen.contains(&name) {
                return err(format!("the file is missing {name}"));
            }
        }
        Ok(())
    }

    /// The joint named `name`, if the file has it. Files are keyed by name on purpose.
    pub fn joint(&self, name: &str) -> Option<&Joint> {
        self.joints.iter().find(|joint| joint.name == name)
    }

    /// The joints in [`JOINT_NAMES`] order, which is the order the bus and the state stream use.
    /// Only a validated file can be turned into this, so the array cannot have holes.
    pub fn joints_in_model_order(&self) -> [&Joint; JOINT_NAMES.len()] {
        let mut out: Vec<&Joint> = Vec::with_capacity(JOINT_NAMES.len());
        for name in JOINT_NAMES {
            // validate() guarantees a hit; a miss here would be a file mutated after validation,
            // and a panic is the honest outcome — driving on a guessed zero is worse.
            out.push(
                self.joint(name)
                    .unwrap_or_else(|| panic!("validated calibration lost {name}")),
            );
        }
        match out.try_into() {
            Ok(array) => array,
            Err(_) => unreachable!("one entry per name was pushed"),
        }
    }

    /// Where each joint's zero is, in [`JOINT_NAMES`] order.
    pub fn zeros(&self) -> [f64; JOINT_NAMES.len()] {
        self.joints_in_model_order()
            .map(|joint| f64::from(joint.zero_raw))
    }

    /// Which way each joint counts positive, in [`JOINT_NAMES`] order.
    pub fn directions(&self) -> [f64; JOINT_NAMES.len()] {
        self.joints_in_model_order().map(|joint| joint.direction)
    }

    /// Each joint's travel limits, in [`JOINT_NAMES`] order.
    pub fn limits(&self) -> [Option<(f64, f64)>; JOINT_NAMES.len()] {
        self.joints_in_model_order().map(Joint::limits)
    }

    /// Every joint's zero was measured.
    pub fn all_calibrated(&self) -> bool {
        self.joints.iter().all(|joint| joint.calibrated)
    }

    /// Enough of the file is filled in for the robot to walk: every zero measured, every limit
    /// measured, and the IMU mounting confirmed by a person.
    ///
    /// The three together are what the motion safety gate wants and what the calibration pages
    /// display as a finished robot. A file can be perfectly valid and still fail this —
    /// a factory file with fifteen zeros and no limits is exactly that, and saying "calibrated"
    /// about it is how a robot gets asked to walk on guessed numbers.
    pub fn is_complete(&self) -> bool {
        self.all_calibrated()
            && self.joints.iter().all(|joint| joint.limits().is_some())
            && self.imu_mount_verified
    }

    /// Record the IMU mounting a person confirmed, and mark it confirmed.
    ///
    /// Returns the previous value so a caller can log what changed. Does not validate: the
    /// caller has already been given [`Calibration::validate`]'s verdict on the stored document,
    /// and this only overwrites two fields of it — the shape cannot go bad.
    pub fn set_imu_mount(&mut self, quat: [f64; 4]) -> [f64; 4] {
        let previous = self.imu_mount_quat;
        self.imu_mount_quat = quat;
        self.imu_mount_verified = true;
        previous
    }

    /// The IMU mount to use, and which chip it came from. The bus board is the default: it is
    /// the one the shipped file describes and the one a duck has.
    pub fn bus_mount(&self) -> [f64; 4] {
        self.imu_mount_quat
    }

    /// The same file, written the way the helper writes it: top-level keys, then
    /// `[imu_mounts.*]`, then one `[[joints]]` table per joint in model order.
    ///
    /// The order matters to nobody reading TOML, but writing the helper's own layout keeps a
    /// diff between "what Studio wrote" and "what robotd echoed back" empty, which is what makes
    /// the read-back after `calib-commit` a real check rather than a formatting comparison.
    pub fn render(&self) -> String {
        let mut lines = vec![
            format!("motion_enabled = {}", bool_text(self.motion_enabled)),
            format!(
                "imu_mount_verified = {}",
                bool_text(self.imu_mount_verified)
            ),
            format!("imu_mount_quat = {}", quat_text(self.imu_mount_quat)),
            format!("current_ma_per_count = {}", num(self.current_ma_per_count)),
            format!("units_verified = {}", bool_text(self.units_verified)),
        ];
        if let Some(protocol) = &self.imu_bus_protocol {
            lines.push(format!("imu_bus_protocol = \"{protocol}\""));
        }
        for chip in IMU_CHIPS {
            let Some(mount) = self.imu_mounts.get(chip) else {
                continue;
            };
            lines.push(String::new());
            lines.push(format!("[imu_mounts.{chip}]"));
            lines.push(format!("quat = {}", quat_text(mount.quat)));
            lines.push(format!("verified = {}", bool_text(mount.verified)));
        }
        for joint in self.joints_in_model_order() {
            lines.push(String::new());
            lines.push("[[joints]]".to_owned());
            lines.push(format!("name = \"{}\"", joint.name));
            lines.push(format!("id = {}", joint.id));
            lines.push(format!("zero_raw = {}", joint.zero_raw));
            lines.push(format!("direction = {}", joint.direction as i8));
            lines.push(format!("calibrated = {}", bool_text(joint.calibrated)));
            if let Some(min) = joint.min_rad {
                lines.push(format!("min_rad = {}", num(min)));
            }
            if let Some(max) = joint.max_rad {
                lines.push(format!("max_rad = {}", num(max)));
            }
        }
        lines.join("\n") + "\n"
    }

    /// The file as the JSON `calibration-status` reports: the same keys Studio would have handed
    /// to the helper, so a page can show what is written without reading the file itself.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

fn check_quat(quat: [f64; 4], what: &str) -> Result<(), CalibrationError> {
    if !quat.iter().all(|x| x.is_finite()) {
        return err(format!("{what} is not four finite numbers"));
    }
    let norm2: f64 = quat.iter().map(|x| x * x).sum();
    if !(0.98..=1.02).contains(&norm2) {
        return err(format!(
            "{what} is not a unit quaternion (|q|² = {norm2:.6})"
        ));
    }
    Ok(())
}

fn bool_text(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

/// Shortest round-trip text that still reads as a float, which is what the helper's `repr` does.
fn num(value: f64) -> String {
    if value == 0.0 {
        return "0.0".to_owned();
    }
    let text = format!("{value}");
    if text.contains('.') || text.contains('e') || text.contains("inf") || text.contains("NaN") {
        text
    } else {
        format!("{text}.0")
    }
}

fn quat_text(quat: [f64; 4]) -> String {
    let inner: Vec<String> = quat.iter().map(|x| num(*x)).collect();
    format!("[{}]", inner.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn printed() -> String {
        Calibration::factory().render()
    }

    #[test]
    fn the_factory_file_names_every_joint_once_at_the_factory_zero() {
        let cal = Calibration::factory();
        cal.validate().expect("factory calibration validates");
        assert_eq!(cal.joints.len(), JOINT_NAMES.len());
        assert!(!cal.motion_enabled);
        assert!(!cal.all_calibrated());
        for (name, &id) in JOINT_NAMES.iter().zip(JOINT_IDS.iter()) {
            let joint = cal.joint(name).expect(name);
            assert_eq!(joint.id, id);
            assert_eq!(joint.zero_raw, FACTORY_ZERO_RAW);
            assert!(!joint.calibrated);
        }
        assert_eq!(cal.zeros(), [2048.0; JOINT_NAMES.len()]);
        assert_eq!(cal.directions(), [-1.0; JOINT_NAMES.len()]);
        assert_eq!(cal.limits(), [None; JOINT_NAMES.len()]);
    }

    #[test]
    fn a_rendered_file_reads_back_as_the_same_thing() {
        let mut cal = Calibration::factory();
        cal.motion_enabled = true;
        cal.imu_mount_verified = true;
        cal.imu_bus_protocol = Some("feetech".to_owned());
        cal.imu_mounts.insert(
            "bno08x".to_owned(),
            ImuMount {
                quat: [1.0, 0.0, 0.0, 0.0],
                verified: true,
            },
        );
        let first = cal.joints[1].clone();
        cal.joints[1] = Joint {
            zero_raw: 1993,
            direction: 1.0,
            calibrated: true,
            min_rad: Some(-0.436_332_312_998_582_4),
            max_rad: Some(0.523_598_775_598_298_8),
            ..first
        };
        let text = cal.render();
        let back = Calibration::parse(&text).expect("round trip");
        assert_eq!(back, cal);
        // And rendering what was parsed is byte-identical, which is what the helper's read-back
        // comparison leans on.
        assert_eq!(back.render(), text);
    }

    #[test]
    fn a_file_that_describes_a_different_robot_is_refused() {
        let mut cal = Calibration::factory();
        cal.joints[0].id = 34;
        let e = cal.validate().expect_err("wrong id");
        assert!(e.0.contains("left_hip_yaw"), "{}", e.0);
    }

    #[test]
    fn a_missing_joint_is_named() {
        let mut cal = Calibration::factory();
        cal.joints.pop();
        let e = cal.validate().expect_err("14 joints");
        assert!(e.0.contains("15"), "{}", e.0);
    }

    #[test]
    fn a_direction_that_is_not_a_sign_is_refused() {
        let mut cal = Calibration::factory();
        cal.joints[3].direction = 0.0;
        let e = cal.validate().expect_err("zero direction");
        assert!(e.0.contains("direction"), "{}", e.0);
    }

    #[test]
    fn an_upside_down_limit_pair_is_refused() {
        let mut cal = Calibration::factory();
        cal.joints[2].min_rad = Some(1.0);
        cal.joints[2].max_rad = Some(-1.0);
        let e = cal.validate().expect_err("min above max");
        assert!(e.0.contains("min_rad"), "{}", e.0);
    }

    #[test]
    fn a_mount_that_is_not_a_rotation_is_refused() {
        let mut cal = Calibration::factory();
        cal.imu_mount_quat = [0.0, 0.0, 0.0, 0.0];
        let e = cal.validate().expect_err("zero quaternion");
        assert!(e.0.contains("unit quaternion"), "{}", e.0);
    }

    #[test]
    fn an_unknown_imu_chip_is_refused_rather_than_ignored() {
        let mut cal = Calibration::factory();
        cal.imu_mounts.insert(
            "icm42688".to_owned(),
            ImuMount {
                quat: [1.0, 0.0, 0.0, 0.0],
                verified: false,
            },
        );
        let e = cal.validate().expect_err("unknown chip");
        assert!(e.0.contains("icm42688"), "{}", e.0);
    }

    #[test]
    fn a_bare_file_with_defaults_parses() {
        // The shipped image's file, minified to the keys that must be there.
        let text = r#"
motion_enabled = false
imu_mount_verified = false
imu_mount_quat = [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]
current_ma_per_count = 0.0
units_verified = false
"#
        .to_owned()
            + &Calibration::factory()
            .joints
            .iter()
            .map(|joint| {
                format!(
                    "\n[[joints]]\nname = \"{}\"\nid = {}\nzero_raw = 2048\ndirection = -1\ncalibrated = false\n",
                    joint.name, joint.id
                )
            })
            .collect::<String>();
        let cal = Calibration::parse(&text).expect("shipped shape");
        assert_eq!(cal.joints.len(), JOINT_NAMES.len());
        assert_eq!(cal.imu_mount_quat, DEFAULT_BUS_QUAT);
        assert!(cal.imu_mounts.is_empty());
    }

    #[test]
    fn the_rendered_text_is_the_helper_layout() {
        let text = printed();
        assert!(text.starts_with("motion_enabled = false\n"), "{text}");
        assert!(text.contains("\n\n[[joints]]\nname = \"left_hip_yaw\"\n"));
        assert!(text.contains("direction = -1\n"));
        assert!(text.ends_with("\n"));
    }
}
