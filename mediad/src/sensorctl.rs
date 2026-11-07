//! 和 vendor 的 IMX219 驱动说话：一个私有 ioctl 带着曝光和增益。
//!
//! **为什么不是标准 V4L2 控件。** 板上的 `imx219.ko` 把 `V4L2_CID_EXPOSURE` 和
//! `V4L2_CID_GAIN` 注册成了普通控件，但实测（2026-11-07，Zero 3W + imx219）写进去只改了控件框架
//! 的缓存：`--get-ctrl=gain` 能读回 409600，抓出来的帧和 1x 时逐像素一样（均值 68.58、p99 94，
//! 1x 到 512x 全部相同）。真正落到 `sensor_write()` 上的只有 `sensor_ioctl()` 里那条私有通路。
//!
//! **为什么不是 `v4l2-ctl --set-ctrl`。** 同一个原因：那个工具最后也是走标准控件。
//! 所以这里直接 `ioctl` 到传感器的 subdev 节点上。
//!
//! **结构体必须和内核侧一致。** `bsp/include/media/sunxi_camera_v2.h` 里
//! `struct sensor_exp_gain` 在非 `CONFIG_ARCH_SUN50IW10` 平台有八个 `int`（32 字节），
//! 而仓库里 `vin_test/sunxi_camera_v2.h` 那份只有四个（16 字节）。按后者算出来的 ioctl 号
//! 是错的，内核对不上就回 `EINVAL` —— 这就是第一次尝试失败的全部原因。
//!
//! 单位：曝光是行，驱动里 `exp_val >>= 4` 之后写进 0x015a/0x015b；增益 16 是 1x，
//! 驱动把它夹在 `[16, 159]` 之间（也就是 1x 到 9.94x），超出部分静默丢弃。

use std::os::unix::io::AsRawFd;

/// 内核的 `BASE_VIDIOC_PRIVATE`。
const BASE_VIDIOC_PRIVATE: u8 = 192;

/// 内核的 `_IOWR('V', BASE_VIDIOC_PRIVATE + 61, struct sensor_exp_gain)`。
const fn ioc(dir: u32, kind: u8, nr: u8, size: u32) -> u64 {
    ((dir as u64) << 30) | ((kind as u64) << 8) | (nr as u64) | ((size as u64) << 16)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SensorExpGain {
    exp_val: i32,
    exp_mid_val: i32,
    exp_short_val: i32,
    gain_val: i32,
    gain_mid_val: i32,
    gain_short_val: i32,
    r_gain: i32,
    b_gain: i32,
}

/// `_IOWR` 的方向位是 3（读加写）。
const VIDIOC_VIN_SENSOR_EXP_GAIN: u64 =
    ioc(3, b'V', BASE_VIDIOC_PRIVATE + 61, std::mem::size_of::<SensorExpGain>() as u32);

/// 驱动的增益单位：16 是 1x，上限 159。
const GAIN_UNITY: u32 = 16;
const GAIN_MAX: u32 = 159;

/// 写一次曝光和增益。`exposure_lines` 是行，`analogue_gain` 用传感器自己的单位
/// （[`crate::sensor::Exposure::unity_gain`] 是 1x）。
pub fn set_exp_gain(
    subdev: &str,
    exposure_lines: u32,
    analogue_gain: u32,
    unity_gain: u32,
) -> std::io::Result<()> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(subdev)?;

    let gain = (u64::from(analogue_gain) * u64::from(GAIN_UNITY) / u64::from(unity_gain.max(1)))
        .clamp(u64::from(GAIN_UNITY), u64::from(GAIN_MAX)) as i32;
    let exp = (exposure_lines << 4) as i32;

    let mut params = SensorExpGain {
        exp_val: exp,
        exp_mid_val: exp,
        exp_short_val: exp,
        gain_val: gain,
        gain_mid_val: gain,
        gain_short_val: gain,
        r_gain: 256,
        b_gain: 256,
    };

    // SAFETY: the pointer is to a local of the exact layout the kernel's `_IOWR` size encodes, and
    // it stays alive for the duration of the call.
    let rc = unsafe {
        libc::ioctl(
            file.as_raw_fd(),
            VIDIOC_VIN_SENSOR_EXP_GAIN as libc::c_ulong,
            &mut params as *mut SensorExpGain,
        )
    };
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ioctl_number_matches_the_kernel_header() {
        // 算出来的值必须是 0xc02056fd：这是 2026-11-07 在板上验证过的那个号（写成功、画面变亮），
        // 也是 `_IOWR('V', 253, 32 字节)` 的定义。
        assert_eq!(VIDIOC_VIN_SENSOR_EXP_GAIN, 0xc020_56fd);
    }

    #[test]
    fn the_gain_conversion_follows_the_sensor_units() {
        // unity_gain 256（imx219）：1024 是 4x，对应驱动里的 64。
        assert_eq!(gain_of(1024, 256), 64);
        // 1x 以下夹到 1x，9.94x 以上夹到上限。
        assert_eq!(gain_of(0, 256), 16);
        assert_eq!(gain_of(100_000, 256), 159);
    }

    fn gain_of(analogue_gain: u32, unity_gain: u32) -> i32 {
        (u64::from(analogue_gain) * u64::from(GAIN_UNITY) / u64::from(unity_gain.max(1)))
            .clamp(u64::from(GAIN_UNITY), u64::from(GAIN_MAX)) as i32
    }
}
