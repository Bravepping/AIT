use serde::Serialize;

#[derive(Serialize)]
pub struct Calibration {
    gain_x: f64,
    gain_y: f64,
    invert_x: bool,
    invert_y: bool,
}

pub fn estimate(observed_x: f64, observed_y: f64) -> Result<Calibration, String> {
    let axis = |pixels: f64| -> Result<(f64, bool), String> {
        let gain = 10.0 / pixels.abs();
        if !(0.01..=10.0).contains(&gain) {
            return Err("请填写 +10 单轴测试对应的有符号位移，绝对值须在 1–1000 像素".into());
        }
        Ok((gain, pixels < 0.0))
    };
    let (gain_x, invert_x) = axis(observed_x)?;
    let (gain_y, invert_y) = axis(observed_y)?;
    Ok(Calibration { gain_x, gain_y, invert_x, invert_y })
}
