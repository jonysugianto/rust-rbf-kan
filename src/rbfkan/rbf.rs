use candle::{Device, Tensor};
use candle_einops::einops;

// Radial Basis Function
pub struct Rbf {
    pub cs: Tensor, //center of the rbf function dimension:(1, NUM_RBF_CENTER)
    pub hs: Tensor, //parameter to control the width or spread the rbf dimension:(1, NUM_RBF_CENTER)
    pub ws: Tensor, //weights or adjustable coefficent of the rbf function dimension:(1, NUM_RBF_CENTER)
}

impl Rbf {
    pub fn new(
        device: &Device,
        num_rbf_center: usize,
        min_center: f32,
        max_center: f32,
        weight_init_scale: f32,
    ) -> Self {
        let hs = Tensor::ones((1, num_rbf_center), candle::DType::F32, device).unwrap();

        let ws = Tensor::rand(
            -weight_init_scale,
            weight_init_scale,
            (1, num_rbf_center),
            device,
        )
        .unwrap();
        let ws = ws.to_dtype(candle::DType::F32).unwrap();

        let cs = Tensor::rand(min_center, max_center, (1, num_rbf_center), device).unwrap();
        let cs = cs.to_dtype(candle::DType::F32).unwrap();

        let ret = Self {
            hs: hs,
            cs: cs,
            ws: ws,
        };
        return ret;
    }

    pub fn gaussian(&self, x: &Tensor) -> Tensor {
        let out = einops!("num_samples -> num_samples 1", x);
        let out = out.broadcast_sub(&self.cs).unwrap();
        let out = out.broadcast_div(&self.hs).unwrap();
        let out = out.neg().unwrap();
        let out = out.exp().unwrap();
        let out = out.broadcast_mul(&self.ws).unwrap();
        let out = einops!("num_samples sum(num_rbf_center)-> num_samples", &out);
        return out;
    }
}
