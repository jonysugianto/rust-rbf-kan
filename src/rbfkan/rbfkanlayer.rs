use candle::{Device, Tensor};
use candle_einops::einops;

pub struct RbfKanLayer {
    pub cs: Tensor, //center of the rbf function dimension:(1, NUM_RBF_CENTER, INPUT_DIM, OUTPUT_DIM)
    pub hs: Tensor, //parameter to control the width or spread the rbf dimension:(1, NUM_RBF_CENTER, INPUT_DIM, OUTPUT_DIM)
    pub ws: Tensor, //weights or adjustable coefficent of the rbf function dimension:(1, NUM_RBF_CENTER, INPUT_DIM, OUTPUT_DIM)
}

impl RbfKanLayer {
    pub fn new(
        device: &Device,
        num_rbf_center: usize,
        min_center: f32,
        max_center: f32,
        weight_init_scale: f32,
        input_dim: usize,
        output_dim: usize,
    ) -> Self {
        let hs = Tensor::ones(
            (1, num_rbf_center, input_dim, output_dim),
            candle::DType::F32,
            device,
        )
        .unwrap();

        let ws = Tensor::rand(
            -weight_init_scale,
            weight_init_scale,
            (1, num_rbf_center, input_dim, output_dim),
            device,
        )
        .unwrap();
        let ws = ws.to_dtype(candle::DType::F32).unwrap();

        let cs = Tensor::rand(
            min_center,
            max_center,
            (1, num_rbf_center, input_dim, output_dim),
            device,
        )
        .unwrap();
        let cs = cs.to_dtype(candle::DType::F32).unwrap();

        let ret = Self {
            hs: hs,
            cs: cs,
            ws: ws,
        };
        return ret;
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = einops!("bs id -> bs 1 id 1", x);
        let out = out.broadcast_sub(&self.cs).unwrap();
        let out = out.broadcast_div(&self.hs).unwrap();
        let out = out.neg().unwrap();
        let out = out.exp().unwrap();
        let out = out.broadcast_mul(&self.ws).unwrap();
        let out = einops!("bs sum(nc id) od-> bs od", &out);
        let out = out.tanh().unwrap(); //scale output to -1 to 1
        return out;
    }
}
