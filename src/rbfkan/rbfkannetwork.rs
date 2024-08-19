use candle::{Device, Module, Tensor};

use super::multirbfkanlayer::MultiRbfKanLayer;

pub struct RbfKanNetwork {
    pub hiddenlayers: MultiRbfKanLayer,
    pub head: Tensor,
    pub classification: bool,
}

impl RbfKanNetwork {
    pub fn new(
        device: &Device,
        num_rbf_center: usize,
        input_min_center: f32,
        input_max_center: f32,
        layers_dims: &Vec<usize>,
        weight_init_scale: f32,
        num_output: usize,
        classification: bool,
    ) -> Self {
        let hiddens = MultiRbfKanLayer::new(
            device,
            num_rbf_center,
            input_min_center,
            input_max_center,
            layers_dims,
            weight_init_scale,
        );

        let lod = &layers_dims[layers_dims.len() - 1];
        let head = Tensor::rand(
            -weight_init_scale,
            weight_init_scale,
            (lod.clone(), num_output),
            device,
        )
        .unwrap();

        let ret = RbfKanNetwork {
            hiddenlayers: hiddens,
            head: head,
            classification: classification,
        };

        return ret;
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.hiddenlayers.forward(x);
        let mut out = out.broadcast_matmul(&self.head).unwrap();
        if self.classification {
            out = candle_nn::ops::sigmoid(&out).unwrap();
        }
        return out;
    }
}
