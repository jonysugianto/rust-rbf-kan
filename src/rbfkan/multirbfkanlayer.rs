use super::rbfkanlayer::RbfKanLayer;
use candle::{Device, Tensor};

pub struct MultiRbfKanLayer {
    pub layers: Vec<RbfKanLayer>,
}

impl MultiRbfKanLayer {
    pub fn new(
        device: &Device,
        num_rbf_center: usize,
        input_min_center: f32,
        input_max_center: f32,
        layers_dims: &Vec<usize>,
        weight_init_scale: f32,
    ) -> Self {
        let mut layers: Vec<RbfKanLayer> = Vec::new();
        let layers_size = layers_dims.len() - 1;

        let firstlayer = RbfKanLayer::new(
            device,
            num_rbf_center,
            input_min_center,
            input_max_center,
            weight_init_scale,
            layers_dims[0].clone(),
            layers_dims[1].clone(),
        );

        layers.push(firstlayer);

        for i in 1..layers_size {
            let input_dim = &layers_dims[i];
            let output_dim = &layers_dims[i + 1];

            let l = RbfKanLayer::new(
                device,
                num_rbf_center,
                -1.0,
                1.0,
                weight_init_scale,
                input_dim.clone(),
                output_dim.clone(),
            );
            layers.push(l);
        }

        let ret = Self { layers: layers };

        return ret;
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let size = self.layers.len();
        let l = &self.layers[0];
        let mut out = l.forward(x);

        for i in 1..size {
            let l = &self.layers[i];
            out = l.forward(&out);
        }

        return out;
    }
}
