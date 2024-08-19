use candle::Tensor;
use rust_rbf_kan::rbfkan::multirbfkanlayer::MultiRbfKanLayer;

fn main() {
    let device = &candle::Device::Cpu;
    let num_rbf_center: usize = 5;
    let input_min_center: f32 = 0.0;
    let input_max_center: f32 = 1.0;
    let weight_init_scale: f32 = 0.1;
    let batchsize = 3;
    let layer_dims: Vec<usize> = vec![10, 8, 6, 2];

    let multirbflayer = MultiRbfKanLayer::new(
        device,
        num_rbf_center,
        input_min_center,
        input_max_center,
        &layer_dims,
        weight_init_scale,
    );

    let x = Tensor::rand(0.0f32, 1.0, (batchsize, layer_dims[0].clone()), &device).unwrap();
    let y = multirbflayer.forward(&x);

    println!("x:{}", x);
    println!("y:{}", y);
}
