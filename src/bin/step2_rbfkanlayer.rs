use candle::Tensor;
use rust_rbf_kan::rbfkan::rbfkanlayer::RbfKanLayer;

fn main() {
    let device = &candle::Device::Cpu;
    let num_rbf_center: usize = 5;
    let min_center: f32 = 0.0;
    let max_center: f32 = 1.0;
    let weight_init_scale: f32 = 0.1;
    let input_dim: usize = 10;
    let output_dim: usize = 3;
    let batchsize = 3;

    let rbflayer = RbfKanLayer::new(
        device,
        num_rbf_center,
        min_center,
        max_center,
        weight_init_scale,
        input_dim,
        output_dim,
    );

    let x = Tensor::rand(0.0f32, 1.0, (batchsize, input_dim), &device).unwrap();
    let y = rbflayer.forward(&x);

    println!("x:{}", x);
    println!("y:{}", y);
}
