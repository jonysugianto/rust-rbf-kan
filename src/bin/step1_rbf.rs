use candle::{Device, Tensor};
use candle_einops::einops;
use rust_rbf_kan::rbfkan::rbf::Rbf;

fn main() {
    let device = &candle::Device::Cpu;
    let num_rbf_center: usize = 5;
    let min_center: f32 = 0.0;
    let max_center: f32 = 1.0;
    let weight_init_scale: f32 = 0.1;
    let grbf = Rbf::new(
        device,
        num_rbf_center,
        min_center,
        max_center,
        weight_init_scale,
    );

    let x = Tensor::rand(0.0f32, 1.0, (10), &device).unwrap();
    let y = grbf.gaussian(&x);

    println!("x:{}", x);
    println!("y:{}", y);
}
