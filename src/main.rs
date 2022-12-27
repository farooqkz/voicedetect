pub mod mlp;
pub mod layer;
pub mod neuron;


use rand::rngs::SmallRng;
use rand::prelude::*;

fn error(actual: Vec<f32>, predicted: Vec<f32>) -> f32 {
    assert!(actual.len() == predicted.len());
    let mut error_val: f32 = 0.0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        error_val += (a - p).exp2();
    }
    error_val
}


fn main() {
    let mut rng = SmallRng::from_entropy();
    const DATA_COUNT: usize = 9;
    let mut my_cool_mlp = mlp::MLP::new(vec![19, 10, 9, 6, 2], DATA_COUNT, &mut rng);
    let result: Vec<f32> = my_cool_mlp.feed(vec![0.5, -0.75, -0.22, -0.4, -0.5, -0.5, -0.999, -0.9, 0.0]);
    println!("{:?}", result);
}
