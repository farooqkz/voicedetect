use rand::rngs::SmallRng;
use rand::prelude::*;

fn sigmoid(x: f32) -> f32 {
    1f32 / (1f32 + (-x).exp())
}

fn d_sigmoid(x: f32) -> f32 {
    let ex: f32 = (-x).exp();
    ex + (2f32*ex).exp2() + ex.exp2()*ex
}

#[derive(Debug)]
pub struct Neuron {
    pub bias: f32,
    pub weights: Vec<f32>,
}

impl Neuron {
    pub fn new(weights_n: usize, rng: &mut SmallRng)-> Self {
        Neuron {
            bias: rng.gen_range(-1.0..1.0),
            weights: (0..weights_n).map(|_| rng.gen_range(-1.0..1.0)).collect(),
        }
    }

    pub fn feed(&self, data: Vec<f32>) -> f32 {
        assert!(data.len() == self.weights.len());
        let mut sum: f32 = self.bias;
        for (d, w) in data.iter().zip(self.weights.iter()) {
            sum += d*w;
        }
        sigmoid(sum)
    }

    pub fn get_derivative(&self, data: Vec<f32>) -> f32 {
        let weights_sum: f32 = {
            let mut sum: f32 = 0.0;
            for weight in self.weights.iter() {
                sum += weight;
            }
            sum
        };
        let weighted_data: f32 = {
            let mut sum: f32 = self.bias;
            for (w, d) in self.weights.iter().zip(data.iter()) {
                sum += w*d;
            }
            sum
        };
        d_sigmoid(weighted_data) + weights_sum
    }
}
