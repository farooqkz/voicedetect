use rand::rngs::SmallRng;
use crate::neuron::Neuron;

fn error(actual: f32, predicted: f32) -> f32 {
    (actual - predicted).exp2()
}

pub struct Layer {
    pub neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(neurons: usize, data_n: usize, rng: &mut SmallRng) -> Self {
        Layer {
            neurons: (0..neurons).map(Neuron::new(data_n, rng)).collect(),
        }
    }

    pub fn feed(&self, data: Vec<f32>) -> Vec<f32> {
        assert!(self.neurons[0].weights.len() == data.len());
        self.neurons.iter().map(|n| n.feed(data.clone())).collect()
    }

    pub fn update(&self, actual: Vec<f32>) {
        assert!(self.neurons[0].weights.len() == actual.len());
        for neuron in self.neurons.iter() {
            let predicted = neuron.feed(actual.clone());
            
        }
    }
}
