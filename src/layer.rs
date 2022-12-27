use rand::rngs::SmallRng;
use crate::neuron::Neuron;



pub struct Layer {
    pub neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(neurons: usize, data_n: usize, rng: &mut SmallRng) -> Self {
        let mut n: Vec<Neuron> = vec![];
        for _ in 0..neurons {
            n.push(Neuron::new(data_n, rng));
        }
        Layer {
            neurons: n,
        }
    }

    pub fn feed(&self, data: Vec<f32>) -> Vec<f32> {
        assert!(self.neurons[0].weights.len() == data.len());
        let mut sums: Vec<f32> = vec![];
        for neuron in self.neurons.iter() {
            sums.push(neuron.feed(data.clone()));
        }
        sums
    }

    pub fn get_derivative(self, data: Vec<f32>) -> f32 {
        0.0
    }
}
