use rand::rngs::SmallRng;
use rand::prelude::*;



fn sigmoid(x: f32) -> f32 {
    1f32 / (1f32 + (-x).exp())
}

#[derive(Debug)]
struct Neuron {
    b: f32,
    w: Vec<f32>,
}


struct Layer {
    neurons: Vec<Neuron>
}

impl Neuron {
    fn new(weights: usize, rng: &mut SmallRng)-> Self {
        let mut w: Vec<f32> = vec![];
        for _ in 0..weights {
            w.push(rng.gen_range(-1.0..1.0));
        }
        Neuron {
            b: rng.gen_range(-1.0..1.0),
            w: w,
        }
    }
}

impl Layer {
    fn new(neurons: usize, data_n: usize, rng: &mut SmallRng) -> Self {
        let mut n: Vec<Neuron> = vec![];
        for _ in 0..neurons {
            n.push(Neuron::new(data_n, rng));
        }
        Layer {
            neurons: n,
        }
    }

    fn feed(&self, data: Vec<f32>) -> Vec<f32> {
        assert!(self.neurons[0].w.len() == data.len());
        let mut sums: Vec<f32> = vec![];
        for neuron in self.neurons.iter() {
            let mut sum: f32 = 0.0;
            sum += neuron.b;
            for (w, d) in data.iter().zip(neuron.w.iter()) {
                sum += w*d;
            }
            sums.push(sum);
        }
        sums
    }
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn new(neurons_n: Vec<usize>, data_n: usize, rng: &mut SmallRng) -> Self {
        assert!(neurons_n.len() > 0);
        let mut mlp: MLP = MLP { layers: vec![] };
        let mut prev_n: usize = neurons_n[0];
        for n in neurons_n.iter() {
            let n = *n;
            if n == prev_n {
                mlp.layers.push(Layer::new(n, data_n, rng));
            } else {
                mlp.layers.push(Layer::new(n, prev_n, rng));
            }
            prev_n = n;
        }
        mlp
    }

    fn feed(self, data: Vec<f32>) -> Vec<f32> {
        let mut data: Vec<f32> = data.clone();
        let mut i = 0;
        for layer in self.layers.iter() {
            println!("Layer: {}", i);
            i += 1;
            data = layer.feed(data);
        }
        data
    }
}

fn main() {
    let mut rng = SmallRng::from_entropy();
    const DATA_COUNT: usize = 9;
    let mut my_cool_mlp = MLP::new(vec![3, 6, 7, 3], DATA_COUNT, &mut rng);
    println!("{:?}", my_cool_mlp.feed(vec![0.5, 0.75, 0.22, 0.4, 0.5, 0.5, 0.999, -0.9, 0.0]));
}
