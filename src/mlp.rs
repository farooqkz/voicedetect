use rand::rngs::SmallRng;
use crate::layer::Layer;


pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(neurons_n: Vec<usize>, data_n: usize, rng: &mut SmallRng) -> Self {
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

    pub fn feed(self, data: Vec<f32>) -> Vec<f32> {
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


