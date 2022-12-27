# VoiceDetect [WIP]

This is an implementation of ANN in pure Rust with no use of GPU as an exercise. It is meant to do binary classification and I hope to use it to distinguish human voice from other sounds in a given voice file.

The stuff has been divided into 3 files(excluding `main.rs`):

 - `neuron.rs`: This implements a single neuron with sigmoid activation function.
 - `layer.rs`: This implements a layer of neurons.
 - `mlp.rs`: This implements the entire network. Accepts input data size and a vector of number of neurons in each layer.

## License

This is free software under MIT/X11 license.
