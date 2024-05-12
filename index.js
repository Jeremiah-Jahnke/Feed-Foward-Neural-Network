const NeuralNetwork = require("./modules/NN-Class.js");

const nn = new NeuralNetwork([2, 1, 1], 0.001);

const trainingData = [
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] },
];

const epochs = 1000;
for (let epoch = 0; epoch < epochs; epoch++) {
  // console.log(`Epoch: ${epoch + 1}`);
  for (const data of trainingData) {
    const output = nn.feedForward(data.input);
    // console.log(`  Input: ${data.input}, Predicted Output: ${output[0]}`);

    const error = nn.calculateError(output, data.target);
    // console.log(`  Error: ${error}`);

    nn.backpropagate(error);

    // [DEBUGGING] Print weights and biases after backpropagation
    // console.log("  Weights after Backprop:");
    // console.log(nn.weights);
    // console.log("  Biases after Backprop:");
    // console.log(nn.biases);
  }
}

const testData = [
  { input: [0, 0] },
  { input: [0, 1] },
  { input: [1, 0] },
  { input: [1, 1] },
];

for (const data of testData) {
  const output = nn.feedForward(data.input);
  console.log(`Input: ${data.input}, Predicted Output: ${output[0]}`);
}
