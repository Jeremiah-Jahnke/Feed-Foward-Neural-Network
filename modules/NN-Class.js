/**
 * Represents a feed-forward neural network.
 * @class
 */
class NeuralNetwork {
    /**
     * Represents a neural network.
     * @constructor
     * @param {Array<number>} layers - An array of numbers representing the number of neurons in each layer.
     * @param {number} [learningRate=0.1] - The learning rate of the neural network.
     */
    constructor(layers, learningRate = 0.1) {
        this.layers = layers;
        this.weights = [];
        this.biases = [];
        this.learningRate = learningRate;
        this.initWeightsAndBiases();
    }

    /**
     * Initializes the weights and biases for the neural network.
     */
    initWeightsAndBiases() {
        for (let i = 1; i < this.layers.length; i++) {
            const weights = new Array(this.layers[i]);
            const biases = new Array(this.layers[i]);

            for (let j = 0; j < this.layers[i]; j++) {
                weights[j] = new Array(this.layers[i - 1]).fill(0.01);
                biases[j] = 0;
            }

            this.weights.push(weights);
            this.biases.push(biases);
        }
    }

    /**
     * Applies the Rectified Linear Unit (ReLU) activation function to the given input.
     * @param {number} x - The input value.
     * @returns {number} The result of applying the ReLU function to the input.
     */
    relu(x) {
        // console.log("X: " + x + " Max: " + Math.max(0, x));
        return Math.max(0, x);
    }

    /**
     * Calculates the sigmoid of a given number.
     * @param {number} x - The input value.
     * @returns {number} The sigmoid of the input value.
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    dSigmoid(x) {
        const sigmoid = this.sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }

    /**
     * Activates the given input using the specified activation function.
     * @param {number} x - The input value to activate.
     * @param {string} [activation="sigmoid"] - The activation function to use. Defaults to "sigmoid".
     * @returns {number} The activated value.
     * @throws {Error} If an unsupported activation function is provided.
     */
    activate(x, activation = "sigmoid") {
        switch (activation) {
            case "sigmoid":
                return this.sigmoid(x);
            case "dSigmoid":
                return this.dSigmoid(x);
            case "relu":
                return this.relu(x);
            default:
                throw new Error("Unsupported activation function");
        }
    }

    /**
     * Performs the feed-forward operation of the neural network.
     * @param {Array} input - The input values for the neural network.
     * @returns {Array} - The output values after the feed-forward operation.
     */
    feedForward(input) {
        let activations = input;
        for (let i = 0; i < this.weights.length; i++) {
          activations = activations.map((value, index) => {
            const sum = this.weights[i].reduce((sum, weight, innerIndex) => sum + weight[innerIndex] * activations[innerIndex], 0);
            const output = this.activate(sum + this.biases[i][index]);
            // console.log("Layer", i + 1, "Neuron", index + 1, "Activation:", output);
            return output;
          });
        }
        return activations;
      }
      

    /**
     * Calculates the error between the output and target values.
     *
     * @param {number[]} output - The output values of the neural network.
     * @param {number[]} target - The target values for the neural network.
     * @returns {number[]} The error between the output and target values.
     */
    calculateError(output, target) {
        const error = output.map((value, index) => value - target[index]);
        // console.log("Error:", error);
        return error;
    }

    /**
     * Performs backpropagation to update the weights and biases of the neural network based on the given error.
     * @param {number[]} error - The error values for each output neuron.
     */
    backpropagate(error) {
        for (let i = this.weights.length - 1; i >= 0; i--) {
          const layerErrors = [];
          for (let j = 0; j < this.weights[i].length; j++) {
            let sum = 0;
            for (let k = 0; k < this.weights[i][j].length; k++) {
              sum += error[k] * this.weights[i][j][k];
            }
            layerErrors.push(sum);
      
            this.biases[i][j] -= this.learningRate * error[j];
      
            for (let k = 0; k < this.weights[i][j].length; k++) {
             //   console.log("Weight update:", "Layer:", i + 1, "Neuron:", j + 1, "Weight:", this.weights[i][j][k]);
              this.weights[i][j][k] -= this.learningRate * error[j] * (this.weights[i][j][k] > 0 ? 1 : 0); 
            }
          }
          error = layerErrors;
        }
      }

    /**
     * Trains the neural network by performing a feed forward pass, calculating the error, and backpropagating the error.
     * @param {Array} input - The input values for the neural network.
     * @param {Array} target - The target output values for the neural network.
     */
    train(input, target) {
        // console.log("Input:", input);
        const output = this.feedForward(input);
        const error = this.calculateError(output, target);
        this.backpropagate(error);
    }
}

module.exports = NeuralNetwork;