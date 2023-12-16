import {
  reLU,
  linear,
  sig,
  tanh,
  reLUDerivative,
  linearDerivative,
  sigDerivative,
  tanhDerivative,
} from "./ActivationFunctions.js";
import "./Perceptron.js";
import { Perceptron } from "./Perceptron.js";

export default class NeuralNetwork {
  #isLearning = false;
  #inputLayer = [];
  #hiddenLayers = [];
  #outputLayer = [];
  #data = null;
  #activationFunction = tanh;
  #activationFunctionDerivative = tanhDerivative;
  #outputExpectedValuesByClasses = {};

  constructor(data, activationFunctionName, isClassification) {
    this.#data = data;
    [this.#activationFunction, this.#activationFunctionDerivative] =
      this.#getActivationFunction(activationFunctionName);

    // Initialize input neurons
    const numOfInputNeurons = data.columnsCount - 1;
    for (let i = 0; i < numOfInputNeurons; i++) {
      const perceptron = new Perceptron(linear, linearDerivative);
      perceptron.initialize();
      this.#inputLayer.push(perceptron);
    }

    // Initialize 1-hidden layer with two neurons
    this.#hiddenLayers.push([
      new Perceptron(
        this.#activationFunction,
        this.#activationFunctionDerivative
      ).initialize(this.#inputLayer.length),
      new Perceptron(
        this.#activationFunction,
        this.#activationFunctionDerivative
      ).initialize(this.#inputLayer.length),
    ]);

    let numberOfOutputNeurons = 1;
    if (isClassification) {
      // We need to extract how many classes does the data has
      const outputs = data[data.columnsCount - 1];
      const classes = {};
      for (const c of outputs) {
        classes[c] = true;
      }

      numberOfOutputNeurons = Object.getOwnPropertyNames(classes).length;
      let classNum = 0;
      for (const c in classes) {
        this.#outputExpectedValuesByClasses[c] = new Array(
          numberOfOutputNeurons
        ).fill(0);
        this.#outputExpectedValuesByClasses[c][classNum++] = 1;
      }
    }

    const numberOfPreviousHiddenLayerNeurons =
      this.#hiddenLayers[this.#hiddenLayers.length - 1].length;
    for (let i = 0; i < numberOfOutputNeurons; i++) {
      const perceptron = new Perceptron(
        this.#activationFunction,
        this.#activationFunctionDerivative
      );
      perceptron.initialize(numberOfPreviousHiddenLayerNeurons);
      this.#outputLayer.push(perceptron);
    }
  }

  startLearning(learningRate, epochs, targetError) {
    this.#isLearning = true;
    const layers = this.#hiddenLayers.concat([[...this.#outputLayer]]);
    for (let epochNumber = 1; epochNumber <= epochs; epochNumber++) {
      let sse = 0;
      for (let i = 0; i < this.#data.rowsCount; i++) {
        // Activate
        let activatedValues = [];
        for (let x = 0; x < this.#data.columnsCount - 1; x++) {
          activatedValues.push({ value: parseFloat(this.#data[x][i]) });
        }

        activatedValues = [[...activatedValues]];
        for (let l = 0; l < layers.length; l++) {
          const layerInputs = activatedValues[l].map(
            (layerValues) => layerValues.value
          );
          activatedValues.push(
            layers[l].map((neuron) => neuron.activate(layerInputs))
          );
        }

        // Weight Training
        let errors = [];
        for (let l = layers.length - 1; l >= 0; l--) {
          const expectedOutput =
            this.#outputExpectedValuesByClasses[
              this.#data[this.#data.columnsCount - 1][i]
            ];
          let layerOutput = activatedValues.pop();

          if (l === layers.length - 1) {
            // Calculate output layer errors and consider them in the SSE calculation
            errors = expectedOutput.map(
              (value, index) => value - layerOutput[index].value
            );
            sse += errors.reduce((prev, curr) => prev + curr * curr, 0);
          }

          let sigmas = errors.map(
            (err, index) => err * layerOutput[index].derivedValue
          );
          let deltas = sigmas.map((sigma, index) => {
            return layers[l][index].inputWeights.map(
              (w) => w * sigma * learningRate
            );
          });
          errors = sigmas;

          // Update the weights
          deltas.forEach((delta, index) => {
            layers[l][index].inputWeights = delta.map(
              (d, dIndex) => layers[l][index].inputWeights[dIndex] + d
            );
            layers[l][index].theta +=
              layers[l][index].theta *
              layers[l][index].thetaSign *
              sigmas[index];
          });
        }
      }

      // Check sse and mse
      if (sse <= targetError) {
        // Achieved target error; end learning
        break;
      }
    }
  }

  stopLearning() {
    this.#isLearning = false;
  }

  increaseHiddenLayers() {
    this.#assertNotLearning();
    const numberOfNeuronsInLastLayer =
      this.#hiddenLayers[this.#hiddenLayers.length - 1].length;
    this.#hiddenLayers.push([
      new Perceptron(this.#activationFunction).initialize(
        numberOfNeuronsInLastLayer
      ),
      new Perceptron(this.#activationFunction).initialize(
        numberOfNeuronsInLastLayer
      ),
    ]);
    this.#updateOutputLayer();
  }

  decreaseHiddenLayers() {
    this.#assertNotLearning();
    if (this.#hiddenLayers.length === 1) return;
    this.#hiddenLayers.pop();
    this.#updateOutputLayer();
  }

  addNeuronToHiddenLayer(level) {
    this.#assertNotLearning();
    const prevLayer =
      level === 0 ? this.#inputLayer : this.#hiddenLayers[level - 1];
    const nextLayer =
      level === this.#hiddenLayers.length - 1
        ? this.#outputLayer
        : this.#hiddenLayers[level + 1];
    const neuron = new Perceptron(
      this.#activationFunction,
      this.#activationFunctionDerivative
    ).initialize(prevLayer.length);
    this.#hiddenLayers[level].push(neuron);
    nextLayer.forEach((neuron) =>
      neuron.initialize(this.#hiddenLayers[level].length)
    );
  }

  removeNeuronFromHiddenLayer(level) {
    this.#assertNotLearning();
    const nextLayer =
      level === this.#hiddenLayers.length - 1
        ? this.#outputLayer
        : this.#hiddenLayers[level + 1];
    this.#hiddenLayers[level].pop();
    nextLayer.forEach((neuron) =>
      neuron.initialize(this.#hiddenLayers[level].length)
    );
  }

  extractWeights() {
    const weights = [];
    for (let i = 0; i < this.#hiddenLayers.length; i++) {
      for (const neuron of this.#hiddenLayers[i]) {
        weights.concat(neuron.inputWeights);
      }
    }

    return weights;
  }

  updateLayerActivationFunction(actFuncName, level) {
    const [func, funcDerv] = this.#getActivationFunction(actFuncName);
    for (const neuron of this.#hiddenLayers[level]) {
      neuron.activationFunction = func;
      neuron.activationFunctionDerivative = funcDerv;
    }
  }

  getInputLayer() {
    return this.#inputLayer;
  }

  getHiddenLayers() {
    return this.#hiddenLayers;
  }

  getOutputLayer() {
    return this.#outputLayer;
  }

  #updateOutputLayer() {
    const prevLayerNeuronsCount =
      this.#hiddenLayers[this.#hiddenLayers.length - 1].length;
    this.#outputLayer = this.#outputLayer.map((neuron) =>
      neuron.initialize(prevLayerNeuronsCount)
    );
  }

  #getActivationFunction(name) {
    switch (name) {
      case "relu":
        return [reLU, reLUDerivative];
      case "linear":
        return [linear, linearDerivative];
      case "sig":
        return [sig, sigDerivative];
      case "tanh":
        return [tanh, tanhDerivative];
    }
  }

  #assertNotLearning() {
    if (this.#isLearning) {
      throw new Error("Can't update network while learning.");
    }
  }
}
