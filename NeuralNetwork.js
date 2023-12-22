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
  #hiddenActivationFunction = tanh;
  #hiddenActivationFunctionDerivative = tanhDerivative;
  #outputActivationFunction = tanh;
  #outputActivationFunctionDerivative = tanhDerivative;
  #outputExpectedValuesByClasses = {};

  onIterationCompleted = () => {};
  onEpochCompleted = () => {};

  constructor(
    data,
    hiddenLayersActivation,
    outputLayerActivation,
    isClassification,
    params
  ) {
    this.#data = data;
    [this.#hiddenActivationFunction, this.#hiddenActivationFunctionDerivative] =
      this.#getActivationFunction(hiddenLayersActivation);
    [this.#outputActivationFunction, this.#outputActivationFunctionDerivative] =
      this.#getActivationFunction(outputLayerActivation);

    // Initialize input neurons
    const numOfInputNeurons = data.columnsCount - 1;
    for (let i = 0; i < numOfInputNeurons; i++) {
      const perceptron = new Perceptron(linear, linearDerivative);
      perceptron.initialize();
      this.#inputLayer.push(perceptron);
    }

    if (params) {
      this.#setNetworkParams(params);
      return this;
    }

    // Initialize 1-hidden layer with two neurons
    this.#hiddenLayers.push([
      new Perceptron(
        this.#hiddenActivationFunction,
        this.#hiddenActivationFunctionDerivative
      ).initialize(this.#inputLayer.length),
      new Perceptron(
        this.#hiddenActivationFunction,
        this.#hiddenActivationFunctionDerivative
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

      let numberOfClasses = Object.getOwnPropertyNames(classes).length;
      numberOfOutputNeurons = numberOfClasses > 2 ? numberOfClasses : 1;
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
        this.#outputActivationFunction,
        this.#outputActivationFunctionDerivative
      );
      perceptron.initialize(numberOfPreviousHiddenLayerNeurons);
      this.#outputLayer.push(perceptron);
    }
  }

  startLearning(
    learningRate,
    [learnPercentage, validationPercentage],
    epochs,
    targetError
  ) {
    this.#isLearning = true;
    const learnDataLength = Math.floor(this.#data.rowsCount * learnPercentage);
    const validationDataLength = Math.floor(
      this.#data.rowsCount * validationPercentage
    );
    const layers = this.#hiddenLayers.concat([[...this.#outputLayer]]);
    for (let epochNumber = 1; epochNumber <= epochs; epochNumber++) {
      let sse = 0;
      for (
        let dataRowIndex = 0;
        dataRowIndex < learnDataLength;
        dataRowIndex++
      ) {
        // Activate
        let activatedValues = [];
        for (
          let dataVariableIndex = 0;
          dataVariableIndex < this.#data.columnsCount - 1;
          dataVariableIndex++
        ) {
          activatedValues.push({
            value: parseFloat(this.#data[dataVariableIndex][dataRowIndex]),
          });
        }

        activatedValues = [[...activatedValues]];
        for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
          const layerInputs = activatedValues[layerIndex].map(
            (layerValues) => layerValues.value
          );
          activatedValues.push(
            layers[layerIndex].map((neuron) => neuron.activate(layerInputs))
          );
        }

        // Weight Training
        let errors = [];
        const expectedOutput = this.#getExpectedOutput(
          this.#data[this.#data.columnsCount - 1][dataRowIndex]
        );
        for (
          let layerIndex = layers.length - 1;
          layerIndex >= 0;
          layerIndex--
        ) {
          let layerOutput = activatedValues.pop();

          if (layerIndex === layers.length - 1) {
            // Calculate output layer errors and consider them in the SSE calculation
            errors = expectedOutput.map((value, index) => {
              return value - layerOutput[index].value;
            });
            sse += errors.reduce((prev, curr) => prev + curr * curr, 0);
          }

          const layer = layers[layerIndex];

          let sigmas = errors.map(
            (err, index) => err * layerOutput[index].derivedValue
          );

          let deltas = [];
          errors = [];
          for (let neuronIndex = 0; neuronIndex < layer.length; neuronIndex++) {
            let sigma = sigmas[neuronIndex];
            deltas.push(
              layer[neuronIndex].inputWeights.map(
                (_, index) =>
                  activatedValues[activatedValues.length - 1][index].value *
                  sigma *
                  learningRate
              )
            );

            // Update errors for the previous layer (hidden)
            if (!errors.length)
              errors = new Array(layer[neuronIndex].inputWeights.length).fill(
                0
              );
            for (
              let inputIndex = 0;
              inputIndex < layer[neuronIndex].inputWeights.length;
              inputIndex++
            ) {
              errors[inputIndex] =
                errors[inputIndex] +
                layer[neuronIndex].inputWeights[inputIndex] * sigma;
            }
          }

          // Update the weights
          deltas.forEach((delta, index) => {
            layer[index].inputWeights = delta.map(
              (d, dIndex) => layer[index].inputWeights[dIndex] + d
            );
            layer[index].theta +=
              layer[index].thetaSign * sigmas[index] * learningRate;
          });
        }

        this.onIterationCompleted();
      }

      // Check sse and mse
      const mse = sse / this.#data.rowsCount;
      if (mse <= targetError) {
        // Achieved target error; end learning
        break;
      }

      const validationError = this.validate(
        learnDataLength,
        validationDataLength
      );
      this.onEpochCompleted(mse, validationError);
    }

    this.#isLearning = false;
  }

  stopLearning() {
    this.#isLearning = false;
  }

  validate(dataStartIndex, validationDataLength) {
    let validationErrors = 0;
    for (
      let validationIndex = dataStartIndex;
      validationIndex < validationDataLength + dataStartIndex;
      validationIndex++
    ) {
      let inputValues = [];
      for (
        let dataVariableIndex = 0;
        dataVariableIndex < this.#data.columnsCount - 1;
        dataVariableIndex++
      ) {
        inputValues.push(this.#data[dataVariableIndex][validationIndex]);
      }
      const expected = this.#getExpectedOutput(
        this.#data[this.#data.columnsCount - 1][validationIndex]
      );
      const result = this.predict(inputValues);
      validationErrors += expected
        .map((e, i) => e - result[i])
        .reduce((s, e) => s + e * e, 0);
    }

    return validationErrors / validationDataLength;
  }

  increaseHiddenLayers() {
    this.#assertNotLearning();
    const numberOfNeuronsInLastLayer =
      this.#hiddenLayers[this.#hiddenLayers.length - 1].length;
    this.#hiddenLayers.push([
      new Perceptron(
        this.#hiddenActivationFunction,
        this.#hiddenActivationFunctionDerivative
      ).initialize(numberOfNeuronsInLastLayer),
      new Perceptron(
        this.#hiddenActivationFunction,
        this.#hiddenActivationFunctionDerivative
      ).initialize(numberOfNeuronsInLastLayer),
    ]);
    this.#updateOutputLayer();
  }

  decreaseHiddenLayers() {
    this.#assertNotLearning();
    if (this.#hiddenLayers.length === 1) return;
    this.#hiddenLayers.pop();
    this.#updateOutputLayer();
  }

  addNeuronToHiddenLayer(level, actFuncName) {
    this.#assertNotLearning();
    const [actFunc, actFuncDer] = this.#getActivationFunction(actFuncName);
    const prevLayer =
      level === 0 ? this.#inputLayer : this.#hiddenLayers[level - 1];
    const nextLayer =
      level === this.#hiddenLayers.length - 1
        ? this.#outputLayer
        : this.#hiddenLayers[level + 1];
    const neuron = new Perceptron(actFunc, actFuncDer).initialize(
      prevLayer.length
    );
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

  extractNetworkParams() {
    return {
      hiddenLayers: this.#hiddenLayers.map((layer) =>
        layer.map((neuron) => {
          return { weights: neuron.inputWeights, theta: neuron.theta };
        })
      ),
      outputLayer: this.#outputLayer.map((neuron) => {
        return { weights: neuron.inputWeights, theta: neuron.theta };
      }),
    };
  }

  extractModel() {
    const layers = this.#hiddenLayers.concat([[...this.#outputLayer]]);
    return (inputs) => {};
  }

  updateHiddenLayersActivationFunction(actFuncName) {
    [this.#hiddenActivationFunction, this.#hiddenActivationFunctionDerivative] =
      this.#getActivationFunction(actFuncName);
    for (const layer of this.#hiddenLayers) {
      for (const neuron of layer) {
        neuron.activationFunction = this.#hiddenActivationFunction;
        neuron.activationFunctionDerivative =
          this.#hiddenActivationFunctionDerivative;
      }
    }
  }

  updateOutputLayerActivationFunction(actFuncName) {
    [this.#outputActivationFunction, this.#outputActivationFunctionDerivative] =
      this.#getActivationFunction(actFuncName);
    for (const neuron of this.#outputLayer) {
      neuron.activationFunction = this.#outputActivationFunction;
      neuron.activationFunctionDerivative =
        this.#outputActivationFunctionDerivative;
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

  predict(inputs) {
    const layers = this.#hiddenLayers.concat([[...this.#outputLayer]]);
    let layerValues = inputs.map((val) => {
      return { value: parseFloat(val) };
    });
    for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
      layerValues = layers[layerIndex].map((neuron) =>
        neuron.activate(layerValues.map((val) => val.value))
      );
    }

    return layerValues.map((val) => val.value);
  }

  reset() {
    this.#assertNotLearning();
    const layers = this.#hiddenLayers.concat([[...this.#outputLayer]]);
    for (const layer of layers) {
      for (const neuron of layer) {
        neuron.initialize(neuron.inputWeights.length);
      }
    }
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

  #setNetworkParams({ hiddenLayers, outputLayer }) {
    this.#hiddenLayers = [];
    for (let i = 0; i < hiddenLayers.length; i++) {
      const hiddenLayer = hiddenLayers[i];
      this.#hiddenLayers.push([]);
      for (const neuronInfo of hiddenLayer) {
        const neuron = new Perceptron(
          this.#hiddenActivationFunction,
          this.#hiddenActivationFunctionDerivative
        );
        neuron.inputWeights = neuronInfo.weights;
        neuron.theta = neuronInfo.theta;
        this.#hiddenLayers[i].push(neuron);
      }
    }

    this.#outputLayer = [];
    for (const neuronInfo of outputLayer) {
      const neuron = new Perceptron(
        this.#outputActivationFunction,
        this.#outputActivationFunctionDerivative
      );
      neuron.inputWeights = neuronInfo.weights;
      neuron.theta = neuronInfo.theta;
      this.#outputLayer.push(neuron);
    }
  }

  #assertNotLearning() {
    if (this.#isLearning) {
      throw new Error("Can't update network while learning.");
    }
  }

  #getExpectedOutput(actualValue) {
    const expectedValue = parseInt(actualValue);
    if (this.#outputLayer.length === 1) {
      return [
        this.#outputActivationFunction.name === "tanh" && expectedValue === 0
          ? -1
          : expectedValue,
      ];
    }

    if (this.#outputLayer.length > 1) {
      const expectedOutput = new Array(this.#outputLayer.length).fill(
        this.#outputActivationFunction.name === "tanh" ? -1 : 0
      );
      expectedOutput[expectedValue] = 1;
      return expectedOutput;
    }
  }
}
