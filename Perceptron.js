export class Perceptron {
  static #WEIGHT_INITIALIZATION_COEFFICIENT = 2.4;
  #inputWeights = [];
  #activationFunction = () => 0;
  #activationFunctionDerivative = () => 0;
  #theta = 0;
  #thetaSign = -1;

  static #getRandomWeight(range) {
    return range * (2 * Math.random() - 1);
  }

  constructor(activationFunction, activationFunctionDerivative) {
    this.#activationFunction = activationFunction;
    this.#activationFunctionDerivative = activationFunctionDerivative;
  }

  initialize(numberOfInputs) {
    this.#inputWeights = [];
    if (!numberOfInputs) {
      // Deal with it as an input layer perceptron.
      this.#theta = 0;
      this.#inputWeights.push(1);
      return this;
    }

    const range =
      Perceptron.#WEIGHT_INITIALIZATION_COEFFICIENT / numberOfInputs;
    this.#theta = Perceptron.#getRandomWeight(range);
    for (let i = 0; i < numberOfInputs; i++) {
      this.#inputWeights.push(Perceptron.#getRandomWeight(range));
    }

    return this;
  }

  activate(inputs) {
    if (inputs.length !== this.#inputWeights.length) {
      throw new Error("Inputs must have the same length as weights.");
    }

    let sum = this.#theta * this.#thetaSign;
    for (let i = 0; i < this.#inputWeights.length; i++) {
      sum += this.#inputWeights[i] * inputs[i];
    }

    return {
      value: this.#activationFunction(sum),
      derivedValue: this.#activationFunctionDerivative(sum),
    };
  }

  set thetaSign(s) {
    this.#thetaSign = s;
  }

  get thetaSign() {
    return this.#thetaSign;
  }

  get theta() {
    return this.#theta;
  }

  set theta(t) {
    this.#theta = t;
  }

  get inputWeights() {
    return this.#inputWeights;
  }

  set inputWeights(weights) {
    this.#inputWeights = weights;
  }

  set activationFunction(activationFunction) {
    this.#activationFunction = activationFunction;
  }

  get activationFunction() {
    return this.#activationFunction;
  }

  set activationFunctionDerivative(activationFunctionDerivative) {
    this.#activationFunctionDerivative = activationFunctionDerivative;
  }

  get activationFunctionDerivative() {
    return this.#activationFunctionDerivative;
  }
}
