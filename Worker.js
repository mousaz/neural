import NeuralNetwork from "./NeuralNetwork.js";

self.onmessage = (e) => {
  const {
    networkParams,
    hiddenActFunc,
    outputActFunc,
    data,
    learnTesting,
    epochs,
    learningRate,
    errorGoal,
  } = e.data;
  const network = new NeuralNetwork(
    data,
    hiddenActFunc,
    outputActFunc,
    true,
    networkParams
  );
  network.onIterationCompleted = () => {
    self.postMessage({
      action: "IterationCompleted",
      networkParams: network.extractNetworkParams(),
    });
  };
  network.onEpochCompleted = (mse, validationError) => {
    self.postMessage({
      action: "EpochCompleted",
      networkParams: network.extractNetworkParams(),
      mse,
      validationError,
    });
  };
  network.startLearning(learningRate, learnTesting, epochs, errorGoal);
};
