export function reLU(x) {
  return x < 0 ? 0 : x;
}

export function reLUDerivative(x) {
  return x <= 0 ? 0 : 1;
}

export function linear(x) {
  return x;
}

export function linearDerivative(x) {
  return 1;
}

export function tanh(x) {
  return Math.tanh(x);
}

export function tanhDerivative(x) {
  return 1 - Math.pow(Math.tanh(x), 2);
}

export function sig(x) {
  return 1 / (1 + Math.pow(Math.E, -x));
}

export function sigDerivative(x) {
  const s = sig(x);
  return s * (1 - s);
}
