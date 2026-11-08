const expected = 8.145;
const density = (8.2 * 8.0 * 8.1 * 8.3) ** 0.25;
if (Math.abs(density - expected) > 0.01) throw new Error('density mismatch');
console.log('ok prompt-kiln density=' + density.toFixed(3));
