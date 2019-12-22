import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import labels from "./labels.json";

const ASSETS_URL = `${window.location.origin}/assets`;
const MODEL_URL = `${ASSETS_URL}/model/model.json`;
const IMAGE_SIZE = 224; // Model input size

const loadModel = async () => {
  const model = await loadGraphModel(MODEL_URL);
  // Warm up GPU
  const input = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
  model.predict({ input }); // MobileNet V1
  return model;
};

const predict = async (img, model) => {
  const t0 = performance.now();
  const image = tf.browser.fromPixels(img).toFloat();
  const resized = tf.image.resizeBilinear(image, [IMAGE_SIZE, IMAGE_SIZE]);
  const offset = tf.scalar(255 / 2);
  const normalized = resized.sub(offset).div(offset);
  const input = normalized.expandDims(0);
  const output = await tf.tidy(() => model.predict({ input })).data(); // MobileNet V1
  const predictions = labels
    .map((label, index) => ({ label, accuracy: output[index] }))
    .sort((a, b) => b.accuracy - a.accuracy);
  const time = `${(performance.now() - t0).toFixed(1)} ms`;
  return { predictions, time };
};

const start = async () => {
  const input1 = document.getElementById("input1");
  const output1 = document.getElementById("output1");

  const input2 = document.getElementById("input2");
  const output2 = document.getElementById("output2");

  const input3 = document.getElementById("input3");
  const output3 = document.getElementById("output3");

  const input4 = document.getElementById("input4");
  const output4 = document.getElementById("output4");

  const model = await loadModel();
  const predictions1 = await predict(input1, model);
  const predictions2 = await predict(input2, model);
  const predictions3 = await predict(input3, model);
  const predictions4 = await predict(input4, model);

  output1.append(JSON.stringify(predictions1, null, 2));
  output2.append(JSON.stringify(predictions2, null, 2));
  output3.append(JSON.stringify(predictions3, null, 2));
  output4.append(JSON.stringify(predictions4, null, 2));
};

start();
