let model;
let rawLabels = [];
let labels = [];
const labelContainer = document.getElementById("label-container");
const historyContainer = document.getElementById("history-container");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

let predictionQueue = [];
const queueSize = 15;
const confidenceThreshold = 80; // higher for strong model
const unrecognizedThreshold = 15;
const confirmMajority = 10;

let lastStableLetter = '';
let videoStream = null;
let running = false;
let videoElement = document.getElementById('webcam');

let loopRunning = false; // to prevent multiple loops

startBtn.onclick = async () => {
  if (running) return;  // prevent multiple starts
  running = true;
  await startModel();
};

stopBtn.onclick = () => {
  running = false;
  labelContainer.innerText = "Model stopped";
  predictionQueue = [];
};

async function startModel() {
  labelContainer.innerText = "Loading model...";
  if (!model) {
    model = await tf.loadLayersModel('model/model.json');
    await loadMetadata();
  }
  if (!videoStream) {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = videoStream;
  }

  await new Promise(resolve => {
    videoElement.onloadedmetadata = () => {
      if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
        resolve();
      }
    };
  });

  labelContainer.innerText = "Model started. Move your hand...";
  predictLoop();
}

async function loadMetadata() {
  const response = await fetch('model/metadata.json');
  const metadata = await response.json();
  rawLabels = metadata.labels;

  labels = rawLabels.map(label => {
    if (!isNaN(label)) {
      return String.fromCharCode(65 + parseInt(label)); // "0" → "A"
    } else {
      return label; // Already "A", "B", "No_Gesture", etc.
    }
  });

  console.log("Loaded labels:", labels);
}

async function predictLoop() {
  if (loopRunning) return; // prevent reentry if already running
  loopRunning = true;

  while (running) {
    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
      await tf.nextFrame();
      continue;
    }

    const img = tf.browser.fromPixels(videoElement)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .expandDims();

    const prediction = await model.predict(img).data();
    img.dispose();

    const maxIndex = prediction.indexOf(Math.max(...prediction));
    const letter = labels[maxIndex];
    const accuracy = (prediction[maxIndex] * 100).toFixed(2);

    predictionQueue.push({ letter, accuracy: parseFloat(accuracy) });

    if (predictionQueue.length > queueSize) {
      predictionQueue.shift();
    }

    const filtered = predictionQueue.filter(item => item.letter === letter);
    const avgAccuracy = filtered.reduce((sum, item) => sum + item.accuracy, 0) / filtered.length;

    if (letter.toLowerCase() === 'no_gesture' && avgAccuracy >= confidenceThreshold && filtered.length >= confirmMajority) {
      labelContainer.innerText = `No Gesture Detected`;
      labelContainer.style.color = 'orange';
    }
    else if (avgAccuracy >= confidenceThreshold && filtered.length >= confirmMajority) {
      if (lastStableLetter !== letter) {
        lastStableLetter = letter;
        labelContainer.innerText = `Detected: ${letter} (Accuracy: ${avgAccuracy.toFixed(2)}%)`;
        labelContainer.style.color = 'black';
        addToHistory(letter, avgAccuracy.toFixed(2));
      } else {
        labelContainer.innerText = `Detected: ${letter} (Accuracy: ${avgAccuracy.toFixed(2)}%)`;
        labelContainer.style.color = 'black';
      }
    }
    else if (avgAccuracy < unrecognizedThreshold && filtered.length >= confirmMajority) {
      labelContainer.innerText = `Unrecognized Gesture`;
      labelContainer.style.color = 'red';
    } else {
      labelContainer.innerText = `Move hand closer...`;
      labelContainer.style.color = 'gray';
    }

    await tf.nextFrame();
    // Optional small delay to reduce GPU load:
    await new Promise(r => setTimeout(r, 30));
  }

  loopRunning = false;
}

function addToHistory(letter, accuracy) {
  const span = document.createElement('span');
  span.innerText = `${letter} - ${accuracy}%`;
  historyContainer.appendChild(span);
}
