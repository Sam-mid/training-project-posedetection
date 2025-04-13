import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// Variabelen & Elementen
let image = document.querySelector("#myimage");
let nn; // Neural network
let capturedPoses = [];
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawUtils = new DrawingUtils(canvasCtx);

// Initialisatie
async function initialize() {
    await tf.ready();
    nn = ml5.neuralNetwork({ task: 'classification', debug: true });
    console.log(" Neuraal netwerk is klaar");

    fetch("./handposes.json")
        .then(response => response.json())
        .then(data => startTraining(data))
        .catch(error => console.error(" Kon trainingdata niet laden:", error));
}

// Model trainen
async function startTraining(data) {
    console.log(" Training starten...");

    data.forEach(pose => nn.addData(pose.points, { label: pose.label }));

    nn.normalizeData();
    nn.train({ epochs: 100 }, () => {
        console.log(" Training voltooid");
        loadTestData();
    });
}

// Model testen
function loadTestData() {
    fetch("./handposes_test.json")
        .then(response => response.json())
        .then(testData => evaluateModel(testData))
        .catch(error => console.error(" Kon testdata niet laden:", error));
}

async function evaluateModel(testData) {
    let correct = 0;

    for (let sample of testData) {
        const prediction = await nn.classify(sample.points);
        const predicted = prediction[0].label;
        const actual = sample.label;

        console.log(` Verwacht: ${actual} — Voorspeld: ${predicted}`);
        if (predicted === actual) correct++;
    }

    const accuracy = (correct / testData.length * 100).toFixed(2);
    console.log(`Accuracy op testdata: ${accuracy}%`);
}

// Handlandmarker maken
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });

    console.log("Model geladen, start de webcam");
    enableWebcamButton.addEventListener("click", enableCam);
    logButton.addEventListener("click", logAllHands);
};

// Webcam
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;

        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";

            predictWebcam();
        });
    } catch (error) {
        console.error(" Webcam fout:", error);
    }
}

// Realtime voorspellen
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now());

    if (results.landmarks[0]) {
        const thumb = results.landmarks[0][4];
        image.style.transform = `translate(${video.videoWidth - thumb.x * video.videoWidth}px, ${thumb.y * video.videoHeight}px)`;
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    for (let hand of results.landmarks) {
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// Handcoördinaten loggen
function logAllHands() {
    results.landmarks.forEach(hand => {
        const flat = flattenLandmarks(hand);
        console.log(flat);
    });
}

function flattenLandmarks(landmarks) {
    return landmarks.map(point => [point.x, point.y, point.z]).flat();
}

//  Pose opslaan
function savePose(label) {
    if (!results?.landmarks[0]) {
        console.log("Geen hand gevonden");
        return;
    }

    const pose = flattenLandmarks(results.landmarks[0]);

    capturedPoses.push({ points: pose, label });
    console.log(` Pose opgeslagen als "${label}"`);
    console.log(`${pose}`);
}

//  Live classificatie
async function classifyPose() {
    await tf.ready();

    if (!results?.landmarks[0]) {
        console.log(" Geen hand gevonden");
        return;
    }

    const pose = flattenLandmarks(results.landmarks[0]);
    const prediction = await nn.classify(pose);

    console.log(` Ik denk dat dit: "${prediction[0].label}" is`);
}

// Data exporteren
function downloadPoses() {
    const blob = new Blob([JSON.stringify(capturedPoses, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "handposes.json";
    a.click();
    URL.revokeObjectURL(url);
}

function downloadModel() {
    nn.save();
}

// Start app
if (navigator.mediaDevices?.getUserMedia) {
    createHandLandmarker();
    initialize();
}

// Functies beschikbaar maken in HTML
window.savePose = savePose;
window.downloadPoses = downloadPoses;
window.classifyPose = classifyPose;
window.downloadModel = downloadModel;
