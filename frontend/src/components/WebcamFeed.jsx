import Webcam from "react-webcam";
import { useRef, useState, useEffect } from "react";
import axios from "axios";

function WebcamFeed() {
  const API_URL = import.meta.env.API_URL;
  const webcamRef = useRef(null);
  const [hindiText, setHindiText] = useState("");
  const [isDetecting, setIsDetecting] = useState(false);
  const [prediction, setPrediction] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [loading, setLoading] = useState(false);
  const [sentence, setSentence] = useState("");
  const lastAppendedRef = useRef("");
  const currentStableRef = useRef("");
  const stableStartRef = useRef(null);
  const [cameras, setCameras] = useState([]);
  const [deviceId, setDeviceId] = useState("");
  const videoConstraints = {
    width: 640,
    height: 480,
    deviceId: deviceId ? { exact: deviceId } : undefined,
  };
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      const videoDevices = devices.filter((d) => d.kind === "videoinput");

      setCameras(videoDevices);

      const droidCam = videoDevices.find((d) => d.label.includes("DroidCam"));

      if (droidCam) {
        setDeviceId(droidCam.deviceId);
      } else if (videoDevices.length > 0) {
        setDeviceId(videoDevices[0].deviceId);
      }
    });
  }, []);
  // const duplicateAllowedRef = useRef(false);
  const lastAppendTimeRef = useRef(0);
  const detectSign = async () => {
    if (loading) return;
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;
    const blob = await fetch(imageSrc).then((res) => res.blob());
    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");
    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/predict`, formData);
      const currentLetter = res.data.text;
      setPrediction(currentLetter);
      setConfidence((res.data.confidence * 100).toFixed(1));
      const now = Date.now();
      if (currentLetter === "No Hand") {
        currentStableRef.current = "";
        stableStartRef.current = null;
        return;
      }
      if (currentLetter !== currentStableRef.current) {
        currentStableRef.current = currentLetter;
        stableStartRef.current = now;
        return;
      }
      if (!stableStartRef.current) return;
      if (now - stableStartRef.current < 1500) return;
      // SPACE gesture
      if (currentLetter === "SPACE") {
        setSentence((prev) => {
          if (prev.length === 0) return prev;
          if (prev.endsWith(" ")) return prev;
          return prev + " ";
        });

        return;
      }

      // DELETE gesture
      if (currentLetter === "DELETE") {
        if (now - lastAppendTimeRef.current >= 1500) {
          setSentence((prev) => prev.slice(0, -1));
          lastAppendTimeRef.current = now;
        }

        stableStartRef.current = now;
        return;
      }
      const DUPLICATE_DELAY = 2000; // 2 sec

      // New letter
      if (currentLetter !== lastAppendedRef.current) {
        setSentence((prev) => prev + currentLetter);

        lastAppendedRef.current = currentLetter;
        lastAppendTimeRef.current = now;
        stableStartRef.current = now;

        return;
      }

      // Same letter again after delay
      if (
        currentLetter === lastAppendedRef.current &&
        now - lastAppendTimeRef.current >= DUPLICATE_DELAY
      ) {
        setSentence((prev) => prev + currentLetter);

        lastAppendTimeRef.current = now;
        stableStartRef.current = now;
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    if (!sentence.trim()) {
      setHindiText("");
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const res = await axios.post(`${API_URL}/translate`, {
          text: sentence,
        });

        setHindiText(res.data.translation);
      } catch (err) {
        console.error(err);
      }
    }, 1000);

    return () => clearTimeout(timer);
  }, [sentence]);
  useEffect(() => {
    let interval;
    if (isDetecting) {
      interval = setInterval(() => {
        detectSign();
      }, 500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isDetecting]);
  const clearAll = () => {
    setSentence("");
    setPrediction("-");
    setConfidence(0);
    lastAppendedRef.current = "";
    currentStableRef.current = "";
    stableStartRef.current = null;
    lastAppendTimeRef.current = 0;
  };
  const speakSentence = () => {
    if (!sentence) return;
    const speech = new SpeechSynthesisUtterance(sentence);
    speech.rate = 1;
    speech.pitch = 1;
    speech.volume = 1;
    window.speechSynthesis.speak(speech);
  };
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-[96vh]">
      {/* LEFT */}
      <div className="absolute top-3 left-3 z-10">
        <select
          value={deviceId}
          onChange={(e) => setDeviceId(e.target.value)}
          className="bg-black/70 text-white px-3 py-2 rounded-lg"
        >
          {cameras.map((camera) => (
            <option key={camera.deviceId} value={camera.deviceId}>
              {camera.label}
            </option>
          ))}
        </select>
      </div>
      <div className="relative h-full">
        <Webcam
          mirrored
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          className="w-full h-full object-cover rounded-2xl border  border-white/20"
        />

        <div className="absolute top-3 right-3">
          <span
            className={`px-3 py-1 rounded-full text-xs font-semibold ${
              isDetecting ? "bg-green-500" : "bg-red-500"
            }`}
          >
            {isDetecting ? "LIVE" : "OFFLINE"}
          </span>
        </div>
      </div>

      {/* RIGHT */}
      <div className="flex flex-col gap-3 h-full">
        <div className="bg-white/10 rounded-2xl p-4 flex-1">
          <p className="text-slate-400 text-sm">Prediction</p>

          <div className="flex items-center justify-center h-full">
            <h2 className="text-5xl font-bold text-cyan-400">{prediction}</h2>
          </div>
        </div>

        <div className="bg-white/10 rounded-2xl p-4">
          <p className="text-slate-400 text-sm">Confidence</p>

          <h2 className="text-3xl font-bold text-green-400 mt-2">
            {confidence}%
          </h2>

          <div className="mt-3 h-3 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-400"
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <p className="text-white text-sm mb-2">English</p>

            <div className="bg-white/10 rounded-xl text-white p-3 min-h-[120px]">
              {sentence}
              <span className="animate-pulse text-cyan-400">|</span>
            </div>
          </div>

          <div>
            <p className="text-white text-sm mb-2">Hindi</p>

            <div className="bg-white/10 text-white rounded-xl p-3 min-h-[120px]">
              {hindiText || (
                <span className="text-slate-500">
                  Translation will appear here...
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 justify-center">
          <button
            onClick={() => setIsDetecting(!isDetecting)}
            className={`px-4 py-2 rounded-lg font-semibold ${
              isDetecting ? "bg-red-500" : "bg-cyan-500"
            }`}
          >
            {isDetecting ? "Stop" : "Start"}
          </button>

          <button
            onClick={clearAll}
            className="px-4 py-2 rounded-lg bg-slate-700"
          >
            Clear
          </button>

          <button
            onClick={speakSentence}
            className="px-4 py-2 rounded-lg bg-green-500"
          >
            Speak
          </button>
        </div>
      </div>
    </div>
  );
}
export default WebcamFeed;
