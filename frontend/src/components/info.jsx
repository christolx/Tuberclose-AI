import { X } from "lucide-react";
import { useEffect, useState } from "react";

export default function Info({ onClose, fadeOut }) {
  // 1. Create a local state to track if the component has mounted
  const [isVisible, setIsVisible] = useState(false);

  // 2. Trigger the fade-in animation immediately after mounting
  useEffect(() => {
    // Using a small timeout ensures the browser registers the initial 'opacity-0' state
    // before switching to 'opacity-100' to trigger the transition.
    const timer = setTimeout(() => {
      setIsVisible(true);
    }, 10);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div
      className={`absolute inset-0 w-screen h-screen bg-neutral-800/40 z-10 backdrop-blur-3xl flex justify-center items-center transition-all duration-1000 ${
        // 3. Logic: If closing (fadeOut) OR just mounted (!isVisible) -> opacity 0
        fadeOut || !isVisible ? "opacity-0" : "opacity-100"
      }`}
    >
      <div className="flex flex-col text-white text-md md:text-2xl text-left gap-4 w-2xs md:w-sm">
        <X
          onClick={onClose}
          className="self-end w-5 md:w-6 h-auto transition-transform duration-300 hover:rotate-180 cursor-pointer"
        ></X>
        <div>
          <h1 className="font-light text-left">Model Name</h1>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            ResNet50: Fine-tuned for TB Detection
          </h3>
        </div>
        <div>
          <h1 className="font-light text-left">Dataset</h1>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Kaggle TB Chest X-Ray Dataset -{" "}
            <span className="italic">Tawsifur Rahman</span>
          </h3>
        </div>
        <div>
          <h1 className="font-light text-left">Details</h1>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Epoch: 15
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Batch size: 32
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Optimizer: Adam
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Unfreeze last 10 layers of ResNet50
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Model version: v1.0
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Trained: 2025-11-19
          </h3>
        </div>
        <div>
          <h1 className="font-normal text-left">DISCLAIMER</h1>
          <p className="text-xs md:text-sm font-extralight text-neutral-300">
            This AI model is intended solely for research, learning, and
            demonstration purposes. The outputs produced by this system may
            contain errors, biases, or inaccuracies, and should not be relied
            upon for critical or high-stakes decisions. This system{" "}
            <span className="font-semibold">
              does not provide professional medical, legal, or financial
              guidance.
            </span>{" "}
            The developers and contributors are not liable for any damages,
            losses, or consequences resulting from the use of this model. By
            continuing to use this application, you acknowledge and accept these
            limitations.
          </p>
        </div>
      </div>
    </div>
  );
}