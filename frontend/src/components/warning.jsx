import { ShinyButton } from "./ui/shiny-button";
export default function WarningDesc({ onInfoClick }) {
  return (
    <div className="flex flex-col lg:items-start gap-y-2">
      <div className="text-xs md:text-[10px] lg:text-sm text-neutral-300 font-normal md:text-left lg:text-left">
        For screening purposes only
        <br />
        <span className="font-bold">Do not</span> interpret as an official
        medical diagnosis.
      </div>
      <ShinyButton onClick={onInfoClick} className="p-0 border-0 text-white underline font-normal text-sm md:text-left lg:text-left">AI model info</ShinyButton>
    </div>
  );
}
