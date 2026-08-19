import { AudioModule, RecordingPresets, setAudioModeAsync, AudioRecorder } from "expo-audio";
import * as Speech from "expo-speech";

// Native-speaker recordings (slow + natural, from the Audio Lab pipeline)
// replace this TTS placeholder later.
export function speak(text: string, slow = false) {
  Speech.stop();
  Speech.speak(text, { language: "ko-KR", rate: slow ? 0.55 : 0.9 });
}

let recorder: AudioRecorder | null = null;
let startedAt = 0;

export async function startRecording(): Promise<boolean> {
  const perm = await AudioModule.requestRecordingPermissionsAsync();
  if (!perm.granted) return false;
  await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
  recorder = new AudioRecorder(RecordingPresets.HIGH_QUALITY);
  await recorder.prepareToRecordAsync();
  recorder.record();
  startedAt = Date.now();
  return true;
}

export async function stopRecording(): Promise<{ uri: string; durationMs: number } | null> {
  if (!recorder) return null;
  const durationMs = Date.now() - startedAt;
  await recorder.stop();
  const uri = recorder.uri ?? "";
  recorder = null;
  await setAudioModeAsync({ allowsRecording: false });
  return { uri, durationMs };
}
