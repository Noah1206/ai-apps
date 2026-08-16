import { Audio } from "expo-av";
import * as Speech from "expo-speech";

// Native-speaker recordings (slow + natural, from the Audio Lab pipeline)
// replace this TTS placeholder later.
export function speak(text: string, slow = false) {
  Speech.stop();
  Speech.speak(text, { language: "ko-KR", rate: slow ? 0.55 : 0.9 });
}

let recording: Audio.Recording | null = null;
let startedAt = 0;

export async function startRecording(): Promise<boolean> {
  const perm = await Audio.requestPermissionsAsync();
  if (!perm.granted) return false;
  await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
  const r = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
  recording = r.recording;
  startedAt = Date.now();
  return true;
}

export async function stopRecording(): Promise<{ uri: string; durationMs: number } | null> {
  if (!recording) return null;
  const durationMs = Date.now() - startedAt;
  await recording.stopAndUnloadAsync();
  const uri = recording.getURI() ?? "";
  recording = null;
  await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
  return { uri, durationMs };
}
