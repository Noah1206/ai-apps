// FeedbackEngine — the seam between the app and the speech backend.
// Mirrors the research lab's SurfaceASRAdapter philosophy: the app never
// depends on a concrete model, only on this contract.

export type Evaluation = {
  meaning: number; // 의미 전달
  dialect: number; // 사투리 표현
  pronunciation: number; // 발음
  intonation: number; // 억양
  naturalness: number; // 자연스러움
  heard: string; // what the engine thinks the user said
  coachTip: string; // ONE thing to fix (never more)
};

export type SpeechInput = {
  targetText: string;
  audioUri: string;
  durationMs: number;
};

export interface FeedbackEngine {
  evaluate(input: SpeechInput): Promise<Evaluation>;
}

export const DIMENSIONS: { key: keyof Pick<Evaluation, "meaning" | "dialect" | "pronunciation" | "intonation" | "naturalness">; label: string }[] = [
  { key: "meaning", label: "Meaning" },
  { key: "dialect", label: "Dialect" },
  { key: "pronunciation", label: "Pronunciation" },
  { key: "intonation", label: "Intonation" },
  { key: "naturalness", label: "Naturalness" },
];

const TIPS: Record<string, string> = {
  meaning: "Focus on getting the key word across — say the sentence again slowly.",
  dialect: "Try the Busan ending this time (e.g. -노?, -주이소, -심더).",
  pronunciation: "One word was unclear. Repeat just that word three times.",
  intonation: "Busan endings fall then rise slightly — try lifting the very end.",
  naturalness: "Good accuracy — now say it a little faster, like one breath.",
};

// ponytail: duration-heuristic mock engine — swap for real STT API
// (record → STT → compare with targetText) when the backend exists.
class MockEngine implements FeedbackEngine {
  async evaluate({ targetText, durationMs }: SpeechInput): Promise<Evaluation> {
    // Expected speaking time ≈ 140ms per Korean character.
    const expected = Math.max(600, targetText.replace(/\s/g, "").length * 140);
    const ratio = Math.min(durationMs, expected * 2) / expected;
    const closeness = 1 - Math.min(1, Math.abs(1 - ratio)); // 1 = spoke for a plausible duration
    const base = 55 + Math.round(closeness * 30);
    const jitter = () => Math.round(base + (Math.random() - 0.5) * 20);
    const scores = {
      meaning: clamp(jitter()),
      dialect: clamp(jitter()),
      pronunciation: clamp(jitter()),
      intonation: clamp(jitter() - 8), // intonation is the hardest — bias it low
      naturalness: clamp(jitter()),
    };
    const worst = (Object.entries(scores) as [string, number][]).sort((a, b) => a[1] - b[1])[0][0];
    return { ...scores, heard: targetText, coachTip: TIPS[worst] };
  }
}

const clamp = (n: number) => Math.max(30, Math.min(98, n));

export const engine: FeedbackEngine = new MockEngine();
