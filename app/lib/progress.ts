import AsyncStorage from "@react-native-async-storage/async-storage";
import { Evaluation } from "./feedback";

// Ebbinghaus review intervals (days) — mvp_plan.md §12.7
const INTERVALS = [1, 3, 7, 14];

export type ReviewItem = { phraseId: string; situationId: string; due: number; step: number };

export type Progress = {
  completed: string[]; // situation ids, in order
  results: Record<string, Evaluation>; // last aggregate result per situation
  reviews: ReviewItem[];
  streakDay: string | null;
  streak: number;
};

const KEY = "busan-progress-v1";

const EMPTY: Progress = { completed: [], results: {}, reviews: [], streakDay: null, streak: 0 };

export async function getProgress(): Promise<Progress> {
  const raw = await AsyncStorage.getItem(KEY);
  return raw ? { ...EMPTY, ...JSON.parse(raw) } : EMPTY;
}

async function save(p: Progress) {
  await AsyncStorage.setItem(KEY, JSON.stringify(p));
}

export async function completeSituation(
  situationId: string,
  phraseIds: string[],
  result: Evaluation,
): Promise<Progress> {
  const p = await getProgress();
  if (!p.completed.includes(situationId)) p.completed.push(situationId);
  p.results[situationId] = result;
  const now = Date.now();
  for (const phraseId of phraseIds) {
    if (!p.reviews.some((r) => r.phraseId === phraseId)) {
      p.reviews.push({ phraseId, situationId, due: now + INTERVALS[0] * 86400_000, step: 0 });
    }
  }
  const today = new Date().toDateString();
  if (p.streakDay !== today) {
    const yesterday = new Date(now - 86400_000).toDateString();
    p.streak = p.streakDay === yesterday ? p.streak + 1 : 1;
    p.streakDay = today;
  }
  await save(p);
  return p;
}

export async function completeReview(phraseId: string): Promise<Progress> {
  const p = await getProgress();
  const item = p.reviews.find((r) => r.phraseId === phraseId);
  if (item) {
    item.step = Math.min(item.step + 1, INTERVALS.length - 1);
    item.due = Date.now() + INTERVALS[item.step] * 86400_000;
  }
  await save(p);
  return p;
}

export const dueReviews = (p: Progress) => p.reviews.filter((r) => r.due <= Date.now());
