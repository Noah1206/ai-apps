import { useMemo, useState } from "react";
import { Pressable, ScrollView, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Card, DarkButton, GhostButton, ScoreBar, TipToggle } from "../../components/ui";
import { Recorder } from "../../components/Recorder";
import { getSituation, Phrase } from "../../lib/content";
import { DIMENSIONS, Evaluation } from "../../lib/feedback";
import { speak } from "../../lib/audio";
import { completeSituation } from "../../lib/progress";
import { colors, radius, space, type } from "../../lib/theme";

// Step-up flow (Main Contents doc): listen → shadow → guided response → result.
type Step =
  | { kind: "listen"; phrase: Phrase; choices: string[] }
  | { kind: "shadow"; phrase: Phrase }
  | { kind: "respond"; npc: Phrase; hint: Phrase };

export default function Mission() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const situation = getSituation(id ?? "");

  const steps: Step[] = useMemo(() => {
    if (!situation) return [];
    const p = situation.phrases;
    const choicesFor = (i: number) => {
      const opts = [p[i].en, p[(i + 1) % p.length].en, p[(i + 2) % p.length].en];
      // rotate so the correct answer position varies per step
      return opts.slice(i % 3).concat(opts.slice(0, i % 3));
    };
    return [
      { kind: "listen", phrase: p[0], choices: choicesFor(0) },
      { kind: "listen", phrase: p[1], choices: choicesFor(1) },
      { kind: "shadow", phrase: p[0] },
      { kind: "shadow", phrase: p[2] },
      { kind: "respond", npc: situation.npcGreeting, hint: p[3] },
    ];
  }, [situation]);

  const [stepIdx, setStepIdx] = useState(0);
  const [picked, setPicked] = useState<string | null>(null);
  const [evals, setEvals] = useState<Evaluation[]>([]);
  const [lastEval, setLastEval] = useState<Evaluation | null>(null);
  const [finished, setFinished] = useState(false);

  if (!situation) return null;

  const step = steps[stepIdx];
  const done = stepIdx >= steps.length;

  function next() {
    setPicked(null);
    setLastEval(null);
    setStepIdx((i) => i + 1);
  }

  function onEval(e: Evaluation) {
    setLastEval(e);
    setEvals((prev) => [...prev, e]);
  }

  async function finish() {
    const avg = (k: keyof Evaluation) =>
      Math.round(evals.reduce((s, e) => s + (e[k] as number), 0) / Math.max(1, evals.length));
    const result: Evaluation = {
      meaning: avg("meaning"),
      dialect: avg("dialect"),
      pronunciation: avg("pronunciation"),
      intonation: avg("intonation"),
      naturalness: avg("naturalness"),
      heard: "",
      coachTip: evals.length ? evals[evals.length - 1].coachTip : "",
    };
    await completeSituation(situation!.id, situation!.phrases.map((p) => p.id), result);
    setFinished(true);
  }

  // ---------- Result ----------
  if (done) {
    const avg = (k: keyof Evaluation) =>
      Math.round(evals.reduce((s, e) => s + (e[k] as number), 0) / Math.max(1, evals.length));
    const dims = DIMENSIONS.map((d) => ({ ...d, value: avg(d.key) }));
    const worst = [...dims].sort((a, b) => a.value - b.value)[0];
    return (
      <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
        <ScrollView contentContainerStyle={{ padding: space(3), gap: space(2) }}>
          <Text style={type.hero}>Episode {situation.episode} complete</Text>
          <Card>
            {dims.map((d) => (
              <ScoreBar key={d.key} label={d.label} value={d.value} />
            ))}
          </Card>
          <Card>
            <Text style={[type.caption, { color: colors.ink, marginBottom: 4 }]}>Coach</Text>
            <Text style={type.body}>
              Fix one thing next time: {worst.label.toLowerCase()}.{" "}
              {evals.length ? evals[evals.length - 1].coachTip : ""}
            </Text>
          </Card>
          <DarkButton
            title={finished ? "Done" : "Save & finish"}
            onPress={async () => {
              if (!finished) await finish();
              router.back();
            }}
          />
        </ScrollView>
      </SafeAreaView>
    );
  }

  // ---------- Steps ----------
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
      <ScrollView contentContainerStyle={{ padding: space(3), gap: space(2), flexGrow: 1 }}>
        <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
          <Text style={type.caption}>
            {situation.title} · {stepIdx + 1}/{steps.length}
          </Text>
          <GhostButton title="✕" onPress={() => router.back()} />
        </View>

        {step.kind === "listen" && (
          <>
            <Text style={type.heading}>What does it mean?</Text>
            <Card>
              <View style={{ flexDirection: "row", gap: space(1) }}>
                <GhostButton title="▶ Listen" onPress={() => speak(step.phrase.ko)} />
                <GhostButton title="▶ Slow" onPress={() => speak(step.phrase.ko, true)} />
              </View>
            </Card>
            {step.choices.map((c) => {
              const isCorrect = c === step.phrase.en;
              const chosen = picked === c;
              return (
                <Pressable
                  key={c}
                  onPress={() => setPicked(c)}
                  style={({ pressed }) => pressed && { opacity: 0.8 }}
                >
                  <Card
                    style={
                      chosen
                        ? { borderColor: colors.ink, backgroundColor: isCorrect ? colors.bg : "rgba(28,28,28,0.04)" }
                        : undefined
                    }
                  >
                    <Text style={type.body}>
                      {c}
                      {chosen ? (isCorrect ? "  ✓" : "  ✕") : ""}
                    </Text>
                  </Card>
                </Pressable>
              );
            })}
            {picked && picked !== step.phrase.en && (
              <Text style={type.caption}>Not quite — listen once more and try again.</Text>
            )}
            {picked === step.phrase.en && (
              <>
                <Text style={[type.caption, { color: colors.ink }]}>
                  ✓ "{step.phrase.ko}" — {step.phrase.en}
                </Text>
                <DarkButton title="Next" onPress={next} />
              </>
            )}
          </>
        )}

        {step.kind === "shadow" && (
          <>
            <Text style={type.heading}>Repeat after Busan</Text>
            <Card>
              <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                <Text style={type.cardTitle}>{step.phrase.ko}</Text>
                <TipToggle std={step.phrase.std} nuance={step.phrase.nuance} />
              </View>
              <Text style={[type.caption, { marginBottom: space(1.5) }]}>{step.phrase.en}</Text>
              <View style={{ flexDirection: "row", gap: space(1) }}>
                <GhostButton title="▶ Listen" onPress={() => speak(step.phrase.ko)} />
                <GhostButton title="▶ Slow" onPress={() => speak(step.phrase.ko, true)} />
              </View>
            </Card>
            <Recorder targetText={step.phrase.ko} onResult={onEval} />
            {lastEval && (
              <>
                <Card>
                  <Text style={[type.caption, { color: colors.ink, marginBottom: 4 }]}>Coach</Text>
                  <Text style={type.body}>{lastEval.coachTip}</Text>
                </Card>
                <DarkButton title="Next" onPress={next} />
              </>
            )}
          </>
        )}

        {step.kind === "respond" && (
          <>
            <Text style={type.heading}>Answer {situation.npcName}</Text>
            <View style={{ alignSelf: "flex-start", maxWidth: "85%" }}>
              <Card>
                <Text style={type.caption}>{situation.npcName}</Text>
                <Text style={type.cardTitle}>{step.npc.ko}</Text>
                <Text style={type.caption}>{step.npc.en}</Text>
                <View style={{ marginTop: space(1) }}>
                  <GhostButton title="▶ Listen" onPress={() => speak(step.npc.ko)} />
                </View>
              </Card>
            </View>
            {lastEval && (
              <View
                style={{
                  alignSelf: "flex-end",
                  maxWidth: "85%",
                  backgroundColor: colors.ink,
                  borderRadius: radius.card,
                  padding: space(2),
                }}
              >
                <Text style={{ color: colors.onDark, fontSize: 16 }}>{lastEval.heard || "(your answer)"}</Text>
              </View>
            )}
            <Card>
              <Text style={type.caption}>
                Hint — you could use: "{step.hint.ko}" ({step.hint.en})
              </Text>
            </Card>
            <Recorder targetText={step.hint.ko} onResult={onEval} />
            {lastEval && (
              <>
                <Card>
                  <Text style={[type.caption, { color: colors.ink, marginBottom: 4 }]}>Coach</Text>
                  <Text style={type.body}>{lastEval.coachTip}</Text>
                </Card>
                <DarkButton title="See results" onPress={next} />
              </>
            )}
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
