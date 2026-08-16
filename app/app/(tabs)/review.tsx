import { useCallback, useState } from "react";
import { ScrollView, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useFocusEffect } from "expo-router";
import { Card, GhostButton, TipToggle } from "../../components/ui";
import { Recorder } from "../../components/Recorder";
import { SITUATIONS } from "../../lib/content";
import { Evaluation } from "../../lib/feedback";
import { speak } from "../../lib/audio";
import { completeReview, dueReviews, getProgress, Progress } from "../../lib/progress";
import { colors, space, type } from "../../lib/theme";

const allPhrases = SITUATIONS.flatMap((s) => s.phrases);

export default function Review() {
  const [progress, setProgress] = useState<Progress | null>(null);
  const [feedback, setFeedback] = useState<Record<string, Evaluation>>({});

  useFocusEffect(
    useCallback(() => {
      getProgress().then(setProgress);
    }, []),
  );

  const due = progress ? dueReviews(progress) : [];

  async function onResult(phraseId: string, e: Evaluation) {
    setFeedback((f) => ({ ...f, [phraseId]: e }));
    setProgress(await completeReview(phraseId));
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
      <ScrollView contentContainerStyle={{ padding: space(3), gap: space(2) }}>
        <Text style={type.hero}>Review</Text>
        <Text style={type.caption}>
          {due.length > 0
            ? `${due.length} phrase${due.length > 1 ? "s" : ""} need review today.`
            : "Nothing due today. Finish an episode to start your review cycle."}
        </Text>

        {due.map((item) => {
          const phrase = allPhrases.find((p) => p.id === item.phraseId);
          if (!phrase) return null;
          const fb = feedback[phrase.id];
          return (
            <Card key={phrase.id}>
              <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                <Text style={type.cardTitle}>{phrase.ko}</Text>
                <TipToggle std={phrase.std} nuance={phrase.nuance} />
              </View>
              <Text style={[type.caption, { marginBottom: space(1.5) }]}>{phrase.en}</Text>
              <View style={{ flexDirection: "row", gap: space(1), marginBottom: space(2) }}>
                <GhostButton title="▶ Listen" onPress={() => speak(phrase.ko)} />
                <GhostButton title="▶ Slow" onPress={() => speak(phrase.ko, true)} />
              </View>
              <Recorder targetText={phrase.ko} onResult={(e) => onResult(phrase.id, e)} />
              {fb && (
                <Text style={[type.caption, { marginTop: space(1), color: colors.ink }]}>
                  {fb.coachTip}
                </Text>
              )}
            </Card>
          );
        })}
      </ScrollView>
    </SafeAreaView>
  );
}
