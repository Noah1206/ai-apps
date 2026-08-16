import { useCallback, useState } from "react";
import { Pressable, ScrollView, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useFocusEffect, useRouter } from "expo-router";
import { Card } from "../../components/ui";
import { SITUATIONS } from "../../lib/content";
import { getProgress, Progress } from "../../lib/progress";
import { colors, space, type } from "../../lib/theme";

export default function Path() {
  const router = useRouter();
  const [progress, setProgress] = useState<Progress | null>(null);

  useFocusEffect(
    useCallback(() => {
      getProgress().then(setProgress);
    }, []),
  );

  const completed = progress?.completed ?? [];
  const currentIdx = SITUATIONS.findIndex((s) => !completed.includes(s.id));

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
      <ScrollView contentContainerStyle={{ padding: space(3), gap: space(2) }}>
        <Text style={type.hero}>Busan Story</Text>
        <Text style={type.caption}>Unlock life in Busan, one situation at a time.</Text>

        {SITUATIONS.map((s, i) => {
          const done = completed.includes(s.id);
          const state = done ? "Mastered" : i === currentIdx || currentIdx === -1 ? "Recommended" : "Locked";
          const locked = state === "Locked";
          return (
            <Pressable
              key={s.id}
              disabled={locked}
              onPress={() => router.push(`/mission/${s.id}`)}
              style={({ pressed }) => pressed && { opacity: 0.8 }}
            >
              <Card style={locked ? { opacity: 0.45 } : undefined}>
                <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                  <Text style={type.caption}>Episode {s.episode}</Text>
                  <Text style={[type.caption, state === "Mastered" && { color: colors.ink }]}>
                    {state === "Mastered" ? "✓ Mastered" : state}
                  </Text>
                </View>
                <Text style={[type.cardTitle, { marginTop: 4 }]}>{s.title}</Text>
                <Text style={type.caption}>{s.titleKo}</Text>
              </Card>
            </Pressable>
          );
        })}

        <Card style={{ opacity: 0.45 }}>
          <Text style={type.cardTitle}>Jeju · Jeolla · Chungcheong</Text>
          <Text style={type.caption}>Coming soon</Text>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}
