import { useCallback, useState } from "react";
import { ScrollView, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useFocusEffect, useRouter } from "expo-router";
import { Card, DarkButton } from "../../components/ui";
import { SITUATIONS } from "../../lib/content";
import { dueReviews, getProgress, Progress } from "../../lib/progress";
import { colors, space, type } from "../../lib/theme";

export default function Home() {
  const router = useRouter();
  const [progress, setProgress] = useState<Progress | null>(null);

  useFocusEffect(
    useCallback(() => {
      getProgress().then(setProgress);
    }, []),
  );

  const next = SITUATIONS.find((s) => !progress?.completed.includes(s.id)) ?? SITUATIONS[0];
  const due = progress ? dueReviews(progress).length : 0;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
      <ScrollView contentContainerStyle={{ padding: space(3), gap: space(2) }}>
        <Text style={type.hero}>Today's Busan{"\n"}Training</Text>
        <Text style={type.caption}>
          Learn how people actually speak in Busan. Listen, speak, respond.
        </Text>

        <Card>
          <Text style={type.caption}>Episode {next.episode}</Text>
          <Text style={[type.cardTitle, { marginVertical: 6 }]}>{next.title}</Text>
          <Text style={[type.caption, { marginBottom: space(2) }]}>{next.goal}</Text>
          <DarkButton title="Continue" onPress={() => router.push(`/mission/${next.id}`)} />
        </Card>

        <View style={{ flexDirection: "row", gap: space(2) }}>
          <Card style={{ flex: 1 }}>
            <Text style={type.heading}>{due}</Text>
            <Text style={type.caption}>phrases to review</Text>
          </Card>
          <Card style={{ flex: 1 }}>
            <Text style={type.heading}>{progress?.streak ?? 0}</Text>
            <Text style={type.caption}>day streak</Text>
          </Card>
        </View>

        <Card>
          <Text style={type.caption}>
            Completed: {progress?.completed.length ?? 0} / {SITUATIONS.length} situations
          </Text>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}
