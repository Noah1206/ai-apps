import { Tabs } from "expo-router";
import { Text } from "react-native";
import { colors } from "../../lib/theme";

const icon = (glyph: string) => ({ focused }: { focused: boolean }) => (
  <Text style={{ fontSize: 18, opacity: focused ? 1 : 0.4 }}>{glyph}</Text>
);

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: colors.ink,
        tabBarInactiveTintColor: colors.muted,
        tabBarStyle: { backgroundColor: colors.bg, borderTopColor: colors.border },
        sceneStyle: { backgroundColor: colors.bg },
      }}
    >
      <Tabs.Screen name="index" options={{ title: "Home", tabBarIcon: icon("⌂") }} />
      <Tabs.Screen name="path" options={{ title: "Busan", tabBarIcon: icon("🗺") }} />
      <Tabs.Screen name="review" options={{ title: "Review", tabBarIcon: icon("↻") }} />
    </Tabs>
  );
}
