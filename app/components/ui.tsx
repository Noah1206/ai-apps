import React, { useState } from "react";
import { Pressable, Text, View, ViewStyle, StyleSheet } from "react-native";
import { colors, radius, space, type } from "../lib/theme";

export function Card({ children, style }: { children: React.ReactNode; style?: ViewStyle }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function DarkButton({ title, onPress, disabled }: { title: string; onPress: () => void; disabled?: boolean }) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      style={({ pressed }) => [styles.dark, (pressed || disabled) && { opacity: 0.8 }]}
    >
      <Text style={{ color: colors.onDark, fontSize: 16 }}>{title}</Text>
    </Pressable>
  );
}

export function GhostButton({ title, onPress }: { title: string; onPress: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      style={({ pressed }) => [styles.ghost, pressed && { opacity: 0.8 }]}
    >
      <Text style={{ color: colors.ink, fontSize: 14 }}>{title}</Text>
    </Pressable>
  );
}

export function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <View style={{ marginBottom: space(1.5) }}>
      <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 4 }}>
        <Text style={type.caption}>{label}</Text>
        <Text style={[type.caption, { color: colors.ink }]}>{value}</Text>
      </View>
      <View style={styles.track}>
        <View style={[styles.fill, { width: `${value}%` }]} />
      </View>
    </View>
  );
}

// The small yellow icon → standard-Korean comparison tip (Main Contents doc)
export function TipToggle({ std, nuance }: { std: string; nuance: string }) {
  const [open, setOpen] = useState(false);
  return (
    <View>
      <Pressable
        onPress={() => setOpen(!open)}
        accessibilityLabel="Standard Korean tip"
        style={styles.tipDot}
      >
        <Text style={{ fontSize: 11, color: colors.ink }}>표</Text>
      </Pressable>
      {open && (
        <Card style={{ position: "absolute", right: 0, top: 30, width: 240, zIndex: 10 }}>
          <Text style={[type.caption, { color: colors.ink }]}>Standard: {std}</Text>
          <Text style={type.caption}>{nuance}</Text>
        </Card>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.card,
    padding: space(2),
  },
  dark: {
    backgroundColor: colors.ink,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: radius.sm,
    alignItems: "center",
    // inset-shadow signature approximated within RN's shadow model
    borderTopWidth: 0.5,
    borderTopColor: "rgba(255,255,255,0.2)",
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 2,
    shadowOffset: { width: 0, height: 1 },
  },
  ghost: {
    borderWidth: 1,
    borderColor: colors.borderStrong,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: radius.sm,
    alignItems: "center",
  },
  track: {
    height: 8,
    borderRadius: radius.pill,
    backgroundColor: "rgba(28,28,28,0.06)",
    overflow: "hidden",
  },
  fill: { height: 8, borderRadius: radius.pill, backgroundColor: colors.ink },
  tipDot: {
    width: 24,
    height: 24,
    borderRadius: radius.pill,
    backgroundColor: colors.tip,
    alignItems: "center",
    justifyContent: "center",
  },
});
