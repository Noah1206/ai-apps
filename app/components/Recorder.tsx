import React, { useState } from "react";
import { Alert, Pressable, StyleSheet, Text, View } from "react-native";
import { startRecording, stopRecording } from "../lib/audio";
import { engine, Evaluation } from "../lib/feedback";
import { colors, radius, type } from "../lib/theme";

// Hold-free tap-to-record pill: tap → recording, tap again → evaluate.
export function Recorder({
  targetText,
  onResult,
}: {
  targetText: string;
  onResult: (e: Evaluation) => void;
}) {
  const [state, setState] = useState<"idle" | "recording" | "scoring">("idle");

  async function toggle() {
    if (state === "idle") {
      const ok = await startRecording();
      if (!ok) {
        Alert.alert("Microphone needed", "Enable microphone access to practice speaking.");
        return;
      }
      setState("recording");
    } else if (state === "recording") {
      setState("scoring");
      const rec = await stopRecording();
      if (rec) {
        const evaluation = await engine.evaluate({
          targetText,
          audioUri: rec.uri,
          durationMs: rec.durationMs,
        });
        onResult(evaluation);
      }
      setState("idle");
    }
  }

  return (
    <View style={{ alignItems: "center" }}>
      <Pressable
        onPress={toggle}
        accessibilityLabel={state === "recording" ? "Stop recording" : "Start recording"}
        style={({ pressed }) => [
          styles.pill,
          state === "recording" && styles.recording,
          pressed && { opacity: 0.8 },
        ]}
      >
        <Text style={{ color: state === "recording" ? colors.onDark : colors.ink, fontSize: 28 }}>
          {state === "recording" ? "■" : "🎙"}
        </Text>
      </Pressable>
      <Text style={[type.caption, { marginTop: 8 }]}>
        {state === "idle" ? "Tap to speak" : state === "recording" ? "Listening… tap to finish" : "Scoring…"}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    width: 72,
    height: 72,
    borderRadius: radius.pill,
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.borderStrong,
    alignItems: "center",
    justifyContent: "center",
  },
  recording: { backgroundColor: colors.ink, borderColor: colors.ink },
});
