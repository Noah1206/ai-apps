// Design tokens — Lovable design system (see docs/app_mvp_ux_plan.md §6)
export const colors = {
  bg: "#f7f4ed",
  ink: "#1c1c1c",
  ink83: "rgba(28,28,28,0.83)",
  muted: "#5f5f5d",
  border: "#eceae4",
  borderStrong: "rgba(28,28,28,0.4)",
  onDark: "#fcfbf8",
  tip: "#f2c744", // the single chromatic exception: 표준어 comparison tip icon
};

export const radius = { sm: 6, md: 8, card: 12, lg: 16, pill: 9999 };

export const space = (n: number) => n * 8;

export const type = {
  hero: { fontSize: 36, fontWeight: "600" as const, letterSpacing: -0.9, color: colors.ink },
  heading: { fontSize: 24, fontWeight: "600" as const, letterSpacing: -0.5, color: colors.ink },
  cardTitle: { fontSize: 20, fontWeight: "400" as const, color: colors.ink },
  body: { fontSize: 16, fontWeight: "400" as const, color: colors.ink83, lineHeight: 24 },
  caption: { fontSize: 14, fontWeight: "400" as const, color: colors.muted, lineHeight: 21 },
};
