// MVP content — Busan only. 5 situations from mvp_plan.md §10.
// Phrase audio: device TTS placeholder until Audio Lab native recordings land.

export type Phrase = {
  id: string;
  ko: string; // surface_text (Busan dialect)
  en: string; // meaning
  std: string; // standard Korean (shown only in the yellow tip sheet)
  nuance: string;
};

export type Situation = {
  id: string;
  episode: number;
  title: string;
  titleKo: string;
  goal: string;
  npcName: string;
  npcGreeting: Phrase; // opening NPC line for the guided-response step
  phrases: Phrase[];
};

export const SITUATIONS: Situation[] = [
  {
    id: "friends",
    episode: 1,
    title: "Meeting a Friend at Busan Station",
    titleKo: "부산역에서 친구 만나기",
    goal: "Understand a Busan friend's basic questions and answer back.",
    npcName: "Minjun",
    npcGreeting: {
      id: "friends-npc",
      ko: "니 지금 뭐 하노?",
      en: "What are you doing right now?",
      std: "너 지금 뭐 해?",
      nuance: "Casual question between close friends.",
    },
    phrases: [
      { id: "f1", ko: "뭐 하노?", en: "What are you doing?", std: "뭐 해?", nuance: "The -노 ending marks a friendly open question." },
      { id: "f2", ko: "어데 가노?", en: "Where are you going?", std: "어디 가?", nuance: "어데 = 어디. Very common in Busan." },
      { id: "f3", ko: "밥 뭇나?", en: "Did you eat?", std: "밥 먹었어?", nuance: "The classic Korean greeting, Busan style." },
      { id: "f4", ko: "와 이라노?", en: "Why are you like this?", std: "왜 이래?", nuance: "Playful or annoyed depending on tone." },
      { id: "f5", ko: "참말이가?", en: "Really?", std: "정말이야?", nuance: "Surprised confirmation." },
    ],
  },
  {
    id: "restaurant",
    episode: 2,
    title: "Ordering Pork Soup",
    titleKo: "돼지국밥 주문하기",
    goal: "Order food in a Busan restaurant.",
    npcName: "Imo (auntie)",
    npcGreeting: {
      id: "rest-npc",
      ko: "뭐 드릴까예?",
      en: "What can I get you?",
      std: "뭐 드릴까요?",
      nuance: "-예 is the soft Busan polite ending.",
    },
    phrases: [
      { id: "r1", ko: "이모, 국밥 하나 주이소.", en: "Auntie, one gukbap please.", std: "이모, 국밥 하나 주세요.", nuance: "주이소 = 주세요. Warm and polite." },
      { id: "r2", ko: "물 좀 주이소.", en: "Some water, please.", std: "물 좀 주세요.", nuance: "Everyday polite request." },
      { id: "r3", ko: "이거 얼마고?", en: "How much is this?", std: "이거 얼마야?", nuance: "-고 ending for casual questions." },
      { id: "r4", ko: "맵게 해주이소.", en: "Make it spicy, please.", std: "맵게 해주세요.", nuance: "Customize your order like a local." },
      { id: "r5", ko: "잘 묵겠심더.", en: "Thanks for the meal (I'll eat well).", std: "잘 먹겠습니다.", nuance: "-심더 = -습니다. Signature Busan formal ending." },
    ],
  },
  {
    id: "market",
    episode: 3,
    title: "Haggling at Jagalchi Market",
    titleKo: "시장에서 말하기",
    goal: "Ask prices and react like a local at the market.",
    npcName: "Vendor",
    npcGreeting: {
      id: "mkt-npc",
      ko: "학생, 이거 하나 묵어봐라.",
      en: "Hey, try one of these.",
      std: "학생, 이거 하나 먹어봐요.",
      nuance: "Market vendors speak warmly and directly.",
    },
    phrases: [
      { id: "m1", ko: "이거 얼마고?", en: "How much is this?", std: "이거 얼마예요?", nuance: "Your most-used market phrase." },
      { id: "m2", ko: "좀 깎아주이소.", en: "Please give me a discount.", std: "좀 깎아주세요.", nuance: "Haggling is expected — say it with a smile." },
      { id: "m3", ko: "싱싱하네예.", en: "It's so fresh.", std: "싱싱하네요.", nuance: "Compliment the goods before you buy." },
      { id: "m4", ko: "하나만 주이소.", en: "Just one, please.", std: "하나만 주세요.", nuance: "Simple and polite." },
      { id: "m5", ko: "많이 파이소.", en: "Sell a lot! (goodbye to a vendor)", std: "많이 파세요.", nuance: "The friendly way to leave a market stall." },
    ],
  },
  {
    id: "directions",
    episode: 4,
    title: "Finding Haeundae",
    titleKo: "길 묻기",
    goal: "Ask for directions in Busan.",
    npcName: "Passerby",
    npcGreeting: {
      id: "dir-npc",
      ko: "어데 찾아가는교?",
      en: "Where are you trying to go?",
      std: "어디 찾아가세요?",
      nuance: "-는교 is a polite Busan question ending.",
    },
    phrases: [
      { id: "d1", ko: "해운대 어데로 가면 됩니꺼?", en: "How do I get to Haeundae?", std: "해운대 어디로 가면 돼요?", nuance: "-니꺼 = -니까/나요. Polite question." },
      { id: "d2", ko: "여서 멉니꺼?", en: "Is it far from here?", std: "여기서 멀어요?", nuance: "여서 = 여기서." },
      { id: "d3", ko: "지하철역 어데 있습니꺼?", en: "Where is the subway station?", std: "지하철역 어디 있어요?", nuance: "Essential travel phrase." },
      { id: "d4", ko: "이 버스 맞습니꺼?", en: "Is this the right bus?", std: "이 버스 맞아요?", nuance: "Confirm before you hop on." },
      { id: "d5", ko: "감사합니다.", en: "Thank you.", std: "감사합니다.", nuance: "Same as standard — politeness is universal." },
    ],
  },
  {
    id: "intro",
    episode: 5,
    title: "Introducing Yourself",
    titleKo: "자기소개",
    goal: "Introduce yourself and respond to new people.",
    npcName: "New friend",
    npcGreeting: {
      id: "intro-npc",
      ko: "어데서 왔는교?",
      en: "Where are you from?",
      std: "어디에서 왔어요?",
      nuance: "The first question every traveler hears.",
    },
    phrases: [
      { id: "i1", ko: "저는 미국에서 왔심더.", en: "I'm from the U.S.", std: "저는 미국에서 왔습니다.", nuance: "-심더 makes you sound instantly local." },
      { id: "i2", ko: "부산은 처음입니더.", en: "It's my first time in Busan.", std: "부산은 처음입니다.", nuance: "-ㅂ니더 = -ㅂ니다." },
      { id: "i3", ko: "한국어 배우고 있심더.", en: "I'm learning Korean.", std: "한국어 배우고 있습니다.", nuance: "Guaranteed to earn a smile." },
      { id: "i4", ko: "만나서 반갑심더.", en: "Nice to meet you.", std: "만나서 반갑습니다.", nuance: "Warm Busan-style greeting." },
      { id: "i5", ko: "천천히 말해주이소.", en: "Please speak slowly.", std: "천천히 말해주세요.", nuance: "Your survival phrase — use it freely." },
    ],
  },
];

export const getSituation = (id: string) => SITUATIONS.find((s) => s.id === id);
