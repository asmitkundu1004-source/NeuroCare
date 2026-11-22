// Builds docs/NeuroCare_Project_Report.docx from the research results.
// Every number, table and figure comes from research/results and research/figures, so the
// report always matches the paper and the website. Edit TEAM below, then: npm run build
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat, NumberFormat, Packer,
  PageBreak, PageNumber, Paragraph, ShadingType, Table, TableCell, TableOfContents, TableRow, TextRun, WidthType,
} from 'docx'

// ------------------------------------------------------------------ edit me
const TEAM = [
  { name: 'Asmit Kundu', id: '12023052004049' },
]
const GUIDE = { name: 'Dr. Susovan Jana', title: 'Department of Information Technology' }
const HOD = { name: '', title: 'Head of the Department, Information Technology' }
const INSTITUTE = 'Institute of Engineering & Management, Kolkata'
const DEPARTMENT = 'Department of Information Technology'
const DEGREE = 'Bachelor of Technology in Information Technology'
const YEAR = '2026'
const SITE = 'https://<your-deployment>.vercel.app'
// ------------------------------------------------------------------

const here = dirname(fileURLToPath(import.meta.url))
const repo = resolve(here, '..', '..')
const R = (file) => {
  const p = join(repo, 'research', 'results', file)
  return existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null
}
const S = R('summary.json') || {}
const AGEM = R('agem_report.json') || {}
const SAFE = R('safety_report.json') || {}
const FACE = R('aura_face_report.json') || {}
const LAT = R('latency_report.json') || {}
const PAR = R('aura_sense_onnx_parity.json') || {}
const A = S.aura_sense || {}
const f = (x, d = 3) => (x == null || Number.isNaN(x) ? '—' : Number(x).toFixed(d))
const pct = (x, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`)
const tableOf = (id) => (S.tables || []).find((t) => t.id === id)

// ------------------------------------------------------------------ building blocks

const FONT = 'Times New Roman'
const BODY = 24 // half-points: 12 pt
const children = { front: [], main: [] }
let part = 'front'
const add = (...xs) => children[part].push(...xs)
let chapter = 0
let figN = 0
let tabN = 0

function runs(text, base = {}) {
  // **bold** and _italic_ inline markup
  return text.split(/(\*\*[^*]+\*\*|_[^_]+_)/g).filter(Boolean).map((t) => {
    if (t.startsWith('**')) return new TextRun({ text: t.slice(2, -2), bold: true, ...base })
    if (t.startsWith('_') && t.endsWith('_') && t.length > 2) return new TextRun({ text: t.slice(1, -1), italics: true, ...base })
    return new TextRun({ text: t, ...base })
  })
}

const P = (text, opts = {}) => add(new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 360 }, ...opts }))
const center = (text, size = BODY, bold = false, after = 120) =>
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after }, children: [new TextRun({ text, size, bold, font: FONT })] })
const bullets = (items) => items.forEach((t) => add(new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 80, line: 340 }, children: runs(t) })))
let listInstance = 0
const numbered = (items) => {
  listInstance += 1 // each list restarts at 1
  items.forEach((t) => add(new Paragraph({ numbering: { reference: 'numbers', level: 0, instance: listInstance }, spacing: { after: 80, line: 340 }, children: runs(t) })))
}
const pageBreak = () => add(new Paragraph({ children: [new PageBreak()] }))

function H1(title) {
  chapter += 1
  figN = 0
  tabN = 0
  if (children[part].length) pageBreak() // a new section already starts on a new page
  add(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: `Chapter ${chapter}: ${title}` })] }))
}
const H2 = (t) => add(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })] }))
const H3 = (t) => add(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t })] }))
const unnumbered = (t) => {
  pageBreak()
  add(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })] }))
}

function figure(file, caption, widthIn = 6.0) {
  const path = join(repo, file)
  if (!existsSync(path)) return
  const data = readFileSync(path)
  const w = data.readUInt32BE(16)
  const h = data.readUInt32BE(20)
  const width = Math.round(widthIn * 96)
  figN += 1
  add(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: 'png', data, transformation: { width, height: Math.round((width * h) / w) } })] }))
  add(new Paragraph({ style: 'Caption', alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: `Figure ${chapter}.${figN}: ${caption}` })] }))
}

const border = { style: BorderStyle.SINGLE, size: 4, color: 'A8A29E' }
const borders = { top: border, bottom: border, left: border, right: border }

function table(columns, rows, caption, { highlight = [], widths } = {}) {
  const total = 9026 // A4 text width at 1-inch margins, in DXA
  const w = widths ? [...widths] : columns.map((_, i) => (i === 0 ? Math.round(total * 0.34) : Math.round((total * 0.66) / (columns.length - 1))))
  w[0] += total - w.reduce((a, b) => a + b, 0)
  tabN += 1
  add(new Paragraph({ style: 'Caption', alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 },
    children: [new TextRun({ text: `Table ${chapter}.${tabN}: ${caption}` })] }))
  const cell = (text, i, head, hl) => new TableCell({
    width: { size: w[i], type: WidthType.DXA }, borders,
    shading: head ? { type: ShadingType.CLEAR, fill: 'EDE8DF', color: 'auto' } : hl ? { type: ShadingType.CLEAR, fill: 'FBEDE6', color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(text), bold: head || hl, size: 19, font: FONT })] })],
  })
  add(new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, children: columns.map((c, i) => cell(c, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, highlight.includes(ri))) })),
    ],
  }))
  add(new Paragraph({ spacing: { after: 160 }, children: [] }))
}

function resultTable(id, caption) {
  const t = tableOf(id)
  if (!t) return
  table(t.columns, t.rows, caption || t.title, { highlight: t.highlight || [] })
  P(`_${t.caption}_`, { alignment: AlignmentType.LEFT })
}

// ------------------------------------------------------------------ front matter

add(center(INSTITUTE.toUpperCase(), 28, true, 60))
add(center(DEPARTMENT.toUpperCase(), 24, true, 900))
add(center('A PROJECT REPORT ON', 24, false, 200))
add(center('NeuroCare', 56, true, 120))
add(center('Closed-Loop Affective Control for a Private, On-Device Mental-Health Companion', 28, true, 700))
add(center('Submitted in partial fulfilment of the requirements for the degree of', 22, false, 60))
add(center(DEGREE, 24, true, 700))
add(center('Submitted by', 22, false, 100))
TEAM.forEach((m) => add(center(`${m.name}  (${m.id})`, 24, true, 60)))
add(center('', 22, false, 500))
add(center('Under the guidance of', 22, false, 100))
add(center(GUIDE.name, 24, true, 60))
add(center(GUIDE.title, 22, false, 900))
add(center(INSTITUTE, 22, false, 60))
add(center(YEAR, 22, true, 60))

unnumbered('Certificate')
P(`This is to certify that the project report entitled **"NeuroCare: Closed-Loop Affective Control for a Private, On-Device Mental-Health Companion"**, submitted by ${TEAM.map((m) => `**${m.name}** (${m.id})`).join(', ')} in partial fulfilment of the requirements for the award of the degree of ${DEGREE} at the ${INSTITUTE}, is a record of bona fide work carried out under my supervision and guidance during the academic year ${Number(YEAR) - 1}–${YEAR}.`)
P('To the best of my knowledge, the work reported here has not been submitted elsewhere for the award of any other degree or diploma.')
add(new Paragraph({ spacing: { before: 1400 }, children: [new TextRun({ text: '____________________________', font: FONT })] }))
add(new Paragraph({ children: [new TextRun({ text: GUIDE.name, bold: true, font: FONT })] }))
add(new Paragraph({ children: [new TextRun({ text: `Project Guide, ${GUIDE.title}`, font: FONT })] }))
add(new Paragraph({ spacing: { before: 1000 }, children: [new TextRun({ text: '____________________________', font: FONT })] }))
add(new Paragraph({ children: [new TextRun({ text: HOD.name || 'Head of the Department', bold: true, font: FONT })] }))
add(new Paragraph({ children: [new TextRun({ text: HOD.title, font: FONT })] }))

unnumbered('Declaration')
P('We declare that this report, **NeuroCare**, is our own work. Where we have used the work of others it is cited, and the datasets, pre-trained models and libraries we relied on are listed with their sources. The experiments reported were run by us, and every number in this report is produced by the evaluation scripts released with the project.')
TEAM.forEach((m) => {
  add(new Paragraph({ spacing: { before: 900 }, children: [new TextRun({ text: '____________________________', font: FONT })] }))
  add(new Paragraph({ children: [new TextRun({ text: `${m.name} (${m.id})`, font: FONT, bold: true })] }))
})

unnumbered('Acknowledgement')
P(`We are grateful to our guide, **${GUIDE.name}**, for steady guidance throughout this project, for pushing us to measure our claims rather than assert them, and for the freedom to rebuild the system when the evidence asked for it.`)
P(`We thank the ${DEPARTMENT} at the ${INSTITUTE} for the facilities and support that made this work possible, and our classmates who tried early versions of NeuroCare and told us honestly what felt wrong.`)
P('This project builds on openly shared research. We thank the authors of GoEmotions, EmoBank, the NRC VAD Lexicon, the Shreevastava–Foltz cognitive-distortion corpus, AffectNet, Sentence-Transformers, MediaPipe, ONNX Runtime, Ollama, Llama 3.2 and Kokoro for making their work available.')

unnumbered('Abstract')
P('Digital mental-health tools usually ask people to trust a remote server with their most private sentences, and conversational agents tend to treat emotion as something to display rather than something to act on. NeuroCare is a companion that runs entirely on the user\'s device and treats affect as a **control signal**.')
P(`On every turn a single forward pass of **Aura-Sense**, a 22-million-parameter encoder we trained jointly for 28-way emotion, valence and arousal, crisis risk and cognitive distortions, produces both the affect reading and the embedding used to search memory. Retrieval-preserving self-distillation keeps that embedding useful for search (STS-B Spearman ${f(A.stsb)} against ${f(A.stsb_teacher)} for the original encoder), and a circumplex consistency loss learns where each emotion sits on the valence–arousal plane; the learned map agrees with human norms the model never saw (valence r = ${f(A.circ_nrc_valence_r, 2)}). Aura-Sense reaches a macro-F1 of ${f(A.emotion_macro_f1)} on 28 emotions and an AUROC of ${f(A.risk_auroc)} for crisis risk on held-out data.`)
P(`The reading drives a three-tier safety gate that can bypass the language model, a router that chooses among six therapeutic stances, and **Affect-Gated Episodic Memory (AGEM)**, which recalls relevant memories but keeps painful ones quiet when they are not the topic. A facial channel, **Aura-Face**, classifies expression from 52 blendshape coefficients without seeing pixels (${pct(S.aura_face?.accuracy)} accuracy across 8 AffectNet classes). The system runs with a LoRA fine-tuned Llama 3.2 3B on a laptop, or entirely inside a web browser, where it can be hosted for free.`)
P('**Keywords:** affective computing, mental health, multi-task learning, retrieval-augmented generation, on-device AI, privacy, web deployment.')

unnumbered('Table of Contents')
add(new TableOfContents('Contents', { hyperlink: true, headingStyleRange: '1-2' }))

// ------------------------------------------------------------------ chapters
part = 'main'

H1('Introduction')
H2('1.1 Background')
P('Anxiety and depression are common, rising, and under-treated. The World Health Organization reports that most people who need mental-health care do not receive it, and waiting lists for therapy are measured in months. Conversational agents have been explored as a first point of contact, a place to put thoughts into words at two in the morning when no one else is available. Early evidence, including a randomised trial of the Woebot agent, suggests such tools can help with mild symptoms.')
P('Two problems recur. The first is **trust**. Most mental-health applications send everything a user writes to a remote server, where it is stored and processed by a third party. For people who already hesitate to seek help, that is often reason enough not to. The second is **behaviour**. A general-purpose language model, even one fine-tuned to sound empathetic, does not know by itself when to slow down, when to mention something said last week, or when to stop generating and direct the person to a human.')
H2('1.2 From the first version to this one')
P('The first version of NeuroCare, described in our progress report, was a local pipeline: a LoRA fine-tuned Llama 3.2 3B model served by Ollama, a ChromaDB store of past turns, Kokoro speech synthesis, and six personas chosen by hand. It worked, but three things limited it as research. Emotion was not measured at all. Memory always injected the single most similar past turn, whatever the person was feeling. Safety depended entirely on the fine-tuned model behaving well.')
P('This report describes how we rebuilt NeuroCare around one idea: what the person seems to feel should decide what the system does. We also re-measured everything with scripts released alongside the code, and where earlier figures could not be reproduced from the implementation we replaced them with numbers we can stand behind.')
H2('1.3 Problem statement')
P('To design, build and evaluate a mental-health companion that (i) runs entirely on the user\'s device, (ii) senses affect from text, and optionally voice and face, with models trained for the purpose, (iii) uses that affect to control safety, response style and memory, and (iv) can be deployed publicly at no cost while preserving the same privacy guarantees.')
H2('1.4 Objectives')
numbered([
  'Train a single compact model that reads emotion, valence and arousal, crisis risk and cognitive distortions from text, and whose embedding can also be used for memory search.',
  'Design a memory policy that recalls relevant context without resurfacing painful memories at the wrong time or repeating itself.',
  'Place a deterministic, testable safety layer in front of the language model.',
  'Add privacy-preserving facial and vocal channels and fuse them with text on one valence–arousal plane.',
  'Build a polished web application that works with a local server or entirely in the browser, on desktop and mobile.',
  'Evaluate every component on held-out data, report limitations honestly, and release code, models and scripts.',
])
H2('1.5 Contributions')
bullets([
  '**Aura-Sense**, a multi-task encoder trained with retrieval-preserving self-distillation and a learned circumplex, so one forward pass serves memory and the full affect readout.',
  '**AGEM**, Affect-Gated Episodic Memory, which ranks memories by relevance, emotional salience and recency, gates painful ones by the person\'s current state and avoids repetition.',
  '**A closed affective loop** with a hybrid lexicon and classifier safety gate and an explainable stance router.',
  '**Aura-Face**, facial affect from blendshape coefficients only, and a **zero-server deployment** whose browser runtime is tested against the server implementation.',
])
H2('1.6 Organisation of the report')
P('Chapter 2 reviews related work. Chapter 3 sets out requirements and feasibility. Chapter 4 describes the system architecture and Chapter 5 the models and algorithms. Chapter 6 covers implementation and deployment, Chapter 7 testing and results, and Chapter 8 discussion, limitations and ethics. Chapter 9 concludes and outlines future work.')

H1('Literature Review')
H2('2.1 Conversational agents for mental health')
P('Rule-based and retrieval-based chatbots such as Woebot and Wysa deliver structured techniques from cognitive behavioural therapy. A scoping review by Abd-alrazaq et al. found that most mental-health chatbots are rule-based and run in the cloud, with limited evaluation. Large language models make conversation far more natural but also less predictable, which raises the importance of an explicit safety layer.')
H2('2.2 Emotion recognition from text')
P('Two traditions coexist. **Categorical** models assign discrete labels; GoEmotions (Demszky et al., 2020) provides 58,000 Reddit comments annotated with 27 emotions plus neutral. **Dimensional** models place affect on continuous axes; EmoBank (Buechel and Hahn, 2017) provides valence, arousal and dominance ratings for 10,000 sentences. Russell\'s circumplex model connects the two by placing emotions on the valence–arousal plane, and the NRC VAD Lexicon (Mohammad, 2018) offers human ratings for 20,000 words. Most systems predict one view only; we learn the map between them.')
H2('2.3 Sentence embeddings and multi-task learning')
P('Sentence-BERT and compact distilled encoders such as MiniLM produce embeddings suitable for semantic search. Multi-task learning shares one encoder between tasks; Kendall et al. proposed weighting each task\'s loss by a learned uncertainty. Fine-tuning for new tasks risks forgetting earlier behaviour; Learning without Forgetting and knowledge distillation counter this by matching the outputs of the original network. We apply the same idea to the embedding itself.')
H2('2.4 Memory for language agents')
P('Retrieval-augmented generation (Lewis et al., 2020) adds relevant documents to a model\'s context. Agent architectures such as Generative Agents (Park et al., 2023) and MemGPT (Packer et al., 2023) rank memories by relevance, recency and importance. None considers that recalling a painful memory can itself do harm, which matters for a companion more than for a general assistant.')
H2('2.5 Risk and cognitive distortions')
P('Suicide-risk detection from social media is an active area (Coppersmith et al., 2018). The Columbia Suicide Severity Rating Scale (C-SSRS) structures ideation from passive wishes to death through intent and plans. Shreevastava and Foltz (2021) released a corpus of patient questions annotated with ten cognitive distortions from Beck\'s cognitive therapy.')
H2('2.6 Facial and vocal affect')
P('AffectNet (Mollahosseini et al., 2019) is the standard in-the-wild facial-expression dataset. The Facial Action Coding System describes expressions as combinations of muscle movements, and MediaPipe\'s face landmarker estimates 52 related blendshape coefficients in real time on consumer devices. In speech, prosody is a reliable signal of arousal but a weak one of valence (Scherer, 2003).')
H2('2.7 Research gap')
P('We found no system that (a) uses affect to control safety, response style and memory together, (b) does so with a single model whose embedding also serves retrieval, (c) protects users from inappropriate recall of painful memories, and (d) runs entirely on the user\'s device with a public, free deployment. NeuroCare addresses that gap.')

H1('Requirements and System Analysis')
H2('3.1 Functional requirements')
table(['ID', 'Requirement'], [
  ['FR1', 'Accept typed messages, and voice by hold-to-talk or tap-to-toggle.'],
  ['FR2', 'Read emotion, valence, arousal, crisis risk and cognitive distortion from each message.'],
  ['FR3', 'Optionally read facial expression and voice prosody, on the device only.'],
  ['FR4', 'Assess safety before generating any reply and show regional helplines when needed.'],
  ['FR5', 'Choose a therapeutic stance automatically, or let the user choose one, and explain the choice.'],
  ['FR6', 'Remember disclosures, recall relevant ones and hold back painful ones when they are not the topic.'],
  ['FR7', 'Stream replies and speak them aloud; allow interruption at any time.'],
  ['FR8', 'Show a mood timeline, emotion frequencies and a memory vault where items can be forgotten.'],
  ['FR9', 'Offer guided breathing exercises.'],
  ['FR10', 'Export all data as JSON and erase all data on request.'],
], 'Functional requirements', { widths: [1100, 7926] })
H2('3.2 Non-functional requirements')
bullets([
  '**Privacy:** no conversation data leaves the device; data at rest is encrypted.',
  '**Latency:** affect sensing and memory recall in tens of milliseconds; speech begins after the first generated sentence.',
  '**Footprint:** the affect model small enough to download in a browser; the full system usable on an 8 GB laptop.',
  '**Portability:** desktop and mobile browsers; free static hosting.',
  '**Accessibility and comfort:** keyboard operation, reduced-motion support, readable contrast, calm visual design.',
  '**Reproducibility:** every reported number generated by released scripts.',
])
H2('3.3 Feasibility')
P('**Technical.** Every component runs on a mid-range laptop. Aura-Sense trains in about 15 minutes on a 4 GB laptop GPU, Aura-Face in minutes, and inference needs only a CPU. **Economic.** All software is open source, the datasets are freely available for research, and the website can be hosted at no cost on Vercel, Netlify or GitHub Pages. **Operational.** The application needs no account and no installation in its browser form; the full stack needs Python, Node.js and Ollama.')
H2('3.4 Hardware and software used')
table(['Item', 'Specification'], [
  ['Development machine', 'Intel Core i5-11400H, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB), Windows 11'],
  ['Training', 'Python 3.11, PyTorch 2 (CUDA 12.6), Hugging Face Transformers, Sentence-Transformers, scikit-learn'],
  ['Inference', 'ONNX Runtime (server), onnxruntime-web (browser), MediaPipe Tasks Vision, Ollama, WebLLM'],
  ['Backend', 'FastAPI, Uvicorn, SQLite, cryptography (AES-256-GCM, scrypt)'],
  ['Frontend', 'React 19, Vite, Framer Motion, Lenis, custom WebGL shader, IndexedDB, WebCrypto'],
  ['Language model', 'Llama 3.2 3B Instruct, LoRA fine-tuned (Unsloth), Q4_K_M GGUF'],
], 'Hardware and software', { widths: [2600, 6426] })

H1('System Design')
H2('4.1 The closed affective loop')
P('Each turn passes through six stages.')
numbered([
  '**Sense.** Aura-Sense reads the message once and returns the embedding and the affect reading.',
  '**Safeguard.** The safety gate assigns a tier from the lexicon and the risk head. Tier 2 bypasses the language model.',
  '**Fuse.** The text reading is combined with optional face and voice readings on the valence–arousal plane and smoothed across turns.',
  '**Route.** The router picks a therapeutic stance from the fused state, with a stated reason.',
  '**Remember.** AGEM selects memories using the same embedding and the current valence.',
  '**Respond.** The language model streams a reply under the stance\'s prompt and sampling settings; speech starts after the first sentence.',
])
figure('docs/images/landing-loop.png', 'The closed loop as presented on the website', 6.2)
H2('4.2 Two runtimes, one design')
P('The **server runtime** (Python) runs Aura-Sense with ONNX Runtime, streams from Ollama and stores data in an encrypted SQLite file. The **browser runtime** (JavaScript) implements the same pipeline: a WordPiece tokenizer, Aura-Sense through WebAssembly, the safety gate, fusion, AGEM and the router, with data in IndexedDB under a non-extractable WebCrypto key and an optional language model on WebGPU. Both emit the same events (affect, token, sentence, done), so the interface does not care which one it is talking to. The site looks for a local server and falls back to the browser runtime automatically.')
H2('4.3 Data design')
P('The server keeps three tables in a single SQLite file. Text and embeddings are sealed with AES-256-GCM before they are written, because sentence embeddings can be partly inverted back to text.')
table(['Table', 'Columns', 'Purpose'], [
  ['memories', 'id, created, text (sealed), embedding (sealed), valence, arousal, dominant, last_turn, surfaced', 'Episodic memory for AGEM'],
  ['turns', 'id, session, created, role, text (sealed), valence, arousal, dominant, risk_tier, persona', 'Conversation history and mood timeline'],
  ['meta', 'key, value', 'Global turn counter for the refractory window'],
], 'Database schema', { widths: [1500, 5026, 2500] })
H2('4.4 Interface design')
P('The interface was designed to feel calm and trustworthy rather than clinical or playful: warm ink and ivory tones, an editorial serif for the voice of the product, a clean grotesk for controls, and monospace for anything measured. A single living element, the aura, is rendered by a small WebGL shader whose colours follow the person\'s measured valence and arousal and which swells with the voice. Pages are the landing page, the session room, Insights, Breathe, the Model Lab and the Privacy centre. Every page was checked on desktop and phone sizes, and animations respect the reduced-motion setting.')

H1('Models and Methodology')
H2('5.1 Aura-Sense')
H3('Architecture')
P('Aura-Sense starts from all-MiniLM-L6-v2, a 6-layer, 384-dimensional sentence encoder. Token states are mean-pooled and normalised into the embedding e(x). Linear heads read the scaled embedding and produce 28 emotion logits, a valence–arousal pair, a risk probability, a distortion probability and a 10-way distortion type. One forward pass returns all of them.')
H3('Training data')
table(['Task', 'Dataset', 'Train', 'Test', 'Loss'], [
  ['Emotion (28 labels)', 'GoEmotions', '43,410', '5,427', 'binary cross-entropy'],
  ['Valence and arousal', 'EmoBank', '8,062', '1,000', 'mean squared error'],
  ['Crisis risk', 'Reddit SuicideWatch/depression vs. other (subsample)', '30,000', '10,000', 'cross-entropy'],
  ['Cognitive distortion', 'Shreevastava–Foltz', '1,720', '506', 'cross-entropy'],
], 'Training data for Aura-Sense', { widths: [2000, 3026, 1100, 1100, 1800] })
H3('Retrieval-preserving self-distillation')
P('Fine-tuning rotates the embedding space, after which the same vector retrieves poorly. A frozen copy of the original encoder acts as a teacher on every batch, and the loss 1 − cos(e(x), e₀(x)) keeps the student close to it.')
H3('Learned circumplex')
P('A 28 × 2 matrix C places every emotion on the valence–arousal plane. It starts at zero and is learned only from EmoBank ratings: the predicted emotion distribution multiplied by C must match the rating. A second term ties the valence–arousal head to the discrete labels through C. Because the NRC VAD Lexicon is never used in training, comparing C with it is an independent test.')
H3('Objective and training')
P('The four supervised losses are weighted by learned uncertainty; the consistency and distillation terms have fixed weights. Weighting the consistency term by uncertainty as well was harmful: its loss is small, so its learned weight grew very large and the emotion head suffered (see the ablation in Chapter 7). Each training step samples one task in proportion to the square root of its dataset size. Training runs for 8,000 steps with AdamW and mixed precision.')
H3('Export')
P(`The model is exported to ONNX and quantised with 8-bit block-wise weights and an 8-bit embedding table, keeping activations in 32-bit floating point. Ordinary dynamic quantisation, which also quantises activations, shifted the risk probability by up to 0.65 on some sentences, which is not acceptable for a safety signal; the weight-only scheme limits the largest shift to ${f(PAR.weight_only_8bit?.max_abs_risk_prob_diff)}.`)
H2('5.2 Affect-Gated Episodic Memory')
P('Each stored memory keeps its embedding, the valence and arousal read when it was said, its age and the turn it last surfaced. For a query, each memory is scored by similarity weighted by emotional salience, plus a small recency bonus, minus a penalty if it surfaced within the last three turns. A memory is admitted only if it is relevant enough, outside that refractory window, and passes the **affect gate**: a painful memory is admitted only when the person is in a low state themselves or the query is strongly about that memory. Up to three memories are admitted; held-back candidates are reported with their reason, which the interface shows.')
H2('5.3 Safety gate')
P('Tier 2 (crisis) is raised by explicit language about ideation, intent, method or self-harm, or by the risk head above a high-precision threshold together with a distress cue. The language model is then bypassed and a fixed message with helplines for the user\'s region is shown, and the session stays at least at tier 1 for four turns. Tier 1 keeps generation but adds a safety instruction. Questions about diagnoses, doses or stopping medication add a clinical boundary. The lexicon is organised by C-SSRS categories and is the same file in both runtimes.')
table(['Region', 'Service', 'Contact'], [
  ['India', 'Tele-MANAS, Government of India (24x7, free)', '14416 or 1-800-891-4416'],
  ['India', 'AASRA (24x7)', '+91 98204 66726'],
  ['India', 'Emergency services', '112'],
  ['United States', '988 Suicide & Crisis Lifeline', 'Call or text 988'],
  ['United Kingdom', 'Samaritans (24x7, free)', '116 123'],
], 'Helplines shown by the safety gate', { widths: [1800, 4426, 2800] })
H2('5.4 Stance router')
table(['Stance', 'Framework', 'Chosen when'], [
  ['The Listener', 'Person-centred reflection', 'sadness, grief, disappointment, a low mood, or elevated risk without agitation'],
  ['The Anchor', 'Mindfulness and grounding', 'fear, nervousness, high-arousal distress, or elevated risk with agitation'],
  ['The Reframer', 'Cognitive behavioural therapy', 'a cognitive distortion is detected on a negative turn'],
  ['The Coach', 'Behavioural activation', 'the person describes feeling stuck or unmotivated'],
  ['The Caretaker', 'Self-compassion and basic needs', 'signs of exhaustion, poor sleep or skipped meals'],
  ['The Philosopher', 'Stoic resilience and meaning', 'anger, annoyance or disapproval'],
], 'The six stances and the router\'s rules', { widths: [1700, 2600, 4726] })
H2('5.5 Aura-Face')
P(`MediaPipe's face landmarker turns each camera frame into 52 blendshape coefficients inside the browser. Aura-Face sees only these numbers. They are re-expressed in a bilateral form, as the unpaired coefficients plus the mean and absolute asymmetry of each of the 20 left–right pairs, and classified by an ensemble of five small neural networks trained with mirror augmentation. Class probabilities are projected through the circumplex learned by Aura-Sense, so face and text speak the same language. Training features were extracted from 29,000 AffectNet images; ${pct(FACE.face_detection_rate?.test)} of test images yielded a face.`)
H2('5.6 Voice and fusion')
P('While the person holds the microphone, the browser measures loudness, pitch variation and speaking rate and compares them with that person\'s own running baseline, giving an arousal estimate with modest confidence. Text, face and voice readings are combined per dimension using confidence-weighted priors (text dominates valence; voice contributes most to arousal after text) and smoothed across turns, except when the safety tier rises.')
H2('5.7 The language model')
P('The voice of the system is a Llama 3.2 3B Instruct model fine-tuned with LoRA (rank 16, alpha 32) using Unsloth on empathetic dialogue, merged and quantised to a 2.3 GB Q4_K_M GGUF file and served by Ollama. The system prompt combines a base instruction (speak like a person, briefly, without lists, never diagnose), the stance, a plain-language summary of the person\'s affect, and any admitted memories. In the browser, Llama 3.2 1B or 3B can run on WebGPU through WebLLM. When no model is available, a reflective responder chooses an opening based on the dominant emotion and a follow-up question based on the stance.')

H1('Implementation and Deployment')
H2('6.1 Repository structure')
table(['Folder', 'Contents'], [
  ['backend/', 'FastAPI server, the Python runtime of every model, Aura-Sense ONNX files, safety lexicon'],
  ['web/', 'React application; src/engine holds the browser runtime; tests check parity with the server'],
  ['research/', 'training, evaluation and figure scripts; results and figures'],
  ['models/aura-llm/', 'Ollama Modelfile for the fine-tuned language model'],
  ['paper/', 'IEEE-format research paper in LaTeX, with generated tables and numbers'],
  ['docs/', 'deployment and development guides, this report and its generator'],
], 'Repository layout', { widths: [2200, 6826] })
H2('6.2 Backend')
P('The server exposes a WebSocket for conversation and REST endpoints for health, sensing, memories, the mood timeline, export and erasure. Each turn runs as a cancellable task, so a new message or an interruption stops generation immediately. If Ollama is not reachable, the server answers with the reflective responder instead of failing, and the availability check is cached so a missing model costs nothing per turn. Optional Kokoro speech is synthesised sentence by sentence while the model is still generating.')
table(['Endpoint', 'Purpose'], [
  ['GET /api/health', 'model, language-model and storage status'],
  ['POST /api/sense', 'affect reading for a piece of text (used by the Model Lab)'],
  ['GET /api/memories, DELETE /api/memories/{id}', 'list or forget memories'],
  ['GET /api/timeline?days=30', 'mood points and emotion counts'],
  ['GET /api/export, POST /api/wipe', 'export everything as JSON, or erase everything'],
  ['WS /ws', 'conversation: chat, interrupt; server sends affect, token, sentence, done'],
], 'Server interface', { widths: [3600, 5426] })
H2('6.3 Frontend')
P('The site is a React single-page application. The engine provider loads models only on pages that need them, so the landing page stays light. Smooth scrolling, scroll-linked reveals and the WebGL aura are paused off-screen and in background tabs and capped in resolution on phones. The session room supports typing, hold-to-talk or tap-to-toggle voice, Space and Escape shortcuts on desktop, optional camera sensing with a mirrored preview, and a panel (a bottom sheet on phones) that explains each answer: affect, channels, stance, safety tier, memories recalled or held back, and timing.')
H2('6.4 Privacy and security')
bullets([
  'No account, analytics, cookies or third-party requests during a session; the Privacy page counts outbound requests live.',
  'Server data sealed with AES-256-GCM using a random local key or a passphrase-derived key (scrypt).',
  'Browser data sealed with a non-extractable WebCrypto key in IndexedDB.',
  'Camera frames and audio are processed in memory and never stored or transmitted.',
  'Export and erase controls for everything stored.',
])
H2('6.5 Deployment')
P('The browser build is a static site and deploys to Vercel, Netlify, GitHub Pages or Cloudflare Pages with the configuration files in the repository. A single Dockerfile builds the site and runs the server, optionally downloading the fine-tuned model into Ollama inside the container, which makes the full stack deployable on Hugging Face Spaces. Step-by-step instructions are in docs/DEPLOYMENT.md.')
if (SITE && !SITE.includes('<')) P(`The project is live at ${SITE}.`)

H1('Testing and Results')
P('All results in this chapter are produced by the scripts in research/ on test sets the models never trained on. Thresholds were tuned on validation splits only.')
H2('7.1 Text affect: Aura-Sense')
resultTable('text', 'Aura-Sense against baselines and ablations')
P(`Plain multi-task fine-tuning gives the best emotion F1 but damages the embedding (STS-B drops sharply), which would make memory search unreliable. Self-distillation restores the embedding to ${f(A.stsb)}, slightly above the original encoder's ${f(A.stsb_teacher)}, for a small cost in emotion F1. The learned circumplex then adds an interpretable emotion map at no further cost. Weighting the circumplex term by uncertainty, by contrast, cost about ten points of macro-F1, which is why the final model uses a fixed weight.`)
figure('research/figures/fig_ablation.png', 'Emotion F1 and retrieval quality across the ablation', 6.2)
P(`The next figure compares the learned circumplex with NRC VAD human norms. The valence coordinates agree closely (r = ${f(A.circ_nrc_valence_r, 2)}); arousal agrees less (r = ${f(A.circ_nrc_arousal_r, 2)}), which matches arousal being the harder dimension in EmoBank itself.`)
figure('research/figures/fig_circumplex.png', 'Learned emotion map (left) and human norms never seen in training (right)', 6.2)
H2('7.2 Memory: AGEM')
P(`No public benchmark measures whether a companion brings up painful memories at the wrong moment, so we scripted a personal history of ${AGEM.n?.memories ?? 20} disclosures, half of them painful, and probed it with relevant, unrelated, tangentially related happy, and repeated follow-up queries.`)
resultTable('agem', 'Memory policies compared')
P(`The original NeuroCare policy recalled everything but surfaced a painful memory for ${pct(AGEM.methods?.['naive top-1 (original code)']?.isr, 0)} of unrelated or happy queries. The 0.82 threshold quoted in our progress report almost never retrieves anything with this encoder. Plain retrieval recalls well but still resurfaces pain and repeats itself. AGEM keeps recall at ${f(AGEM.methods?.['AGEM (full)']?.['recall@3'], 2)} with no inappropriate resurfacing and no repetition. The ablations show that the affect gate removes the first problem and the refractory window the second.`)
figure('research/figures/fig_agem.png', 'Recall against inappropriate resurfacing for each memory policy', 5.2)
H2('7.3 Safety gate')
P('The risk head is trained on long posts, but a companion receives one-line messages, so we tested the gate on short messages: explicit and indirect risk, ordinary distress, idioms such as "this exam is killing me", and small talk. A held-out set was written before the lexicon was extended.')
resultTable('safety', 'Safety gate on short messages')
P(`The classifier alone misses most short, explicit messages and the lexicon alone misses some indirect ones; together they catch all risk messages in the development set with no over-escalation of ordinary distress. On the held-out set the final gate reached crisis recall ${f(SAFE.heldout?.systems?.hybrid?.explicit_crisis_recall, 2)} and alert recall ${f(SAFE.heldout?.systems?.hybrid?.risk_alert_recall, 2)}, with no false alerts on idioms. Both sets are small and written by us, so these figures are optimistic.`)
figure('research/figures/fig_safety.png', 'Lexicon, classifier and hybrid gate on short messages', 6.2)
H2('7.4 Facial affect: Aura-Face')
resultTable('face', 'Facial affect from blendshape coefficients')
P(`Blendshapes alone carry real signal: zero-shot FACS rules already reach ${pct(FACE.models?.facs_rules_zero_shot?.accuracy)} accuracy, and Aura-Face reaches ${pct(S.aura_face?.accuracy)} across eight classes, where chance is 12.5%, on par with gradient boosting and small enough to run per frame in a browser. Our bilateral encoding and mirror augmentation did not improve on a plain network by a measurable margin; ensembling did. Pixel-based models score higher on AffectNet, but they see identity and background, which this design rules out by construction.`)
figure('research/figures/fig_face_confusion.png', 'Aura-Face confusion matrix on the AffectNet test split', 4.6)
H2('7.5 Runtime, export fidelity and parity')
resultTable('runtime', 'Export fidelity and CPU latency')
P(`The deployed 8-bit model is ${f(PAR.weight_only_8bit?.size_mb, 1)} MB instead of ${f(PAR.fp32?.size_mb, 1)} MB and agrees with the PyTorch model on ${pct(PAR.weight_only_8bit?.emotion_top1_agreement)} of top emotions. It is slower on a laptop CPU than the full-precision graph because of block-wise dequantisation, which we accept because a few tens of milliseconds are negligible next to generation and the smaller file is what makes browser use and free hosting practical. AGEM ranks 1,000 memories in about ${f(LAT.agem?.['1000']?.median_ms, 1)} ms and 10,000 in about ${f(LAT.agem?.['10000']?.median_ms, 1)} ms.`)
P('An automated test suite checks that the JavaScript tokenizer produces exactly the same token ids as the Python tokenizer, including accented, Chinese and punctuated text, and that the browser runtime reproduces the server\'s outputs to within one thousandth. Both tests pass, and they run on every deployment to GitHub Pages.')
H2('7.6 Functional and interface testing')
table(['Scenario', 'Expected', 'Result'], [
  ['Send a message in the browser runtime', 'affect panel fills; stance chosen with reason; reply shown', 'Pass'],
  ['Mention an earlier topic after reloading the page', 'AGEM recalls the encrypted memory', 'Pass'],
  ['Type a crisis phrase', 'tier 2; language model bypassed; helplines for the region shown', 'Pass'],
  ['Server without Ollama', 'reflective responder answers immediately', 'Pass'],
  ['Interrupt while speaking', 'audio stops; generation cancelled', 'Pass'],
  ['Phone-sized screen (390 × 844)', 'no horizontal scrolling; controls reachable; sheets usable', 'Pass'],
  ['Privacy page during a session', 'zero requests to other servers', 'Pass'],
], 'Functional test summary', { widths: [3200, 4326, 1500] })

H1('Discussion, Limitations and Ethics')
H2('8.1 What the results show')
P('The main lesson is that the expensive parts of an affect-aware companion can share one small model if the model is trained to keep its embedding intact. That single pass is what makes it practical to sense affect on every turn, in a browser, before deciding anything else. The second lesson is that memory needs a notion of harm: plain retrieval is good at recall and indifferent to timing, and a person on a good day does not need to be reminded of their worst week.')
H2('8.2 Limitations')
bullets([
  'NeuroCare has not been evaluated with users. A controlled study of perceived empathy, usability, trust and safety is the necessary next step.',
  'The risk head is trained on Reddit communities whose labels reflect where a post was made, not clinical assessment.',
  'The memory and safety probes are small and author-written; they demonstrate behaviour rather than estimate population performance.',
  'AffectNet labels are noisy, a facial expression is not an emotion, and the model has not been tested for bias across demographic groups.',
  'Voice transcription uses the browser\'s speech recogniser, which runs on the device in recent Chrome versions but may use the browser vendor\'s service elsewhere; typing is always local.',
  'The router\'s rules reflect our reading of the therapeutic literature and should be reviewed by clinicians.',
])
H2('8.3 Ethical considerations')
P('NeuroCare is a support tool, not a therapist or a medical device, and it says so. It never diagnoses or advises on medication. When risk is detected it points to people rather than trying to manage the situation itself. All data stay on the user\'s device, encrypted, and can be exported or erased at any time. Datasets were used under their research licences, and no new personal data were collected for this project.')
H2('8.4 A note on the progress report')
P('Some figures in our earlier progress report, such as persona-adherence scores and speech-recognition error rates, were not produced by scripts we can release, and the memory threshold it described turns out to retrieve almost nothing with the current encoder. This report replaces them with measurements that anyone can reproduce from the repository.')

H1('Conclusion and Future Work')
H2('9.1 Conclusion')
P('NeuroCare shows that a mental-health companion can be private by construction and still behave thoughtfully. One multi-task encoder reads how a person seems to feel, and that reading decides when the system steps aside for a human, which stance it takes, and which memories it brings up. The models are small enough to run in a browser, the whole system can be hosted for free, and every claim in this report can be checked against released code.')
H2('9.2 Future work')
bullets([
  'A user study with ethics approval, measuring the System Usability Scale, perceived empathy and safety.',
  'Fully on-device speech recognition (for example a small Whisper model) to remove the remaining dependency on the browser\'s recogniser.',
  'A learned router, trained from clinician-labelled examples, with the current rules as a fallback.',
  'Wearable signals such as heart-rate variability as a fourth affect channel.',
  'End-to-end encrypted synchronisation of the vault between a person\'s own devices.',
  'Support for Indian languages in sensing and safety.',
])

unnumbered('References')
const REFS = [
  'World Health Organization, World Mental Health Report: Transforming Mental Health for All. Geneva: WHO, 2022.',
  'A. A. Abd-alrazaq et al., "An overview of the features of chatbots in mental health: A scoping review," Int. J. Med. Inform., vol. 132, 2019.',
  'K. K. Fitzpatrick, A. Darcy and M. Vierhile, "Delivering cognitive behavior therapy to young adults with symptoms of depression and anxiety using a fully automated conversational agent (Woebot)," JMIR Mental Health, vol. 4, no. 2, 2017.',
  'J. A. Russell, "A circumplex model of affect," J. Personality and Social Psychology, vol. 39, no. 6, pp. 1161–1178, 1980.',
  'D. Demszky et al., "GoEmotions: A dataset of fine-grained emotions," in Proc. ACL, 2020.',
  'S. Buechel and U. Hahn, "EmoBank: Studying the impact of annotation perspective and representation format on dimensional emotion analysis," in Proc. EACL, 2017.',
  'S. M. Mohammad, "Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words," in Proc. ACL, 2018.',
  'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. EMNLP, 2019.',
  'W. Wang et al., "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," in Proc. NeurIPS, 2020.',
  'A. Kendall, Y. Gal and R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," in Proc. CVPR, 2018.',
  'Z. Li and D. Hoiem, "Learning without forgetting," IEEE Trans. PAMI, vol. 40, no. 12, 2018.',
  'G. Hinton, O. Vinyals and J. Dean, "Distilling the knowledge in a neural network," arXiv:1503.02531, 2015.',
  'P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Proc. NeurIPS, 2020.',
  'J. S. Park et al., "Generative agents: Interactive simulacra of human behavior," in Proc. UIST, 2023.',
  'C. Packer et al., "MemGPT: Towards LLMs as operating systems," arXiv:2310.08560, 2023.',
  'J. X. Morris et al., "Text embeddings reveal (almost) as much as text," in Proc. EMNLP, 2023.',
  'S. Shreevastava and P. Foltz, "Detecting cognitive distortions from patient-therapist interactions," in Proc. CLPsych Workshop, 2021.',
  'G. Coppersmith et al., "Natural language processing of social media as screening for suicide risk," Biomedical Informatics Insights, vol. 10, 2018.',
  'K. Posner et al., "The Columbia–Suicide Severity Rating Scale," American Journal of Psychiatry, vol. 168, no. 12, 2011.',
  'N. Komati, "Suicide and Depression Detection," Kaggle dataset, 2021.',
  'A. Mollahosseini, B. Hasani and M. H. Mahoor, "AffectNet," IEEE Trans. Affective Computing, vol. 10, no. 1, 2019.',
  'P. Ekman and W. V. Friesen, Facial Action Coding System. Consulting Psychologists Press, 1978.',
  'C. Lugaresi et al., "MediaPipe: A framework for building perception pipelines," arXiv:1906.08172, 2019.',
  'K. R. Scherer, "Vocal communication of emotion: A review of research paradigms," Speech Communication, vol. 40, 2003.',
  'A. Grattafiori et al., "The Llama 3 herd of models," arXiv:2407.21783, 2024.',
  'E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in Proc. ICLR, 2022.',
  'C. F. Ruan et al., "WebLLM: A high-performance in-browser LLM inference engine," arXiv:2412.15803, 2024.',
  'A. T. Beck, A. J. Rush, B. F. Shaw and G. Emery, Cognitive Therapy of Depression. Guilford Press, 1979.',
  'R. W. Picard, Affective Computing. MIT Press, 1997.',
]
REFS.forEach((r, i) => add(new Paragraph({ spacing: { after: 100, line: 300 }, indent: { left: 560, hanging: 560 },
  children: [new TextRun({ text: `[${i + 1}]\t${r}`, size: 22 })] })))

unnumbered('Appendix A: Screenshots')
chapter = 'A'
figN = 0
;[
  ['docs/images/landing.png', 'Landing page'],
  ['docs/images/session.png', 'Session room with the explanation panel'],
  ['docs/images/lab.png', 'Model Lab: Aura-Sense live analysis'],
  ['docs/images/insights.png', 'Insights: mood timeline and memory vault'],
  ['docs/images/mobile.png', 'Session room on a phone'],
].forEach(([p, c]) => figure(p, c, p.includes('mobile') ? 2.6 : 6.2))

unnumbered('Appendix B: Running and Deploying NeuroCare')
P('**In the browser only.** Open the deployed site, or run npm install and npm run dev inside web/. Everything works without a server; replies come from the reflective responder until a model is loaded in the tab.')
P('**Full stack on a laptop.** Copy the fine-tuned GGUF into models/aura-llm/, run ollama create aura -f Modelfile, install backend/requirements.txt, start uvicorn neurocare.server:app --port 8000, then run the site with npm run dev. The site finds the server on its own.')
P('**Free hosting.** Import the repository into Vercel with root directory web, or into Netlify with no settings, or enable GitHub Pages with the included workflow. For the full stack, create a Docker Space on Hugging Face and set AURA_GGUF_URL to the model\'s download link.')
P('**Reproducing the results.** Run the scripts in research/ in the order listed in docs/DEVELOPMENT.md, then python make_summary.py. This regenerates every table, figure and number used in the website, the paper and this report.')

// ------------------------------------------------------------------ document

const PAGE = { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } }
const footer = () => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] })

const doc = new Document({
  creator: TEAM.map((m) => m.name).join(', '),
  title: 'NeuroCare Project Report',
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONT, size: 32, bold: true }, paragraph: { spacing: { before: 240, after: 360 }, keepNext: true, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 300, after: 140 }, keepNext: true, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 2 } },
      { id: 'Caption', name: 'Caption', basedOn: 'Normal', next: 'Normal', run: { font: FONT, size: 20, italics: true } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: 'numbers', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [
    { properties: { page: { ...PAGE, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } }, footers: { default: footer() }, children: children.front },
    { properties: { page: { ...PAGE, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } }, footers: { default: footer() }, children: children.main },
  ],
})

const out = join(repo, 'docs', 'NeuroCare_Project_Report.docx')
writeFileSync(out, await Packer.toBuffer(doc))
console.log(`wrote ${out}`)
