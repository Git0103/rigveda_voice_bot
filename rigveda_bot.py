import argparse
import json
import os
import sys
import threading
import time

from pathlib import Path

# --- Optional imports (attempt to import; if missing, we degrade gracefully) ---
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.panel import Panel
except Exception:
    Console = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except Exception:
    plt = None

# HuggingFace model
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None

# Sentence transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None

# ----------------- Configuration -----------------
MODEL_NAME = "gyanai/sanskrit-139M-hf"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = Path("data/rigveda_corpus.jsonl")  # expected: JSON lines with {"ref":..., "sanskrit":..., "translit":..., "translation":...}
CONSOLE = Console() if Console else None

# ----------------- Utility functions -----------------

def safe_print(*args, **kwargs):
    if CONSOLE:
        CONSOLE.print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def load_corpus(path: Path):
    """Load corpus from JSONL. If not found, return a tiny demo corpus.
    Each JSON line should contain: ref, sanskrit, translit (optional), translation (optional)
    """
    items = []
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    items.append(obj)
                except Exception:
                    continue
    else:
        # demo fallback
        items = [
            {
                "ref": "MandalA 1, Sukta 1, RV 1.1",
                "sanskrit": "अग्निमीळे पुरोहितं",
                "translit": "agni mILe purohitaM",
                "translation": "I invoke Agni, the household priest"
            },
            {
                "ref": "MandalA 1, Sukta 1, RV 1.2",
                "sanskrit": "यज्ञस्य देवतम्",
                "translit": "yaj~nasya devatam",
                "translation": "The deity of the sacrifice"
            }
        ]
    return items


# ----------------- Semantic Search -----------------
class RigvedaSearch:
    def __init__(self, corpus):
        self.corpus = corpus
        self.emb_model = None
        self.corpus_embeddings = None
        if SentenceTransformer:
            try:
                self.emb_model = SentenceTransformer(EMBED_MODEL)
                texts = [ (c.get("sanskrit") or c.get("translation") or "") for c in corpus ]
                self.corpus_embeddings = self.emb_model.encode(texts, convert_to_tensor=True)
                safe_print("[green]Embedding model loaded for semantic search.[/green]")
            except Exception as e:
                safe_print("[yellow]Embedding model unavailable; semantic search disabled.[/yellow]", e)
        else:
            safe_print("[yellow]sentence-transformers not installed; semantic search disabled.[/yellow]")

    def search(self, query, top_k=3):
        if not self.emb_model or self.corpus_embeddings is None:
            # naive substring match
            hits = [c for c in self.corpus if query.strip().lower() in (c.get("sanskrit","") + c.get("translation","") + c.get("translit","")).lower()]
            return hits[:top_k]
        q_emb = self.emb_model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, self.corpus_embeddings, top_k=top_k)[0]
        results = []
        for h in hits:
            idx = h["corpus_id"]
            score = h["score"]
            obj = self.corpus[idx].copy()
            obj["score"] = float(score)
            results.append(obj)
        return results

# ----------------- Sanskrit LM integration (optional) -----------------
class SanskritLM:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        if AutoTokenizer is None:
            safe_print("[yellow]transformers not installed; Sanskrit LM disabled.[/yellow]")
            return
        try:
            safe_print(f"[cyan]Loading Sanskrit LM: {MODEL_NAME} (this may take a while)...[/cyan]")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            safe_print("[green]Sanskrit LM loaded.[/green]")
        except Exception as e:
            safe_print("[red]Failed to load Sanskrit LM. Continuing without LM.[/red]", e)

    def generate(self, prompt, max_length=128):
        if not self.model:
            return None
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_length, do_sample=True, top_p=0.95, temperature=0.8)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------- TTS -----------------
class TTS:
    def __init__(self):
        self.engine = None
        if pyttsx3:
            try:
                self.engine = pyttsx3.init()
            except Exception:
                self.engine = None

    def speak(self, text):
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass
        else:
            safe_print("[yellow]TTS not available on this system.[/yellow]")

# ----------------- Voice input -----------------
class VoiceIn:
    def __init__(self):
        self.recognizer = None
        if sr:
            try:
                self.recognizer = sr.Recognizer()
            except Exception:
                self.recognizer = None

    def listen(self, prompt_text="Listening (silence to end)..."):
        if not self.recognizer:
            safe_print("[yellow]SpeechRecognition not installed or unavailable.[/yellow]")
            return None
        with sr.Microphone() as source:
            safe_print(prompt_text)
            audio = self.recognizer.listen(source, phrase_time_limit=8)
        try:
            # Recognize with Google by default; language="sa" may not be supported; user may prefer "en-IN" for English
            text = self.recognizer.recognize_google(audio, language="en-US")
            return text
        except Exception as e:
            safe_print("[red]Could not understand audio.[/red]", e)
            return None

# ----------------- Visualization -----------------

def visualize_mantra(text: str):
    """Simple visualization: show characters/syllables and animate highlighting."""
    if plt is None:
        safe_print("[yellow]matplotlib not installed; visualization unavailable.[/yellow]")
        return
    # Prepare characters (split grapheme-wise naively)
    chars = list(text.strip())
    fig, ax = plt.subplots(figsize=(min(12, max(6, len(chars)/2)), 2))
    ax.set_axis_off()
    texts = []
    for i, ch in enumerate(chars):
        t = ax.text(i*0.6, 0.5, ch, fontsize=24, ha='center', va='center')
        texts.append(t)
    ax.set_xlim(-0.5, max(2, len(chars)*0.6))
    ax.set_ylim(0,1)

    def animate(frame):
        i = frame % len(texts)
        for j, t in enumerate(texts):
            t.set_color('black')
            t.set_fontweight('normal')
            t.set_fontsize(22)
        texts[i].set_color('crimson')
        texts[i].set_fontweight('bold')
        texts[i].set_fontsize(28)
        return texts

    ani = animation.FuncAnimation(fig, animate, frames=len(texts)*4, interval=300, blit=False)
    plt.show()

# ----------------- Main CLI -----------------

def render_result(obj):
    # obj expected to have: ref, sanskrit, translit, translation
    ref = obj.get('ref', 'unknown')
    sanskrit = obj.get('sanskrit', '')
    translit = obj.get('translit', '')
    translation = obj.get('translation', '')
    score = obj.get('score', None)

    header = f"{ref}"
    body_lines = []
    if sanskrit:
        body_lines.append(f"Sanskrit: {sanskrit}")
    if translit:
        body_lines.append(f"Translit: {translit}")
    if translation:
        body_lines.append(f"Translation: {translation}")
    if score is not None:
        body_lines.append(f"(relevance: {score:.3f})")

    panel_text = "\n".join(body_lines)
    if CONSOLE:
        CONSOLE.print(Panel(panel_text, title=header))
    else:
        print("===", header, "===")
        print(panel_text)


def main(args):
    corpus = load_corpus(DATA_PATH)
    search = RigvedaSearch(corpus)
    lm = SanskritLM()
    tts = TTS()
    voice = VoiceIn() if args.voice else None

    safe_print(Panel("[bold yellow]RigvedaBot — Mandalas 1-10 (Demo)[/bold yellow]\nSource: Place your Rigveda Saṁhitā (Mandalas 1-10) JSONL in data/rigveda_corpus.jsonl\nAttribution: GRETIL or other public-domain editions.\nModel: gyanai/sanskrit-139M-hf (optional)", title="Welcome"))

    while True:
        try:
            if args.voice:
                user_input = voice.listen() or ""
                if user_input is None:
                    user_input = Prompt.ask("(couldn't hear) Type your query") if CONSOLE else input("Type your query: ")
                else:
                    safe_print(f"[blue]Heard:[/blue] {user_input}")
            else:
                user_input = Prompt.ask("Ask about Rigveda (or 'exit' to quit)") if CONSOLE else input("Ask about Rigveda (or 'exit' to quit): ")

            if not user_input:
                continue
            if user_input.strip().lower() in ("exit", "quit"):
                break

            # 1) Try semantic search
            hits = search.search(user_input, top_k=3)
            if hits:
                safe_print("[green]Top corpus hits:[/green]")
                for h in hits:
                    render_result(h)
            else:
                safe_print("[yellow]No close matches in corpus.[/yellow]")

            # 2) Ask Sanskrit LM for a short completion/explanation if available
            if lm.model:
                prompt = f"{user_input}\nExplain briefly in English and give the likely Rigveda reference if any.\n"
                out = lm.generate(prompt, max_length=128)
                if out:
                    safe_print(Panel(out, title="Sanskrit LM output (may be creative)") )
                    if args.tts:
                        tts.speak(out)
            else:
                safe_print("[cyan]Sanskrit LM not available — only corpus search used.[/cyan]")

            # 3) Visualization (optional)
            if args.visual:
                # visualize the first hit's sanskrit if present
                if hits and hits[0].get('sanskrit'):
                    threading.Thread(target=visualize_mantra, args=(hits[0].get('sanskrit'),), daemon=True).start()
                else:
                    safe_print("[yellow]No Sanskrit text to visualize from hits.[/yellow]")

            safe_print(Panel("Attribution: Corpus source must be provided by you (e.g., GRETIL). Bot outputs are AI-assisted and should be verified.", title="Note"))

        except KeyboardInterrupt:
            break
        except Exception as e:
            safe_print("[red]Error in loop:[/red]", e)

    safe_print("[bold]Goodbye — RigvedaBot[/bold]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rigveda CLI Chatbot (Mandalas 1-10)')
    parser.add_argument('--voice', action='store_true', help='Enable voice input')
    parser.add_argument('--ttS', dest='tts', action='store_true', help='Enable text-to-speech')
    parser.add_argument('--visual', action='store_true', help='Show mantra visualization')
    args = parser.parse_args()
    main(args)
