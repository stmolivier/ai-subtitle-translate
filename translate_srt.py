#!/usr/bin/env python3
"""
translate_srt.py - Traduit un fichier SRT avec un LLM local (Ollama) en
conservant la structure (numéros, timestamps) et en tenant compte du contexte.

Principes :
  * Les numéros et timestamps ne sont JAMAIS envoyés au modèle ni modifiés.
  * Le fichier est traité par fenêtres de N blocs ; les fenêtres ne coupent pas
    une phrase en deux (on étend jusqu'à la fin de phrase).
  * Le modèle reçoit en contexte : les blocs précédents (source + traduction
    déjà faite) et les blocs suivants (source), pour gérer les phrases réparties
    sur plusieurs blocs.
  * Sortie JSON contrainte par un schéma (Ollama "format") : mêmes identifiants
    en entrée et en sortie, vérifiés. En cas d'échec : retries, puis découpage
    de la fenêtre en deux, puis conservation du texte source en dernier recours.
  * Budget de caractères par bloc selon sa durée (caractères/seconde).
  * Reprise automatique après interruption (fichier cache).

Dépendances :  pip install srt requests
Exemple :
  ollama pull qwen2.5:14b
  python translate_srt.py film.en.srt film.fr.srt --source English --target French \
      --model qwen2.5:14b --context "Documentaire sur la construction navale"
"""

import argparse
import json
import math
import re
import sys
import textwrap
from datetime import timedelta
from pathlib import Path

import requests
import srt

SENTENCE_END = re.compile(r"""[.!?…。！？」』"')\]♪]\s*$""")
TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Utilitaires texte
# --------------------------------------------------------------------------- #
def plain_len(text):
    """Longueur visible (sans balises, sans retours à la ligne)."""
    return len(TAG_RE.sub("", text).replace("\n", " ").strip())


def ends_sentence(text):
    return bool(SENTENCE_END.search(TAG_RE.sub("", text).strip()))


def wrap_line(line, width):
    """Découpe une ligne trop longue en lignes équilibrées (<= width si possible)."""
    line = line.strip()
    if len(line) <= width:
        return [line]
    n_lines = math.ceil(len(line) / width)
    if " " in line:
        for w in range(math.ceil(len(line) / n_lines), width + 1):
            parts = textwrap.wrap(line, width=w, break_long_words=False)
            if len(parts) <= n_lines:
                return parts
        return textwrap.wrap(line, width=width, break_long_words=False)
    # Pas d'espaces (chinois, japonais...) : découpe par caractères
    size = math.ceil(len(line) / n_lines)
    return [line[i:i + size] for i in range(0, len(line), size)]


def format_text(text, max_line):
    """Nettoie et re-découpe les lignes trop longues, conserve les retours existants."""
    lines = [l.strip() for l in text.replace("\r", "").split("\n") if l.strip()]
    out = []
    for l in lines:
        out.extend(wrap_line(l, max_line))
    return "\n".join(out)


def defaults_for(target):
    """(max_cps, max_line) par défaut selon la langue cible."""
    t = target.lower()
    if any(k in t for k in ("chinese", "mandarin", "cantonese", "chinois", "中文",
                            "japanese", "japonais", "korean", "coréen")):
        return 10, 16
    return 17, 42


def load_glossary(path):
    if not path:
        return ""
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#"):
            lines.append(raw)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Client Ollama
# --------------------------------------------------------------------------- #
class Translator:
    def __init__(self, args, subs):
        self.a = args
        self.subs = subs
        self.n = len(subs)
        self.ids = [str(s.index) for s in subs]
        self.translations = {}
        self.glossary = load_glossary(args.glossary)
        self.use_think_param = True
        self.cache_path = Path(args.output).with_suffix(".cache.json")
        self._load_cache()

    # ---- cache ------------------------------------------------------------
    def _load_cache(self):
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if data.get("target") == self.a.target and data.get("model") == self.a.model:
                    self.translations = data.get("translations", {})
                    log(f"Reprise : {len(self.translations)} blocs déjà traduits ({self.cache_path.name})")
            except Exception:
                pass

    def _save_cache(self):
        self.cache_path.write_text(json.dumps({
            "target": self.a.target, "model": self.a.model,
            "translations": self.translations,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- appel modèle -----------------------------------------------------
    def chat(self, system, user, schema):
        payload = {
            "model": self.a.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "options": {"temperature": self.a.temperature, "num_ctx": self.a.num_ctx},
        }
        if self.use_think_param:
            payload["think"] = False  # désactive le raisonnement (qwen3...)
        r = requests.post(f"{self.a.ollama_url}/api/chat", json=payload, timeout=self.a.timeout)
        if r.status_code == 400 and self.use_think_param and "think" in r.text.lower():
            self.use_think_param = False
            return self.chat(system, user, schema)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        return json.loads(content)

    # ---- prompt -----------------------------------------------------------
    def system_prompt(self):
        a = self.a
        return (
            f"You are an expert subtitle translator from {a.source} to {a.target}.\n"
            "You receive consecutive subtitle blocks from the same video. Rules:\n"
            "- A sentence may be spread over several consecutive blocks. Understand the whole "
            "sentence using the surrounding blocks, translate it naturally in the target language, "
            "then distribute the translation over the SAME blocks, in order, cutting at natural "
            "points of the target language (word order differs between languages).\n"
            "- Never merge, drop, add or reorder blocks. Every id in the input must appear in the output.\n"
            "- Each block's display time is fixed. Stay within 'max_chars' for each block "
            "(be concise, condense if needed, keep the meaning).\n"
            "- Keep dialogue dashes at the start of lines, line breaks between speakers, "
            "and formatting tags such as <i>...</i>. Keep sound cues like [applause] or ♪ translated/kept.\n"
            "- Translate names, numbers and idioms appropriately; stay consistent with the glossary "
            "and with previously translated blocks.\n"
            "- Output ONLY the JSON object {id: translated_text}. No comments, no notes."
        )

    def user_prompt(self, start, end, hint=None):
        a = self.a
        parts = []
        if a.context:
            parts.append(f"VIDEO CONTEXT: {a.context}")
        if self.glossary:
            parts.append("GLOSSARY (source = target):\n" + self.glossary)

        before = []
        for i in range(max(0, start - a.context_blocks), start):
            sid = self.ids[i]
            before.append(f"[{sid}] {self.subs[i].content.replace(chr(10), ' / ')}"
                          f"  =>  {self.translations.get(sid, '').replace(chr(10), ' / ')}")
        if before:
            parts.append("PREVIOUS BLOCKS (already translated, context only, do NOT output):\n"
                         + "\n".join(before))

        items = []
        for i in range(start, end):
            s = self.subs[i]
            dur = (s.end - s.start).total_seconds()
            budget = int(dur * a.max_cps)
            budget = max(12, min(budget, a.max_line * a.max_lines))
            items.append({"id": self.ids[i], "text": s.content, "max_chars": budget})
        parts.append("BLOCKS TO TRANSLATE:\n" + json.dumps(items, ensure_ascii=False, indent=1))

        after = []
        for i in range(end, min(self.n, end + a.context_blocks)):
            after.append(f"[{self.ids[i]}] {self.subs[i].content.replace(chr(10), ' / ')}")
        if after:
            parts.append("FOLLOWING BLOCKS (source, context only, do NOT output):\n" + "\n".join(after))

        if hint:
            parts.append(f"PREVIOUS ATTEMPT FAILED: {hint}\nReturn every id exactly once.")
        parts.append(f"Return a JSON object with exactly these ids: {', '.join(self.ids[start:end])}")
        return "\n\n".join(parts)

    # ---- traduction d'une plage -------------------------------------------
    def _try_range(self, start, end, hint):
        ids = self.ids[start:end]
        schema = {"type": "object",
                  "properties": {i: {"type": "string"} for i in ids},
                  "required": ids, "additionalProperties": False}
        result = self.chat(self.system_prompt(), self.user_prompt(start, end, hint), schema)
        if not isinstance(result, dict):
            raise ValueError("la réponse n'est pas un objet JSON")
        missing = [i for i in ids if i not in result]
        extra = [k for k in result if k not in ids]
        if missing or extra:
            raise ValueError(f"ids manquants {missing} / en trop {extra}")
        for i, s in zip(ids, self.subs[start:end]):
            if not isinstance(result[i], str):
                raise ValueError(f"valeur non textuelle pour {i}")
            if s.content.strip() and not result[i].strip():
                raise ValueError(f"traduction vide pour le bloc {i}")
        return {i: result[i].strip() for i in ids}

    def translate_range(self, start, end):
        hint = None
        for attempt in range(1, self.a.retries + 1):
            try:
                res = self._try_range(start, end, hint)
                self.translations.update(res)
                self._save_cache()
                return
            except requests.ConnectionError:
                sys.exit("Impossible de joindre Ollama. Est-il lancé ? (ollama serve)")
            except (ValueError, json.JSONDecodeError, requests.RequestException, KeyError) as e:
                hint = str(e)
                log(f"  ! blocs {self.ids[start]}-{self.ids[end - 1]}, essai {attempt}/{self.a.retries} : {hint}")
        if end - start > 1:
            mid = (start + end) // 2
            log(f"  -> découpage de la fenêtre {self.ids[start]}-{self.ids[end - 1]}")
            self.translate_range(start, mid)
            self.translate_range(mid, end)
        else:
            log(f"  !! bloc {self.ids[start]} non traduit, texte source conservé")
            self.translations[self.ids[start]] = self.subs[start].content
            self._save_cache()

    # ---- boucle principale ------------------------------------------------
    def run(self):
        a = self.a
        i = 0
        while i < self.n:
            end = min(i + a.window, self.n)
            # ne pas couper au milieu d'une phrase
            limit = min(self.n, i + int(a.window * 1.5))
            while end < limit and not ends_sentence(self.subs[end - 1].content):
                end += 1
            if all(x in self.translations for x in self.ids[i:end]):
                i = end
                continue
            log(f"Blocs {self.ids[i]}-{self.ids[end - 1]} / {self.ids[-1]}")
            self.translate_range(i, end)
            i = end

    # ---- passe de condensation (optionnelle) ------------------------------
    def shorten_pass(self):
        a = self.a
        fixed = 0
        for i, s in enumerate(self.subs):
            sid = self.ids[i]
            text = self.translations.get(sid, "")
            dur = max((s.end - s.start).total_seconds(), 0.5)
            if plain_len(text) / dur <= a.max_cps:
                continue
            target_len = int(dur * a.max_cps)
            prev_t = self.translations.get(self.ids[i - 1], "") if i > 0 else ""
            next_t = self.translations.get(self.ids[i + 1], "") if i + 1 < self.n else ""
            prompt = (
                f"Rewrite this {a.target} subtitle more concisely, at most {target_len} characters, "
                f"keeping the meaning and fitting the flow with the neighbouring blocks.\n"
                f"Previous block: {prev_t}\nCurrent block: {text}\nNext block: {next_t}\n"
                f"Source ({a.source}): {s.content}\n"
                'Return JSON: {"text": "..."}'
            )
            schema = {"type": "object", "properties": {"text": {"type": "string"}},
                      "required": ["text"], "additionalProperties": False}
            try:
                new = self.chat("You are an expert subtitle editor.", prompt, schema)["text"].strip()
            except Exception as e:
                log(f"  ! condensation du bloc {sid} échouée : {e}")
                continue
            if new and len(new) < len(text):
                self.translations[sid] = new
                fixed += 1
        self._save_cache()
        log(f"Condensation : {fixed} blocs raccourcis")


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def check_ollama(url, model):
    try:
        r = requests.get(f"{url}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        sys.exit(f"Impossible de joindre Ollama sur {url}. Lancez 'ollama serve'.")
    names = [m["name"] for m in r.json().get("models", [])]
    if not any(n == model or n.split(":")[0] == model or n == model + ":latest" for n in names):
        sys.exit(f"Modèle '{model}' introuvable. Faites : ollama pull {model}\n"
                 f"Modèles installés : {', '.join(names) or 'aucun'}")


def main():
    p = argparse.ArgumentParser(description="Traduit un SRT avec Ollama en gardant la structure.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--source", default="English", help="langue source (ex. English)")
    p.add_argument("--target", default="French", help="langue cible (ex. Telugu, Spanish, Mandarin Chinese)")
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--context", default="", help="description libre de la vidéo (sujet, ton, personnages)")
    p.add_argument("--glossary", help="fichier texte : une ligne 'terme source = terme cible'")
    p.add_argument("--window", type=int, default=15, help="blocs traduits par requête")
    p.add_argument("--context-blocks", type=int, default=4, help="blocs de contexte avant/après")
    p.add_argument("--max-cps", type=float, default=None, help="caractères par seconde max")
    p.add_argument("--max-line", type=int, default=None, help="caractères max par ligne")
    p.add_argument("--max-lines", type=int, default=2)
    p.add_argument("--shorten", action="store_true", help="passe de condensation des blocs trop longs")
    p.add_argument("--limit", type=int, default=0, help="ne traduire que les N premiers blocs (test)")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    d_cps, d_line = defaults_for(args.target)
    args.max_cps = args.max_cps or d_cps
    args.max_line = args.max_line or d_line

    check_ollama(args.ollama_url, args.model)

    raw = Path(args.input).read_text(encoding="utf-8-sig")
    subs = sorted(srt.parse(raw), key=lambda x: (x.start, x.index))
    if len({s.index for s in subs}) != len(subs):  # numéros dupliqués : on renumérote
        subs = list(srt.sort_and_reindex(subs))
    if args.limit:
        subs = subs[:args.limit]
    log(f"{len(subs)} blocs, modèle {args.model}, {args.source} -> {args.target}")

    tr = Translator(args, subs)
    tr.run()
    if args.shorten:
        tr.shorten_pass()

    # Reconstruction : on ne touche qu'au texte
    over = 0
    for s, sid in zip(subs, tr.ids):
        s.content = format_text(tr.translations.get(sid, s.content), args.max_line)
        dur = max((s.end - s.start).total_seconds(), 0.001)
        if plain_len(s.content) / dur > args.max_cps * 1.15:
            over += 1
    Path(args.output).write_text(srt.compose(subs, reindex=False), encoding="utf-8")
    log(f"Terminé -> {args.output}")
    if over:
        log(f"Attention : {over} blocs dépassent {args.max_cps} car/s "
            f"(relancez avec --shorten ou relisez-les).")


if __name__ == "__main__":
    main()
