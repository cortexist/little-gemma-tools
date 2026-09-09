"use strict";
// Conservative display-only fallback. A model-authored gesture always wins.
function speechGesture(text, rows) {
  if (/[?"“”]/.test(text) || /\b(if|whether)\b/i.test(text)) return null;
  const words = text.toLowerCase().replaceAll("’", "'").match(/[a-z]+(?:'[a-z]+)*/g) || [];
  const negative = new Set(["no", "not", "never", "can't", "cannot", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "won't"]);
  // "not only" is additive, not a denial; mixed yes/no is deliberately skipped.
  if (/\bnot only\b/i.test(text)) return null;
  const neg = words.findIndex(w => negative.has(w));
  if (neg >= 0 && words.includes("yes")) return null;
  const index = neg;
  if (index < 0) return null;
  // Piper preserves spaces between phoneme words. Skip number expansion,
  // abbreviation splitting, or any other non-bijective text/phoneme mapping.
  const starts = [];
  const sounds = [];
  let boundary = true;
  for (const row of rows) {
    const [start, duration, ph] = row.split("\t");
    if (ph === undefined) continue;
    if (/^\s+$/.test(ph)) { boundary = true; continue; }
    if (/^[\^$.,!?:;ˈˌːˑ]+$/.test(ph)) continue;
    if (boundary) { starts.push(+start); sounds.push(""); boundary = false; }
    sounds[sounds.length - 1] += ph;
  }
  let group = index;
  {
    // eSpeak can fuse "I am" into one group. Anchor a unique negative
    // phoneme group instead of discarding the clause or guessing its offset.
    const forms = {
      no: /^n(oʊ|əʊ|o)$/, not: /^n(ɑ|ɒ|ʌ|ə)t$/, never: /^nɛv(ɚ|ə|əɹ)$/,
      "can't": /^k(æ|ɑ)nt$/, "don't": /^d(oʊ|əʊ)nt$/,
      "doesn't": /^dʌzə?nt$/, "isn't": /^ɪzə?nt$/,
      "aren't": /^ɑɹ?nt$/, "wasn't": /^w(ʌ|ɒ|ə)zə?nt$/,
      "weren't": /^w(ɜ|ɚ)ɹ?nt$/, "won't": /^w(oʊ|əʊ)nt$/,
    };
    const pattern = forms[words[index]];
    const matches = sounds.map((sound, i) =>
      pattern?.test(sound.replace(/[ˈˌːˑ.,!?:;^$]/g, "")) ? i : -1).filter(i => i >= 0);
    if (matches.length > 1) return null;
    if (matches.length === 1) group = matches[0];
    else if (starts.length !== words.length) return null;
  }
  if (!Number.isFinite(starts[group])) return null;
  return { kind: "shake", word: words[index], start: starts[group] };
}
if (typeof module !== "undefined") module.exports = { speechGesture };
