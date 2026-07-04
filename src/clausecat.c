// clausecat — the presentation cat: little-gemma's raw reply stream in, one
// speakable CLAUSE per line out. The runner filters nothing by design
// (presentation is the client's job), so between voicecat and a line-based
// TTS something must strip the scaffolding and decide where speech can start.
// That policy lives here, in one place, instead of inside every consumer:
//
//     voicecat /tmp/lg.sock ... | clausecat | piper -m voice.onnx \
//         --output-raw --stream | aplay -r 22050 -f S16_LE -t raw -c 1 -
//
// What it does, in order:
//   - drops thought spans: <|channel> ... <channel|> (unterminated: to end of turn)
//   - drops special tokens: <...> of 1..24 chars (a lone '<' stays literal text)
//   - flushes a line at each clause boundary: , ; : . ! ? followed by a
//     space-like character — the model's own punctuation is the split policy
//     (docs/voice-sys.txt in the runner repo is what makes it appear early)
//   - a bare newline (voicecat's end-of-turn mark) flushes the remainder
//
// bench/clause_pipe.py in the little-gemma repo is the same policy in python —
// the sandbox where changes are tried first; keep the two in agreement.
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#define LINE_MAX_ 8192
#define TAG_MAX  26                     // '<' + up to 24 chars + '>'

static char line[LINE_MAX_];            // speakable text since the last flush
static size_t ln = 0;

static int is_punct(char c) { return strchr(",;:.!?", c) != NULL; }
static int is_after(char c) {           // what may follow punctuation at a boundary
    return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v' ||
           c == '*' || c == ')' || c == '"' || c == '\'' || c == ']';
}

static void flush_line(void) {
    size_t a = 0, b = ln;
    while (a < b && (line[a] == ' ' || line[a] == '\t')) a++;
    while (b > a && (line[b - 1] == ' ' || line[b - 1] == '\t')) b--;
    if (b > a) {
        fwrite(line + a, 1, b - a, stdout);
        putchar('\n');
        fflush(stdout);
    }
    ln = 0;
}

static void emit(char c) {              // speakable char: boundary check, then append
    if (ln > 0 && is_punct(line[ln - 1]) && is_after(c))
        flush_line();
    if (ln < sizeof line - 1) line[ln++] = c;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        fprintf(stderr, "usage: clausecat  (raw reply stream on stdin -> one clause per line)\n");
        return 1;
    }
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif

    char tag[TAG_MAX + 1];              // pending '<...' that may become a token
    size_t tn = 0;
    int thought = 0;                    // inside <|channel> ... <channel|>
    const char *close = "<channel|>";
    size_t cn = 0;                      // chars of `close` matched so far

    int ci;
    while ((ci = getchar()) != EOF) {
        char c = (char)ci;
        if (c == '\n') {                // end of turn: pending '<...' was literal text
            if (!thought)
                for (size_t i = 0; i < tn; i++) emit(tag[i]);
            flush_line();
            tn = 0; thought = 0; cn = 0;
            continue;
        }
        if (thought) {                  // discard until the exact close marker
            cn = (c == close[cn]) ? cn + 1 : (c == close[0] ? 1 : 0);
            if (cn == strlen(close)) { thought = 0; cn = 0; }
            continue;
        }
        if (tn > 0) {                   // inside a potential <token>
            if (c == '>') {
                if (tn == 1) { emit('<'); emit('>'); tn = 0; continue; }   // "<>": literal
                tag[tn] = 0;
                if (!strcmp(tag, "<|channel"))     // "<|channel>" opens a thought span
                    thought = 1;
                tn = 0;                            // any other <...> is dropped
                continue;
            }
            if (c != '<' && tn < TAG_MAX - 1) { tag[tn++] = c; continue; }
            for (size_t i = 0; i < tn; i++) emit(tag[i]);   // too long or '<': literal
            tn = 0;                     // fall through: c starts over
        }
        if (c == '<') { tag[tn++] = c; continue; }
        emit(c);
    }
    if (!thought)
        for (size_t i = 0; i < tn; i++) emit(tag[i]);
    flush_line();
    return 0;
}
