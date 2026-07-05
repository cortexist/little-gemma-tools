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
#define SPAN_MAX 4096                   // a <|tool_call> control span

static const char *g_allow = NULL;      // --allow-control-token pattern

// Wildcard match, '*' = any run of characters (the classic iterative form).
static int glob_match(const char *s, const char *p) {
    const char *star_p = NULL, *star_s = NULL;
    while (*s) {
        if (*p == '*')      { star_p = ++p; star_s = s; }
        else if (*p == *s)  { p++; s++; }
        else if (star_p)    { p = star_p; s = ++star_s; }
        else                return 0;
    }
    while (*p == '*') p++;
    return *p == 0;
}

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
    int bad = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--allow-control-token") && i + 1 < argc)
            g_allow = argv[++i];
        else bad = 1;
    }
    if (bad) {
        fprintf(stderr,
            "usage: clausecat [--allow-control-token PATTERN]\n"
            "  raw reply stream on stdin -> one clause per line on stdout\n"
            "  --allow-control-token   a <|tool_call>...<tool_call|> span matching\n"
            "                          PATTERN ('*' = any run) passes through verbatim\n"
            "                          as its own line, ordered against the clauses;\n"
            "                          e.g. '<|tool_call>call:set_voice{*}<tool_call|>'\n"
            "                          (non-matching spans are dropped whole — their\n"
            "                          payload is never spoken)\n");
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
    char span[SPAN_MAX];                // a <|tool_call> span being captured
    size_t sn = 0;
    int tcall = 0;                      // inside <|tool_call> ... <tool_call|>
    const char *tclose = "<tool_call|>";
    size_t tcn = 0;

    int ci;
    while ((ci = getchar()) != EOF) {
        char c = (char)ci;
        if (c == '\n') {                // end of turn: pending '<...' was literal text
            if (!thought && !tcall)
                for (size_t i = 0; i < tn; i++) emit(tag[i]);
            flush_line();
            tn = 0; thought = 0; cn = 0; tcall = 0; sn = 0; tcn = 0;
            continue;
        }
        if (thought) {                  // discard until the exact close marker
            cn = (c == close[cn]) ? cn + 1 : (c == close[0] ? 1 : 0);
            if (cn == strlen(close)) { thought = 0; cn = 0; }
            continue;
        }
        if (tcall) {                    // capture until the exact close marker
            if (sn < sizeof span - 2) span[sn++] = c;
            tcn = (c == tclose[tcn]) ? tcn + 1 : (c == tclose[0] ? 1 : 0);
            if (tcn == strlen(tclose)) {
                span[sn] = 0;
                if (g_allow && sn < sizeof span - 2 && glob_match(span, g_allow)) {
                    flush_line();       // the span holds its place among the clauses
                    fwrite(span, 1, sn, stdout);
                    putchar('\n');
                    fflush(stdout);
                }                       // no pattern, no match, or overflow: dropped whole
                tcall = 0; sn = 0; tcn = 0;
            }
            continue;
        }
        if (tn > 0) {                   // inside a potential <token>
            if (c == '>') {
                if (tn == 1) { emit('<'); emit('>'); tn = 0; continue; }   // "<>": literal
                tag[tn] = 0;
                if (!strcmp(tag, "<|channel"))     // "<|channel>" opens a thought span
                    thought = 1;
                else if (!strcmp(tag, "<|tool_call")) {   // a control span begins
                    tcall = 1;
                    memcpy(span, "<|tool_call>", 12);
                    sn = 12; tcn = 0;
                }
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
    if (!thought && !tcall)
        for (size_t i = 0; i < tn; i++) emit(tag[i]);
    flush_line();
    return 0;
}
