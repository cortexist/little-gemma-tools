// English personal titles in streamed speech. Keep punctuation in transcripts;
// normalize only the TTS copy because eSpeak can split at abbreviation periods.
#ifndef SPEECH_TITLES_H
#define SPEECH_TITLES_H
#include <stddef.h>
#include <string.h>

static int speech_letter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// Match a complete title ending at text[len-1], including its period.
static const char *speech_title(const char *text, size_t len, size_t *start) {
    static const struct { const char *abbr, *spoken; } titles[] = {
        {"mr", "Mr"}, {"mrs", "Mrs"}, {"ms", "Miz"},
        {"dr", "Doctor"}, {"prof", "Professor"}, {"rev", "Reverend"}
    };
    if (!len || text[len-1] != '.') return NULL;
    size_t a = len-1;
    while (a && speech_letter((unsigned char)text[a-1])) a--;
    // Do not recognize a suffix of an identifier or a UTF-8 word.
    if (a && ((unsigned char)text[a-1] >= 128 || text[a-1] == '_' ||
              (text[a-1] >= '0' && text[a-1] <= '9'))) return NULL;
    for (size_t i = 0; i < sizeof titles / sizeof titles[0]; i++) {
        size_t n = strlen(titles[i].abbr);
        if (len-a-1 != n) continue;
        size_t j = 0;
        for (; j < n; j++) {
            unsigned char c = (unsigned char)text[a+j];
            if (c >= 'A' && c <= 'Z') c += 'a'-'A';
            if (c != (unsigned char)titles[i].abbr[j]) break;
        }
        if (j == n) { if (start) *start = a; return titles[i].spoken; }
    }
    return NULL;
}

static int speech_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

static int speech_title_continues(const char *text, size_t len, char next) {
    return speech_space(next) && speech_title(text, len, NULL) != NULL;
}

// out needs 2*len+1 bytes: none of the replacements exceeds twice its input.
// A bare title at end of turn stays unchanged; only titles before a name change.
static size_t speech_normalize_titles(const char *text, size_t len, char *out) {
    size_t copied = 0, used = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] != '.' || i+1 >= len || !speech_space(text[i+1])) continue;
        size_t next = i+1, start = 0;
        while (next < len && speech_space(text[next])) next++;
        if (next == len || (!speech_letter((unsigned char)text[next]) && (unsigned char)text[next] < 128)) continue;
        const char *spoken = speech_title(text, i+1, &start);
        if (!spoken) continue;
        memcpy(out+used, text+copied, start-copied); used += start-copied;
        size_t n = strlen(spoken); memcpy(out+used, spoken, n); used += n;
        copied = i+1;
    }
    memcpy(out+used, text+copied, len-copied); used += len-copied;
    out[used] = 0;
    return used;
}
#endif
