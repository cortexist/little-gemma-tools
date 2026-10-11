#define main far_field_service_main
#include "../src/far-field-service.c"
#undef main
#include <assert.h>

static void fill_queue(size_t count) {
    mouth_flush();
    m_rendered = 0;
    m_q = realloc(m_q, count * 2);
    m_qcap = m_qn = count;
    for (size_t i = 0; i < count; i++) m_q[i] = (int16_t)(i + 1);
}

static void grace_checks(void) {
    int16_t block[BLOCK];
    // Exercise every cut offset in a render block, including exactly aligned.
    for (unsigned offset = 0; offset < BLOCK; offset++) {
        fill_queue(4800);
        uint64_t marks[] = {640, 1280 + offset, 3200};
        mouth_hold_locked(NULL, 0, 1280);  // alignment may arrive later
        assert(m_hold_not_before == 1280 && m_hold_at == UINT64_MAX);
        assert(mouth_take_block(block));
        mouth_hold_locked(marks, 3, 1280);
        assert(m_hold_not_before == 1280 && m_hold_at == marks[1]);
        size_t source = BLOCK;
        while (!m_held) {
            uint64_t before = m_rendered;
            mouth_take_block(block);
            size_t taken = m_rendered - before;
            for (size_t i = 0; i < taken; i++) assert(block[i] == (int16_t)(++source));
            for (size_t i = taken; i < BLOCK; i++) assert(block[i] == 0);
            // Repeated hold updates must not slide the grace or selected stop.
            mouth_hold_locked(marks, 3, 1280);
            assert(m_hold_at == marks[1] && m_hold_not_before == 1280);
        }
        assert(m_rendered == marks[1] && m_qn == 4800 - source);
        assert(!mouth_take_block(block));
        mouth_resume_locked();
        assert(mouth_take_block(block));
        for (size_t i = 0; i < BLOCK; i++) assert(block[i] == (int16_t)(++source));
        for (size_t i = 0; i < m_qn; i++) assert(m_q[i] == (int16_t)(source + i + 1));
    }
    // A cancellation after grace but before the word boundary is gapless.
    fill_queue(4800);
    uint64_t marks[] = {640, 1600, 3200};
    mouth_hold_locked(marks, 3, 1280);
    for (int n = 0; n < 9; n++) {
        assert(mouth_take_block(block));
        for (int i = 0; i < BLOCK; i++) assert(block[i] == n * BLOCK + i + 1);
    }
    assert(!m_held && m_rendered == 1440);
    mouth_resume_locked();
    for (int n = 9; n < 15; n++) {
        assert(mouth_take_block(block));
        for (int i = 0; i < BLOCK; i++) assert(block[i] == n * BLOCK + i + 1);
    }
    mouth_hold_locked(marks, 3, 0);
    assert(m_hold_not_before == 2400 && m_hold_at == 3200);
    mouth_flush();
    assert(!m_hold_requested && !m_held && m_hold_at == UINT64_MAX);
}

int main(void) {
    int16_t block[BLOCK];
    m_qcap = BLOCK * 3;
    m_q = malloc(m_qcap * 2);
    m_qn = BLOCK * 3;
    for (size_t i = 0; i < m_qn; i++) m_q[i] = (int16_t)(i + 1);
    m_hold_at = BLOCK + 37;
    assert(mouth_take_block(block));
    assert(block[0] == 1 && block[BLOCK-1] == BLOCK);
    assert(mouth_take_block(block));
    assert(block[36] == BLOCK+37 && block[37] == 0);
    assert(m_held && m_rendered == BLOCK+37 && m_qn == BLOCK*2-37);
    assert(!mouth_take_block(block) && block[0] == 0);
    m_held = 0;
    m_hold_at = UINT64_MAX;
    assert(mouth_take_block(block));
    assert(block[0] == BLOCK+38);
    mouth_flush();
    assert(m_qn == 0 && !m_held && m_hold_at == UINT64_MAX);
    grace_checks();
    free(m_q);
    return 0;
}
