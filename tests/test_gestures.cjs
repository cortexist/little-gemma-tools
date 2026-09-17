const assert = require('node:assert/strict');
const {speechGesture} = require('../script/deviceui/gestures.js');
function rows(n) {
  return Array.from({length:n}, (_,i) => [`${i*100}\t50\tæ`,`${i*100+50}\t50\t `]).flat();
}
assert.equal(speechGesture('Yes.',rows(1)),null);
assert.equal(speechGesture('I can explain.',rows(3)),null);
assert.equal(speechGesture('It is flat.',rows(3)),null);
assert.equal(speechGesture('They are lakes.',rows(3)),null);
assert.deepEqual(speechGesture('It is not flat.',rows(4)),{kind:'shake',word:'not',start:200});
assert.deepEqual(speechGesture('I can’t explain.',rows(3)),{kind:'shake',word:"can't",start:100});
assert.deepEqual(speechGesture('No.',rows(1)),{kind:'shake',word:'no',start:0});
for (const text of ['Is it Minnesota?', 'Can you explain?', 'Do they grow corn?', 'If it is flat.', 'Yes, but not always.', 'It is not only flat.', 'She said "no".']) {
  assert.equal(speechGesture(text,rows(5)),null,text);
}
assert.equal(speechGesture('There are 12 lakes.',rows(5)),null);
assert.equal(speechGesture('It is flat.',rows(2)),null);
assert.equal(speechGesture('The land slopes.',rows(3)),null);
console.log('Gesture selection checks passed.');

// eSpeak fuses "I am" into one phoneme group: locate "not" directly.
const fused = ['0\t50\t^', '50\t100\taɪɐm', '150\t20\t ',
  '170\t100\tnˌɑːt', '270\t20\t ', '290\t100\tʃʊɹ'];
assert.deepEqual(speechGesture('I am not sure.', fused),
  {kind:'shake', word:'not', start:170});
assert.equal(speechGesture('I am sure.', fused),null);
assert.equal(speechGesture('I am not sure.', [...fused,'390\t20\t ','410\t100\tnɑt']),null);
