const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('demo/index.html', 'utf8');
const start = html.indexOf('    async function readChatStream(');
const end = html.indexOf('    async function submitPrompt(', start);
const context = vm.createContext({TextDecoder, Error});
vm.runInContext(html.slice(start, end), context);
const readChatStream = context.readChatStream;
const encode = event => new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);

test('renders deltas before completion and preserves fragmented Unicode', async () => {
  let source;
  let first;
  const gotFirst = new Promise(resolve => { first = resolve; });
  const stream = new ReadableStream({start(controller) { source = controller; }});
  const events = [];
  const read = readChatStream({body: stream}, event => {
    events.push(event);
    if (event.type === 'delta') first();
    return event.type === 'done';
  });
  for (const byte of encode({type: 'delta', text: 'Hi 🚀'})) source.enqueue(Uint8Array.of(byte));
  await gotFirst;
  assert.equal(events.length, 1);
  assert.equal(events[0].text, 'Hi 🚀');
  source.enqueue(encode({type: 'done', tokens: 3}));
  await read;
  assert.equal(events[1].tokens, 3);
});

test('rejects a truncated stream', async () => {
  const body = new ReadableStream({start(c) { c.enqueue(encode({type: 'delta', text: 'partial'})); c.close(); }});
  await assert.rejects(readChatStream({body}, () => false), /before the answer completed/);
});

test('propagates model errors and cancels the response', async () => {
  let cancelled = false;
  const body = new ReadableStream({
    start(c) { c.enqueue(encode({type: 'error', error: 'CUDA unavailable'})); },
    cancel() { cancelled = true; }
  });
  await assert.rejects(readChatStream({body}, event => { throw new Error(event.error); }), /CUDA unavailable/);
  assert.equal(cancelled, true);
});
