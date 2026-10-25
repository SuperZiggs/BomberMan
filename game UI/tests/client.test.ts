import { test } from 'node:test';
import assert from 'node:assert/strict';

async function input(send: (message: unknown) => void) {
  let module;
  try { module = await import('../src/input.ts'); }
  catch { assert.fail('Browser input controller does not exist yet'); }
  return new module.InputController(send);
}

test('short tap has a press and release with increasing sequence numbers', async () => {
  const messages: any[] = [];
  const controller = await input(message => messages.push(message));
  controller.press('ArrowRight');
  controller.release('ArrowRight');
  assert.equal(messages[0].direction, 'right');
  assert.equal(messages[1].direction, 'stop');
  assert.ok(messages[1].seq > messages[0].seq);
});

test('held direction resumes after another key is released; autorepeat is ignored', async () => {
  const messages: any[] = [];
  const controller = await input(message => messages.push(message));
  controller.press('w');
  controller.press('w');
  controller.press('d');
  controller.release('d');
  assert.deepEqual(messages.map(m => m.direction), ['up', 'right', 'up']);
});

test('bomb is an edge action; clear drops every held key', async () => {
  const messages: any[] = [];
  const controller = await input(message => messages.push(message));
  controller.press('d');
  controller.press(' ');
  controller.press(' ');
  assert.equal(messages.filter(m => m.bomb).length, 1);
  controller.clear();
  assert.equal(messages.at(-1).direction, 'stop');
  controller.release('d');
  assert.equal(messages.at(-1).direction, 'stop');
});
