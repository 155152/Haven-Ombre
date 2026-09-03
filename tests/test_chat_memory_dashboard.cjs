const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function dashboard() {
  const elements = new Map();
  const context = {
    window: {},
    document: { getElementById(id) {
      if (!elements.has(id)) elements.set(id, { innerHTML: '', textContent: '', classList: { add() {}, remove() {} } });
      return elements.get(id);
    } },
    esc: value => String(value).replaceAll('<', '&lt;'),
    escAttr: value => String(value).replaceAll('"', '&quot;'),
    jsString: value => String(value),
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../dashboard_assets/chat-memory.js'), 'utf8'), context);
  return { context, elements };
}

test('needs_repair is visible with repair action and disabled write action', () => {
  const { context } = dashboard();
  const html = context.window.renderDailyChatMemoryPending([{
    id: 'candidate1', status: 'pending', date: '2026-09-02',
    candidate: { title: 'Test candidate', content: '<sample>', provenance_status: 'needs_repair' },
  }]);
  assert.match(html, /Test candidate/);
  assert.match(html, /待补齐 · needs_repair/);
  assert.match(html, /repairDailyChatMemory/);
  assert.match(html, /disabled title="请先补齐证据"/);
  assert.match(html, /&lt;sample>/);
  assert.doesNotMatch(html, /暂无待确认候选/);
});

test('aligned candidate can be confirmed', () => {
  const { context } = dashboard();
  const html = context.window.renderDailyChatMemoryPending([{
    id: 'candidate1', candidate: { provenance_status: 'aligned' },
  }]);
  assert.match(html, /已对齐 · aligned/);
  assert.doesNotMatch(html, /disabled/);
});

test('repair posts only to repair and reloads pending without confirming', async () => {
  const { context, elements } = dashboard();
  const calls = [];
  context.authFetch = async (url, options) => {
    calls.push({ url, options });
    return { ok: true, json: async () => options ? { aligned: 1 } : { items: [] } };
  };
  const button = { disabled: false, closest: () => null };
  await context.window.repairDailyChatMemory(button, 'candidate1');
  assert.equal(calls[0].url, '/api/daily-chat-memory/repair');
  assert.deepEqual(JSON.parse(calls[0].options.body), { candidate_ids: ['candidate1'] });
  assert.equal(calls[1].url, '/api/daily-chat-memory/pending?limit=20');
  assert.equal(calls.length, 2);
  assert.match(elements.get('daily-chat-memory-message').textContent, /等待你确认写入/);
  assert.equal(button.disabled, false);
});
