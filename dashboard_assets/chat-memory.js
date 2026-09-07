(function () {
  function dailyChatMemoryApiBase() {
    return typeof BASE !== 'undefined' ? BASE : '';
  }

  function setDailyChatMemoryMessage(message, tone) {
    var el = document.getElementById('daily-chat-memory-message');
    if (!el) return;
    el.textContent = message || '';
    el.classList.remove('ok', 'error');
    if (tone) el.classList.add(tone);
  }

  async function loadDailyChatMemoryPending() {
    var target = document.getElementById('daily-chat-memory-pending');
    if (!target) return;
    target.innerHTML = '<div class="loading">读取候选...</div>';
    try {
      var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/pending?limit=20');
      if (!res) return;
      var data = await res.json();
      if (!res.ok) throw new Error(data.error || '读取失败');
      target.innerHTML = renderDailyChatMemoryPending(data.items || []);
    } catch (e) {
      target.innerHTML = '<div class="loading">读取失败: ' + esc(e.message) + '</div>';
    }
  }

  function renderDailyChatMemoryPending(items) {
    if (!items.length) return '<div class="loading">暂无待确认候选。</div>';
    return items.map(function (item) {
      var candidate = item.candidate || {};
      var id = item.id || '';
      var needsRepair = candidate.provenance_status === 'needs_repair';
      return '<div class="chat-memory-card" data-candidate-id="' + escAttr(id) + '">' +
        '<strong>' + esc(candidate.title || id) + '</strong>' +
        '<div class="chat-memory-card-body">' + esc(candidate.content || '') + '</div>' +
        '<div class="chat-memory-card-meta">' +
          esc((candidate.kind || 'memory') + ' · ' + (item.date || '') + ' · confidence ' + (candidate.confidence || '')) +
        '</div>' +
        '<div class="chat-memory-card-meta">证据：' +
          (needsRepair ? '待补齐 · needs_repair' : candidate.provenance_status === 'aligned' ? '已对齐 · aligned' : '历史候选') +
        '</div>' +
        '<div class="chat-memory-edit-panel" hidden>' +
          '<label class="chat-memory-edit-field">标题' +
            '<input type="text" data-field="title" value="' + escAttr(candidate.title || id) + '" />' +
          '</label>' +
          '<label class="chat-memory-edit-field">正文' +
            '<textarea data-field="content" rows="4">' + esc(candidate.content || '') + '</textarea>' +
          '</label>' +
          '<div class="chat-memory-edit-grid">' +
            '<label class="chat-memory-edit-field">类型' +
              '<input type="text" data-field="kind" value="' + escAttr(candidate.kind || 'memory') + '" />' +
            '</label>' +
            '<label class="chat-memory-edit-field">域' +
              '<input type="text" data-field="domain" value="' + escAttr(listText(candidate.domain)) + '" />' +
            '</label>' +
            '<label class="chat-memory-edit-field">标签' +
              '<input type="text" data-field="tags" value="' + escAttr(listText(candidate.tags)) + '" />' +
            '</label>' +
            '<label class="chat-memory-edit-field">重要度' +
              '<input type="number" min="1" max="10" data-field="importance" value="' + escAttr(candidate.importance || '') + '" />' +
            '</label>' +
            '<label class="chat-memory-edit-field">置信度' +
              '<input type="number" min="0" max="1" step="0.01" data-field="confidence" value="' + escAttr(candidate.confidence || '') + '" />' +
            '</label>' +
          '</div>' +
        '</div>' +
        '<div class="chat-memory-card-actions">' +
          '<button type="button" onclick="toggleDailyChatMemoryEdit(this)">编辑</button>' +
          '<button type="button" onclick="repairDailyChatMemory(this, \'' + jsString(id) + '\')">补证据</button>' +
          '<button type="button"' + (needsRepair ? ' disabled title="请先补齐证据"' : '') + ' onclick="confirmDailyChatMemory(this, \'' + jsString(id) + '\', \'confirm\')">写入</button>' +
          '<button type="button" class="danger" onclick="confirmDailyChatMemory(this, \'' + jsString(id) + '\', \'reject\')">拒绝</button>' +
        '</div>' +
      '</div>';
    }).join('');
  }

  function listText(value) {
    return Array.isArray(value) ? value.join(', ') : String(value || '');
  }

  function dailyChatMemoryField(card, name) {
    var el = card && card.querySelector ? card.querySelector('[data-field="' + name + '"]') : null;
    return el ? String(el.value || '').trim() : '';
  }

  function readDailyChatMemoryEdits(card) {
    if (!card) return null;
    var edits = {
      title: dailyChatMemoryField(card, 'title'),
      content: dailyChatMemoryField(card, 'content'),
      kind: dailyChatMemoryField(card, 'kind'),
      domain: dailyChatMemoryField(card, 'domain'),
      tags: dailyChatMemoryField(card, 'tags'),
    };
    var importance = dailyChatMemoryField(card, 'importance');
    var confidence = dailyChatMemoryField(card, 'confidence');
    if (importance) edits.importance = Number(importance);
    if (confidence) edits.confidence = Number(confidence);
    return edits;
  }

  function toggleDailyChatMemoryEdit(button) {
    var card = button && button.closest ? button.closest('.chat-memory-card') : null;
    var panel = card && card.querySelector ? card.querySelector('.chat-memory-edit-panel') : null;
    if (!panel) return;
    panel.hidden = !panel.hidden;
    if (card) card.setAttribute('data-editing', panel.hidden ? 'false' : 'true');
    button.textContent = panel.hidden ? '编辑' : '收起编辑';
  }

  async function confirmDailyChatMemory(buttonOrId, idOrAction, maybeAction) {
    var button = typeof buttonOrId === 'object' ? buttonOrId : null;
    var id = button ? idOrAction : buttonOrId;
    var action = button ? maybeAction : idOrAction;
    var isReject = action === 'reject';
    if (!confirm(isReject ? '拒绝这条候选？' : '写入这条长期记忆候选？')) return;
    try {
      var body = {
        candidate_ids: [id],
        action: isReject ? 'reject' : 'confirm',
        confirm: isReject ? 'REJECT' : 'WRITE',
      };
      if (!isReject) {
        var card = button && button.closest ? button.closest('.chat-memory-card') : null;
        var edits = card && card.getAttribute('data-editing') === 'true' ? readDailyChatMemoryEdits(card) : null;
        if (edits) {
          body.edits = {};
          body.edits[id] = edits;
        }
      }
      var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res) return;
      var data = await res.json();
      if (!res.ok) throw new Error(data.error || '操作失败');
      if (!isReject && (data.results || []).some(function (result) { return result.status === 'blocked'; })) {
        setDailyChatMemoryMessage('候选已保留，证据待补齐；请点击“补证据”后再写入。', 'error');
        loadDailyChatMemoryPending();
        return;
      }
      setDailyChatMemoryMessage(isReject ? '已拒绝候选。' : '已写入候选。', 'ok');
      loadDailyChatMemoryPending();
      if (!isReject) loadBuckets();
    } catch (e) {
      setDailyChatMemoryMessage('操作失败: ' + e.message, 'error');
    }
  }

  async function repairDailyChatMemory(button, id) {
    button.disabled = true;
    setDailyChatMemoryMessage('正在补齐证据，候选仍保留在待确认列表中…');
    try {
      var body = { candidate_ids: [id] };
      var card = button.closest('.chat-memory-card');
      if (card && card.getAttribute('data-editing') === 'true') {
        body.edits = {};
        body.edits[id] = readDailyChatMemoryEdits(card);
      }
      var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/repair', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      if (!res) return;
      var data = await res.json();
      if (!res.ok) throw new Error(data.error || '补证据失败');
      var detail = data.results && data.results[0] && data.results[0].reason ? ' · ' + data.results[0].reason : '';
      setDailyChatMemoryMessage(data.aligned ? '证据已补齐，等待你确认写入。' : '证据暂未补齐，候选已保留' + detail + '。', data.aligned ? 'ok' : 'error');
      await loadDailyChatMemoryPending();
    } catch (e) {
      setDailyChatMemoryMessage('补证据失败，候选已保留：' + e.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  function initDailyChatMemoryTab() {
    loadDailyChatMemoryPending();
  }

  window.setDailyChatMemoryMessage = setDailyChatMemoryMessage;
  window.loadDailyChatMemoryPending = loadDailyChatMemoryPending;
  window.renderDailyChatMemoryPending = renderDailyChatMemoryPending;
  window.confirmDailyChatMemory = confirmDailyChatMemory;
  window.repairDailyChatMemory = repairDailyChatMemory;
  window.toggleDailyChatMemoryEdit = toggleDailyChatMemoryEdit;
  window.initDailyChatMemoryTab = initDailyChatMemoryTab;

  if (typeof getActiveTab === 'function' && getActiveTab() === 'chat-memory') {
    initDailyChatMemoryTab();
  }
})();
