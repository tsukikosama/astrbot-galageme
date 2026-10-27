const $ = (id) => document.getElementById(id);
const categories = { interest: '兴趣爱好', preference: '生活偏好', communication: '沟通习惯', boundary: '相处边界', impression: '互动印象' };
const dimensions = { affection: '好感', trust: '信任', familiarity: '熟悉', closeness: '亲近' };
const sourceNames = { model: '对话提取', summary: '每日整理', manual: '人工维护' };
const kinds = { fact: '事实', preference: '偏好', event: '共同事件', relationship: '关系记忆' };
const jobStates = { sent: '已发送', failed: '已跳过／失败', unknown: '送达情况未知', missed: '本次未触发', reserved: '准备发送' };
const state = { bridge: null, items: [], selected: '', detail: null, offset: 0, limit: 20, total: 0, tab: 'profile', listSequence: 0, detailSequence: 0, editing: null, tagBusy: false };

function el(tag, className = '', content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = String(content);
  return node;
}
function button(label, className, action) {
  const node = el('button', className, label);
  node.type = 'button';
  node.addEventListener('click', action);
  return node;
}
function notice(message = '', error = false) {
  $('status').textContent = message;
  $('status').className = `status${error ? ' error' : ''}`;
}
function date(value) {
  if (!value) return '尚无记录';
  const stamp = new Date(typeof value === 'number' ? value * 1000 : value);
  if (Number.isNaN(stamp.getTime())) return '时间未知';
  return stamp.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
}
function empty(message, compact = true) { return el('p', `empty${compact ? ' compact' : ''}`, message); }
function avatar(name) { return el('div', 'avatar', Array.from(name || '友')[0]); }
function safeScore(value) { return Math.max(0, Math.min(100, Number(value) || 0)); }

function renderUsers() {
  const list = $('user-list');
  list.replaceChildren();
  if (!state.items.length) list.append(empty('还没有匹配的用户。\n请调整筛选，或先在启用群里聊几句。'));
  for (const user of state.items) {
    const name = user.display_name || user.user_id;
    const row = button('', `user-card${user.id === state.selected ? ' selected' : ''}`, () => loadDetail(user.id));
    row.setAttribute('aria-label', `查看 ${name}，群 ${user.group_id}`);
    row.setAttribute('aria-pressed', String(user.id === state.selected));
    const lines = el('div', 'user-lines');
    lines.append(el('span', 'user-name', name), el('span', 'user-meta', `QQ ${user.user_id}`), el('span', 'user-meta', `群 ${user.group_id} · ${user.tag_count || 0} 个标签`));
    row.append(avatar(name), lines, el('span', 'affection', `♡ ${safeScore(user.affection)}`));
    list.append(row);
  }
  $('total').textContent = state.total;
  $('page-info').textContent = state.total ? `${Math.floor(state.offset / state.limit) + 1} / ${Math.ceil(state.total / state.limit)}` : '0 / 0';
  $('previous').disabled = state.offset === 0;
  $('next').disabled = state.offset + state.limit >= state.total;
}

async function loadList() {
  if (!state.bridge) return;
  const sequence = ++state.listSequence;
  $('refresh').disabled = true;
  notice('正在读取用户档案…');
  try {
    const result = await state.bridge.apiGet('users', { q: $('search').value.trim(), group: $('group').value, offset: state.offset, limit: state.limit });
    if (sequence !== state.listSequence) return;
    if (!Array.isArray(result.items)) throw new Error('返回的数据格式不正确，请检查插件版本。');
    if (result.total > 0 && state.offset >= result.total) {
      state.offset = Math.floor((result.total - 1) / state.limit) * state.limit;
      await loadList(); return;
    }
    if (!result.total) state.offset = 0;
    state.items = result.items;
    state.total = result.total;
    $('role-name').textContent = `角色 ${result.role_id}`;
    const known = new Set(Array.from($('group').options, (item) => item.value));
    for (const group of result.groups || []) {
      if (!known.has(String(group))) { const option = el('option', '', `群 ${group}`); option.value = group; $('group').append(option); known.add(String(group)); }
    }
    renderUsers();
    if (!state.items.some((item) => item.id === state.selected)) {
      if (state.items.length) await loadDetail(state.items[0].id);
      else { state.selected = ''; state.detail = null; ++state.detailSequence; $('detail').replaceChildren(empty('暂无用户档案。在已启用群里与机器人交谈后，这里就会出现记录。', false)); }
    }
    if (sequence === state.listSequence) notice();
  } catch (error) {
    if (sequence === state.listSequence) notice(`读取失败：${error.message}。可以点击“刷新数据”重试。`, true);
  } finally {
    if (sequence === state.listSequence) $('refresh').disabled = false;
  }
}

async function loadDetail(id) {
  if (state.selected !== id) state.tab = 'profile';
  state.selected = id;
  state.detail = null;
  renderUsers();
  const sequence = ++state.detailSequence;
  $('detail').replaceChildren(empty('正在读取这段关系…', false));
  try {
    const detail = await state.bridge.apiGet('user', { id });
    if (sequence !== state.detailSequence || id !== state.selected) return;
    if (!detail.profile || detail.profile.id !== id) throw new Error('用户档案不匹配，请刷新。');
    state.detail = detail;
    renderDetail();
  } catch (error) {
    if (sequence !== state.detailSequence) return;
    const retry = button('重试', 'button secondary', () => loadDetail(id));
    const content = empty(`读取详情失败：${error.message}`, false);
    content.append(document.createElement('br'), retry);
    $('detail').replaceChildren(content);
  }
}

function renderDetail() {
  const detail = state.detail;
  if (!detail) return;
  const profile = detail.profile;
  const container = $('detail');
  container.replaceChildren();
  const name = profile.display_name || profile.user_id;
  const header = el('div', 'profile-header');
  const identity = el('div');
  identity.append(el('h2', 'profile-name', name), el('div', 'profile-identities', `QQ ${profile.user_id} · 群 ${profile.group_id}`), el('div', 'profile-identities', `最近互动 ${date(profile.last_interaction)} · 北京时间`));
  const pills = el('div', 'status-pills');
  pills.append(el('span', `pill${profile.memory_enabled ? ' on' : ''}`, profile.memory_enabled ? '记忆开启' : '记忆暂停'), el('span', `pill${profile.proactive_enabled ? ' on' : ''}`, profile.proactive_enabled ? '允许主动联系' : '主动联系已关闭'));
  if (!profile.group_enabled) pills.append(el('span', 'pill', '该群当前未启用'));
  header.append(avatar(name), identity, pills);
  container.append(header);
  const scores = el('div', 'scores');
  for (const [key, label] of Object.entries(dimensions)) {
    const score = el('div', 'score');
    const value = el('div', 'score-value', safeScore(profile[key]));
    value.append(el('small', '', '/ 100'));
    const track = el('div', 'score-track');
    track.setAttribute('role', 'meter'); track.setAttribute('aria-label', label);
    track.setAttribute('aria-valuemin', '0'); track.setAttribute('aria-valuemax', '100'); track.setAttribute('aria-valuenow', safeScore(profile[key]));
    const fill = el('div', 'score-fill'); fill.style.width = `${safeScore(profile[key])}%`;
    track.append(fill); score.append(el('span', 'score-label', label), value, track); scores.append(score);
  }
  container.append(scores);
  const relationship = el('div', 'relationship-note');
  relationship.append(el('span', 'section-caption', '目前的相处方式'), el('div', '', profile.style || '还在慢慢了解对方。相处方式会随着真实经历逐渐形成。'));
  container.append(relationship);
  const tabs = el('nav', 'tabs'); tabs.setAttribute('role', 'tablist'); tabs.setAttribute('aria-label', '档案栏目');
  for (const [key, label] of [['profile', '用户画像'], ['memories', '长期记忆'], ['changes', '关系变化'], ['summaries', '每日整理'], ['proactive', '主动联系']]) {
    const tab = button(label, `tab${state.tab === key ? ' active' : ''}`, () => { state.tab = key; renderDetail(); });
    tab.setAttribute('role', 'tab'); tab.setAttribute('aria-selected', String(state.tab === key)); tab.id = `tab-${key}`;
    tabs.append(tab);
  }
  container.append(tabs);
  const panel = el('section', 'tab-panel'); panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', `tab-${state.tab}`);
  container.append(panel);
  if (state.tab === 'profile') renderTags(panel, detail.tags || []);
  else if (state.tab === 'memories') renderMemories(panel, detail.memories || []);
  else if (state.tab === 'changes') renderChanges(panel, detail.changes || []);
  else if (state.tab === 'summaries') renderSummaries(panel, detail.daily_summaries || []);
  else renderProactive(panel, detail.proactive || []);
}

function heading(panel, title, description, action) {
  const row = el('div', 'section-heading'); const copy = el('div');
  copy.append(el('h2', '', title), el('p', 'muted', description)); row.append(copy);
  if (action) row.append(action); panel.append(row);
}
function renderTags(panel, tags) {
  heading(panel, '关于这个人的一些了解', '点击标签可查看依据、编辑分类或移除。人工维护优先于自动提取。', button('＋ 添加标签', 'button small', () => editTag(null)));
  if (!tags.length) { panel.append(empty('还没有标签。有依据的新偏好会在聊天或每日整理后出现，也可以手动添加。')); return; }
  for (const [category, label] of Object.entries(categories)) {
    const values = tags.filter((item) => item.category === category);
    if (!values.length) continue;
    const section = el('section', 'tag-category'); section.append(el('div', 'category-title', label));
    const chips = el('div', 'tags');
    for (const tag of values) {
      const chip = button('', `tag-chip${tag.source === 'manual' ? ' manual' : ''}`, () => editTag(tag));
      chip.append(el('span', '', tag.name), el('span', 'tag-source', tag.source === 'manual' ? '人工' : `${Math.round((Number(tag.confidence) || 0) * 100)}%`));
      if (!tag.public) chip.append(el('span', 'tag-private', '仅管理页'));
      chip.title = `${sourceNames[tag.source] || tag.source} · ${tag.evidence || '人工维护'} `;
      chips.append(chip);
    }
    section.append(chips); panel.append(section);
  }
}
function record(content, metadata = []) {
  const card = el('article', 'record'); card.append(el('p', 'record-body', content));
  const meta = el('div', 'record-meta'); metadata.filter(Boolean).forEach((value) => meta.append(el('span', '', value))); card.append(meta); return card;
}
function renderMemories(panel, memories) {
  heading(panel, '记得的事情', `最近 ${memories.length} 条。非公开内容只在这个已登录的管理页查看。`);
  if (!memories.length) panel.append(empty('目前没有长期记忆。'));
  for (const memory of memories) {
    const card = record(memory.content, [kinds[memory.kind] || memory.kind, memory.public ? '群聊可参考' : '非公开', date(memory.created_at)]);
    const source = el('details'); source.append(el('summary', '', '查看来源'), el('p', '', `来源类型：${memory.source_type === 'summary' ? '每日整理批次' : '对话消息'}\n来源编号：${(memory.source_ids || []).join('、') || '旧版记录暂无来源'}`));
    card.append(source); panel.append(card);
  }
}
function renderChanges(panel, changes) {
  heading(panel, '关系是怎样变化的', '最近20次有数值变化的记录，附判断原因。');
  if (!changes.length) panel.append(empty('还没有数值变化记录。普通问答保持不变是正常的。'));
  for (const change of changes) {
    const card = el('article', 'record'); const values = el('div', 'changes-values');
    for (const [key, label] of Object.entries(dimensions)) {
      const delta = Number(change.deltas?.[key]) || 0;
      if (!delta) continue;
      values.append(el('span', `delta${delta < 0 ? ' negative' : ''}`, `${label} ${delta > 0 ? '+' : ''}${delta}（${change.old_values?.[key]} → ${change.new_values?.[key]}）`));
    }
    card.append(values, el('p', 'record-body', change.reason || '原因正文未保存或已清理。'), el('div', 'record-meta', date(change.created_at))); panel.append(card);
  }
}
function renderSummaries(panel, summaries) {
  heading(panel, '每日整理', '最近20个整理批次，按北京时间归档；对话较多时会分批完成。');
  if (!summaries.length) panel.append(empty('还没有每日摘要。到设定的整理时间后，会自动处理前一天的互动。'));
  for (const summary of summaries) panel.append(record(summary.content, [summary.day, `整理于 ${date(summary.updated_at)}`]));
}
function renderProactive(panel, jobs) {
  heading(panel, '主动联系记录', '最近20次联系机会。没有抽中、生成放弃或发送失败都不代表关系扣分。');
  if (!jobs.length) panel.append(empty('还没有主动联系记录。需要合适的动机，并通过概率与频率检查。'));
  for (const job of jobs) panel.append(record(job.text || '本次没有发送消息正文。', [jobStates[job.status] || job.status, date(job.created_at)]));
}

function editTag(tag) {
  if (!state.detail) return;
  state.editing = { tag, user: state.selected, revision: state.detail.profile.revision };
  $('tag-title').textContent = tag ? '查看与编辑标签' : '添加用户标签';
  $('tag-name').value = tag?.name || ''; $('tag-name').readOnly = !!tag;
  $('tag-category').value = tag?.category || 'interest'; $('tag-public').checked = !!tag?.public;
  $('tag-evidence').textContent = tag ? `来源：${sourceNames[tag.source] || tag.source}\n依据：${tag.evidence || '管理员手动维护'}\n来源编号：${(tag.source_ids || []).join('、') || '无'}\n更新于：${date(tag.updated_at)}` : '';
  $('tag-error').textContent = ''; $('remove-tag').hidden = !tag;
  $('tag-dialog').showModal();
}
function tagBusy(value) {
  state.tagBusy = value;
  for (const id of ['save-tag', 'remove-tag', 'confirm-delete', 'close-tag', 'cancel-delete']) $(id).disabled = value;
}
async function saveTag(event) {
  event.preventDefault();
  if (state.tagBusy || !state.editing) return;
  tagBusy(true); $('tag-error').textContent = '';
  const target = { ...state.editing };
  try {
    await state.bridge.apiPost('tags/save', { id: target.user, revision: target.revision, name: $('tag-name').value.trim(), category: $('tag-category').value, public: $('tag-public').checked });
    $('tag-dialog').close();
    if (state.selected === target.user) await loadDetail(target.user);
    await loadList();
  } catch (error) { $('tag-error').textContent = `保存失败：${error.message}`; }
  finally { tagBusy(false); }
}
async function deleteTag() {
  if (state.tagBusy || !state.editing?.tag) return;
  tagBusy(true); const target = { ...state.editing };
  try {
    await state.bridge.apiPost('tags/delete', { id: target.user, revision: target.revision, tag_id: target.tag.id });
    $('delete-dialog').close(); $('tag-dialog').close();
    if (state.selected === target.user) await loadDetail(target.user);
    await loadList();
  } catch (error) { $('delete-dialog').close(); $('tag-error').textContent = `移除失败：${error.message}`; }
  finally { tagBusy(false); }
}

$('tag-form').addEventListener('submit', saveTag);
$('close-tag').addEventListener('click', () => $('tag-dialog').close());
$('cancel-delete').addEventListener('click', () => $('delete-dialog').close());
$('remove-tag').addEventListener('click', () => { $('delete-description').textContent = `将移除“${state.editing.tag.name}”。其他标签和关系数值会保留。`; $('delete-dialog').showModal(); });
$('confirm-delete').addEventListener('click', deleteTag);
for (const id of ['tag-dialog', 'delete-dialog']) $(id).addEventListener('cancel', (event) => { if (state.tagBusy) event.preventDefault(); });
$('refresh').addEventListener('click', async () => { const id = state.selected; await loadList(); if (id && state.selected === id) await loadDetail(id); });
$('previous').addEventListener('click', () => { state.offset = Math.max(0, state.offset - state.limit); loadList(); });
$('next').addEventListener('click', () => { state.offset += state.limit; loadList(); });
$('group').addEventListener('change', () => { state.offset = 0; loadList(); });
let searchTimer;
$('search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.offset = 0; loadList(); }, 300); });

async function start() {
  const bridge = window.AstrBotPluginView || window.AstrBotPluginPage;
  if (!bridge) { notice('请从 AstrBot 管理面板的“扩展功能 → 关系与用户画像”打开此页。', true); return; }
  try {
    await bridge.ready(); state.bridge = bridge;
    const theme = (context) => { document.documentElement.dataset.theme = context?.isDark ? 'dark' : 'light'; };
    theme(bridge.getContext?.()); bridge.onContext?.(theme);
    await loadList();
  } catch (error) { notice(`管理面板连接失败：${error.message}`, true); }
}
start();
