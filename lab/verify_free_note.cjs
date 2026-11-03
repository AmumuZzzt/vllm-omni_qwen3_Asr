const { chromium } = require('playwright');
const path = require('node:path');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');

const URL_ = pathToFileURL(
  path.resolve('D:/pi-project/notes/vllm/omni-learning-site/notebook.html')
).href;

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
  });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

  await page.goto(URL_);
  await page.waitForTimeout(700);
  assert.equal(errors.length, 0, 'boot errors: ' + errors.join(' | '));

  // 1. existing seed note must stay an experiment note
  await page.getByText('Qwen3-ASR 单卡部署日志').first().click();
  await page.waitForTimeout(500);
  assert(await page.locator('#write-records-heading').count() === 1,
    'seed lab note lost its EXPERIMENTS section');
  assert(await page.locator('.free-note-foot').count() === 0,
    'lab note should not show free-note footer');

  // 2. new note defaults to free note
  await page.locator('[data-action="new-note"]').first().click();
  await page.waitForTimeout(600);
  assert(await page.locator('#write-records-heading').count() === 0,
    'new note should not render EXPERIMENTS section');
  assert(await page.locator('.free-note-foot').count() === 1,
    'new note should show free-note footer');
  assert(await page.locator('[data-action="kind-note"]').getAttribute('aria-pressed') === 'true',
    'free-note button should be pressed');

  // 3. typing body works
  await page.locator('#body-editor').fill('## 今天的一个想法\n\n自由笔记可以先不写命令。');
  await page.waitForTimeout(400);

  // 4. switch to experiment note -> section appears
  await page.locator('[data-action="kind-lab"]').click();
  await page.waitForTimeout(600);
  assert(await page.locator('#write-records-heading').count() === 1,
    'switching to lab note should reveal EXPERIMENTS');
  const headingText = await page.locator('#write-records-heading').innerText();
  assert(headingText.includes('实验记录'), 'heading text unexpected: ' + headingText);
  const pill = await page.locator('#write-records-heading .count-pill').innerText();
  assert(pill === '0', 'fresh lab note should have 0 records, got: ' + pill);

  // 5. switch back and confirm persistence across reload
  await page.locator('[data-action="kind-note"]').click();
  await page.waitForTimeout(600);
  await page.reload();
  await page.waitForTimeout(800);
  const stored = await page.evaluate(() => {
    const raw = JSON.parse(localStorage.getItem('omni-notebook-standalone-v1'));
    const n = raw.notes.find((x) => x.title === '未命名笔记' || (x.body || '').includes('今天的一个想法'));
    return n ? { kind: n.kind, body: (n.body || '').slice(0, 20) } : null;
  });
  assert(stored && stored.kind === 'note', 'kind not persisted: ' + JSON.stringify(stored));
  assert(stored.body.includes('今天的一个想法'), 'body not persisted');

  // 6. after reload the free note still hides experiments
  const nav = page.locator('.note-nav-item').filter({ hasText: '未命名笔记' }).first();
  if (await nav.count()) { await nav.click(); await page.waitForTimeout(500); }
  assert(await page.locator('#write-records-heading').count() === 0,
    'free note should still hide EXPERIMENTS after reload');

  await page.screenshot({
    path: path.resolve('D:/codex_project/vllm/lab/free-note.png'),
    fullPage: false,
  });

  assert.equal(errors.length, 0, 'runtime errors: ' + errors.join(' | '));
  console.log(JSON.stringify({ ok: true, stored, errors }, null, 2));
  await browser.close();
})().catch((e) => { console.error('FREE NOTE TEST FAILED:', e.message); process.exit(1); });
