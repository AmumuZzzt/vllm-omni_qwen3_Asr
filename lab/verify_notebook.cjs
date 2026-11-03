const { chromium } = require('playwright');
const path = require('node:path');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');

const NOTE_TITLE = 'vLLM 0.26 学习环境与 #5714 capability probe 复现';
const EXPECTED_RECORDS = 11;

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
  });
  // fresh context => empty localStorage => SEED is read
  const context = await browser.newContext({ viewport: { width: 1440, height: 1200 } });
  const page = await context.newPage();

  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

  await page.goto(pathToFileURL(
    path.resolve('D:/pi-project/notes/vllm/omni-learning-site/notebook.html')
  ).href);
  await page.waitForTimeout(700);

  assert.equal(errors.length, 0, 'page errors: ' + errors.join(' | '));
  assert((await page.locator('body').innerText()).includes(NOTE_TITLE),
    'note title missing from list');

  await page.getByText(NOTE_TITLE).first().click();
  await page.waitForTimeout(900);

  const ids = await page.evaluate(() => [...new Set(
    [...document.querySelectorAll('[data-record-id]')].map((e) => e.dataset.recordId)
  )]);
  assert.equal(ids.length, EXPECTED_RECORDS, 'record count mismatch: ' + ids.join(','));

  const text = await page.locator('body').innerText();
  for (const label of [
    '0.32 参考环境的版本与能力探测',
    'out-of-tree qwen3_asr pipeline 冒烟',
    'EngineCore 启动失败的真实根因是 ninja 缺失',
    'study 分支 HEAD 与上游 21495f9e 的 stable patch-id 比对',
    'pip 直装 vllm==0.26.0：60 分钟仅 4.84 GB',
    '改用 uv 并行安装 vllm 0.26.0',
    '运行 21495f9e 自带的 capability probe 测试',
  ]) {
    assert(text.includes(label), 'record label missing: ' + label);
  }

  // record body (command/result) is collapsed by default; expand the key ones
  for (const id of ['lab026-pytest', 'lab026-patchid', 'lab026-uv-install', 'lab026-editable']) {
    await page.locator(`[data-record-toggle="${id}"]`).click();
    await page.waitForTimeout(200);
  }
  const opened = await page.locator('body').innerText();
  assert(opened.includes('21 passed'), 'pytest result not rendered');
  assert(opened.includes('fc29c9c67459312565d21cbb61f3def5af5195f5'), 'patch-id not rendered');
  assert(opened.includes('vllm-omni==0.26.1.dev12+g18649cf74'), 'editable install result not rendered');

  await page.screenshot({
    path: path.resolve('D:/codex_project/vllm/lab/notebook-lab-note.png'),
    fullPage: false,
  });

  console.log(JSON.stringify({
    ok: true,
    noteTitle: NOTE_TITLE,
    records: ids.length,
    errors,
    screenshot: 'lab/notebook-lab-note.png',
  }, null, 2));

  await browser.close();
})().catch((err) => {
  console.error('VERIFY FAILED:', err.message);
  process.exit(1);
});
