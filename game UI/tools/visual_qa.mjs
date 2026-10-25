import { chromium } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

const base = process.env.BOMBERMAN_BASE_URL || 'http://127.0.0.1:8000';
const folder = path.resolve('artifacts/screenshots');
await fs.mkdir(folder, {recursive:true});
const browser = await chromium.launch({executablePath:process.env.BOMBERMAN_CHROME || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',headless:true,
  args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
const consoleErrors = [];
page.on('pageerror', error => consoleErrors.push(error.message));
try {
  await page.goto(base); await page.evaluate(()=>document.fonts.ready);
  await page.screenshot({path:path.join(folder,'01-menu-desktop.png'),fullPage:true});
  await page.getByRole('button',{name:/Chơi ngay/}).click();
  await page.getByRole('button',{name:/Bắt đầu ván/}).waitFor({state:'visible'});
  await page.screenshot({path:path.join(folder,'02-setup.png'),fullPage:true});
  await page.getByRole('button',{name:'Đóng',exact:true}).click();
  await page.getByRole('button',{name:/Cài đặt/}).click();
  await page.screenshot({path:path.join(folder,'03-settings.png'),fullPage:true});
  await page.getByRole('button',{name:'Xong',exact:true}).click();
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(folder,'04-menu-mobile.png'),fullPage:true});
  await page.setViewportSize({width:1366,height:768});
  const catalog = await (await page.request.get(base+'/api/models')).json();
  const fixtures = JSON.parse(await fs.readFile('tests/fixtures/scenes.json','utf8'));
  for (const [name,fixture] of Object.entries(fixtures)) {
    const response = await page.request.post(base+'/api/sessions',{data:{mode:'watch',model_id:catalog.models[0].id,num_agents:4,max_steps:500}});
    if(!response.ok()) throw new Error(await response.text());
    const created=await response.json();
    const route=`**/api/sessions/${created.session_id}/ws`;
    await page.routeWebSocket(route, ws=> {
      ws.send(JSON.stringify({type:'hello',protocol_version:1,session_id:created.session_id}));
      ws.send(JSON.stringify({type:'snapshot',state:{...fixture,session_id:created.session_id,match_id:created.session_id,revision:200}}));
      ws.onMessage(message=>{if(JSON.parse(String(message)).type==='ping')ws.send(JSON.stringify({type:'pong'}));});
    });
    await page.route('**/api/sessions',route=>route.request().method()==='POST'?route.fulfill({json:created}):route.continue());
    await page.getByRole('button',{name:/Xem AI thi đấu/}).click();
    await page.getByRole('button',{name:/Bắt đầu ván/}).click();
    await page.locator('canvas').waitFor();
    await page.waitForTimeout(500);
    if (name === 'ring-warning') await page.locator('#ring-warning').waitFor({state:'visible'});
    if (name === 'draw-result') await page.locator('#result-card').waitFor({state:'visible'});
    await page.screenshot({path:path.join(folder,`fixture-${name}.png`),fullPage:true});
    await page.getByRole('button',{name:'Về menu',exact:true}).first().click();
    await page.unroute('**/api/sessions');
    await page.request.delete(base+`/api/sessions/${created.session_id}`);
  }
  await page.reload();
  await page.getByRole('button',{name:/Xem AI thi đấu/}).click();
  const createPromise=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/api/sessions'));
  await page.getByRole('button',{name:/Bắt đầu ván/}).click();
  const created=await (await createPromise).json();
  await page.locator('canvas').waitFor();
  const timings=[];
  for(let i=0;i<12;i++) {
    if((await (await page.request.get(base+`/api/sessions/${created.session_id}`)).json()).snapshot.done) break;
    const started=performance.now();
    await page.getByRole('button',{name:'Một bước',exact:true}).click();
    await page.waitForFunction(async([url,tick]) => (await (await fetch(url)).json()).snapshot.tick>=tick,
      [`/api/sessions/${created.session_id}`,i+1]);
    const snapshot=(await (await page.request.get(base+`/api/sessions/${created.session_id}`)).json()).snapshot;
    timings.push({bridge_ms:performance.now()-started,engine_ms:snapshot.tick_ms});
  }
  await page.screenshot({path:path.join(folder,'05-live-gameplay.png'),fullPage:true});
  const responsive=[];
  for(const [width,height] of [[1366,768],[1920,1080],[768,1024],[390,844],[844,390]]) {
    await page.setViewportSize({width,height});
    const bounds=await page.locator('canvas').boundingBox();
    const fits=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1);
    if(!bounds||bounds.x<0||bounds.x+bounds.width>width+1||!fits)throw new Error(`Board overflow at ${width}x${height}`);
    responsive.push({width,height,board:bounds,horizontalOverflow:!fits});
    if(width===390)await page.screenshot({path:path.join(folder,'06-live-gameplay-mobile.png'),fullPage:true});
  }
  await page.setViewportSize({width:1366,height:768});
  await page.keyboard.press('Escape');
  await page.locator('#pause-dialog').waitFor({state:'visible'});
  await page.screenshot({path:path.join(folder,'07-live-pause.png'),fullPage:true});
  await page.keyboard.press('Escape');
  await page.context().setOffline(true);
  await page.locator('#connection-banner').waitFor({state:'visible'});
  await page.screenshot({path:path.join(folder,'08-live-disconnect.png'),fullPage:true});
  await page.context().setOffline(false);
  await page.waitForFunction(()=>document.querySelector('#connection')?.textContent==='Đã kết nối');
  const raf=await page.evaluate(()=>new Promise(resolve=>{
    let start=0,last=0;const spans=[];function tick(now){if(!start)start=now;if(last)spans.push(now-last);last=now;
      if(spans.length<120)requestAnimationFrame(tick);else resolve({fps:1000/((now-start)/120),frames:spans.length});}
    requestAnimationFrame(tick);
  }));
  const resources=await page.evaluate(()=>performance.getEntriesByType('resource').filter(e=>!e.name.includes('/api/')).map(e=>({name:e.name,bytes:e.transferSize})));
  const report={environment:'Headless Chrome software WebGL; timings include browser automation; RAF measures browser refresh, not Phaser FPS',timings,raf,responsive,resources,consoleErrors,
    screenshots:'fixture-* are controlled snapshots from real engine fixtures, not live policy matches; browser test screenshots are actual matches'};
  await fs.writeFile('artifacts/visual-qa.json',JSON.stringify(report,null,2));
  await page.getByRole('button',{name:'Về menu',exact:true}).first().click();
  console.log(JSON.stringify({screenshots:folder,steps:timings.length,raf,consoleErrors}));
} catch(error) {
  await page.screenshot({path:path.join(folder,'qa-error.png'),fullPage:true}).catch(()=>{});
  await fs.writeFile('artifacts/visual-qa-error.txt',String(error)+'\n'+await page.locator('body').innerText());
  throw error;
} finally {await browser.close();}
