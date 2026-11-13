const $=selector=>document.querySelector(selector);
async function loadStatus(){
 try{
  const [h,e,m,p]=await Promise.all([
   fetch('/health').then(r=>{if(!r.ok)throw Error('health');return r.json()}),
   fetch('/execution/status').then(r=>r.json()),
   fetch('/markets/supported').then(r=>r.json()),
   fetch('/providers/capabilities').then(r=>r.json())
  ]);
  $('#health').textContent=h.status+' · '+h.mode;
  $('#execution').textContent=e.enabled?'Enabled':'Disabled';
  $('#markets').textContent=m.asset_classes.join(', ')+' · '+m.venues.join(', ');
  $('#providers').textContent=JSON.stringify(p,null,2);
 }catch(err){$('#health').textContent='API unavailable';$('#providers').textContent='Connect to the ETI research backend to view provider status.'}
}
$('#decision-form').addEventListener('submit',async ev=>{
 ev.preventDefault();const f=new FormData(ev.target);const num=k=>Number(f.get(k));const out=$('#decision-result');out.textContent='Assessing…';
 let positions;try{positions=JSON.parse(f.get('positions'))}catch(e){out.textContent='Positions must be valid JSON, for example {"BTC/USDT":50}';return}
 const body={symbol:f.get('symbol'),equity:num('equity'),entry_price:num('entry_price'),stop_price:num('stop_price'),requested_notional:num('requested_notional'),gross_exposure:num('gross_exposure'),current_positions:positions,correlated_group_exposure:num('correlated_group_exposure'),probabilities:{short:num('short'),flat:num('flat'),long:num('long')}};
 try{const r=await fetch('/decision/assess',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();out.textContent=JSON.stringify(data,null,2)}catch(e){out.textContent='Could not reach ETI API. No order was placed.'}
});
const navLinks=[...document.querySelectorAll('.bottom-nav a')];
const updateNav=()=>{const hash=location.hash||'#home';navLinks.forEach(a=>a.classList.toggle('active',a.getAttribute('href')===hash))};
window.addEventListener('hashchange',updateNav);updateNav();
if('serviceWorker'in navigator)window.addEventListener('load',()=>navigator.serviceWorker.register('/sw.js').catch(()=>{}));
loadStatus();