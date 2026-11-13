const $=selector=>document.querySelector(selector);
const TOKEN_KEY='eti.session.v1';
let accessToken=sessionStorage.getItem(TOKEN_KEY)||'';
let activeAccount=null;
let activeWorkspace='';

async function api(path,options={}){
 const headers={...(options.headers||{})};
 if(accessToken)headers.Authorization='Bearer '+accessToken;
 if(options.body)headers['Content-Type']='application/json';
 const response=await fetch(path,{...options,headers,cache:'no-store'});
 const raw=await response.text();
 let data={};
 try{data=raw?JSON.parse(raw):{}}catch(e){data={detail:raw||'Unexpected response'}}
 if(!response.ok){
  if(response.status===401&&path!=='/v1/auth/token'&&path!=='/v1/auth/register'){
   clearSession();showAuth('Your session expired. Please sign in again.');
  }
  throw new Error(data.detail||('Request failed ('+response.status+')'));
 }
 return data;
}
function showAuth(message='Sign in to access your private ETI workspace.'){
 $('#auth-forms').hidden=false;$('#account-view').hidden=true;
 $('#account-message').textContent=message;
}
function showAccount(account){
 activeAccount=account;$('#auth-forms').hidden=true;$('#account-view').hidden=false;
 $('#account-name').textContent=account.username;
 $('#account-message').textContent='Signed in. Your private data is scoped to this account.';
}
function clearSession(){
 accessToken='';activeAccount=null;activeWorkspace='';
 sessionStorage.removeItem(TOKEN_KEY);
 $('#workspace-select').replaceChildren(new Option('Select workspace',''));
 $('#workspace-tools').hidden=true;
}
async function acceptSession(data){
 accessToken=data.access_token;sessionStorage.setItem(TOKEN_KEY,accessToken);
 showAccount(data.account);await loadWorkspaces();await loadAutonomy();
}
let autonomyEnabled=false;
async function loadAutonomy(){
 const status=$('#autonomy-status'),button=$('#autonomy-toggle');
 if(!status||!button)return;
 try{
  const data=await api('/v1/autonomy');autonomyEnabled=Boolean(data.profile?.enabled);
  status.textContent=autonomyEnabled?'Autonomous research is ON · research only':'Autonomous research is OFF';
  button.textContent=autonomyEnabled?'Pause autonomous research':'Enable autonomous research';
 }catch(error){status.textContent='Autonomy status unavailable: '+error.message;button.disabled=true}
}
$('#autonomy-toggle')?.addEventListener('click',async()=>{
 const button=$('#autonomy-toggle'),status=$('#autonomy-status');
 button.disabled=true;status.textContent='Updating…';
 try{
  const data=await api('/v1/autonomy',{method:'PUT',body:JSON.stringify({enabled:!autonomyEnabled,workspace_id:activeWorkspace||null})});
  autonomyEnabled=Boolean(data.profile.enabled);
  status.textContent=autonomyEnabled?'Autonomous research is ON · scans run every 15 minutes':'Autonomous research is OFF';
  button.textContent=autonomyEnabled?'Pause autonomous research':'Enable autonomous research';
  await loadWorkspaceData();
 }catch(error){status.textContent='Could not update autonomy: '+error.message}
 finally{button.disabled=false}
});
async function loadCapabilities(){
 const select=$('#capability-select');
 if(!select)return;
 try{
  const data=await api('/v1/agents/capabilities');
  const previous=select.value;
  select.replaceChildren();
  for(const capability of data.items)select.add(new Option(capability,capability));
  if(data.items.includes(previous))select.value=previous;
  if(!data.items.length)select.add(new Option('No capabilities available',''));
 }catch(error){
  select.replaceChildren(new Option('Capabilities unavailable',''));
  $('#account-message').textContent='Could not load registered agent capabilities: '+error.message;
 }
}
async function loadModels(){
 try{
  const data=await api('/v1/models');
  $('#model-list').textContent=data.items.length
   ? JSON.stringify(data.items,null,2)
   : 'No model versions are registered yet.';
 }catch(error){
  $('#model-list').textContent='Model registry unavailable: '+error.message;
 }
}
async function loadWorkspaces(){
 await loadCapabilities();
 const data=await api('/v1/workspaces');
 const select=$('#workspace-select');select.replaceChildren(new Option('Select workspace',''));
 for(const workspace of data.items)select.add(new Option(workspace.name,workspace.id));
 if(data.items.length){
  const saved=data.items.find(item=>item.id===activeWorkspace);
  activeWorkspace=(saved||data.items[0]).id;select.value=activeWorkspace;
  await loadWorkspaceData();
  await loadAutonomy();
 }else{
  activeWorkspace='';$('#workspace-tools').hidden=true;
  if($('#autonomy-status'))$('#autonomy-status').textContent='Create or select a workspace to enable research.';
  $('#memory-list').textContent='Create a workspace to begin storing research memory.';
  $('#task-list').textContent='No workspace selected.';
  $('#audit-list').textContent='No audit events yet.';
 }
 await loadModels();
}
async function loadWorkspaceData(){
 if(!activeWorkspace)return;
 $('#workspace-tools').hidden=false;
 const [memory,tasks,audit,workerStatus]=await Promise.all([
  api('/v1/workspaces/'+encodeURIComponent(activeWorkspace)+'/memory'),
  api('/v1/workspaces/'+encodeURIComponent(activeWorkspace)+'/agents/tasks'),
  api('/v1/audit/events?limit=100'),
  api('/v1/agents/status')
 ]);
 $('#memory-list').textContent=JSON.stringify(memory.items,null,2);
 $('#task-list').textContent=JSON.stringify(tasks.items,null,2);
 $('#audit-list').textContent=JSON.stringify(audit.items,null,2);
 $('#worker-status').textContent=workerStatus.online_count
  ? JSON.stringify(workerStatus.items.filter(item=>item.online),null,2)
  : 'No active worker heartbeat. Start the separate worker process on the ETI backend.';
}
$('#login-form').addEventListener('submit',async event=>{
 event.preventDefault();const formElement=event.currentTarget;const form=new FormData(formElement);
 $('#account-message').textContent='Signing in…';
 try{
  const data=await api('/v1/auth/token',{method:'POST',body:JSON.stringify({
   username:form.get('username'),password:form.get('password')
  })});
  await acceptSession(data);formElement.reset();
 }catch(error){showAuth(error.message)}
});
$('#register-form').addEventListener('submit',async event=>{
 event.preventDefault();const formElement=event.currentTarget;const form=new FormData(formElement);
 $('#account-message').textContent='Creating invited account…';
 try{
  const data=await api('/v1/auth/register',{method:'POST',body:JSON.stringify({
   username:form.get('username'),password:form.get('password'),invite_code:form.get('invite_code')
  })});
  await acceptSession(data);formElement.reset();
 }catch(error){showAuth(error.message)}
});
$('#signout').addEventListener('click',()=>{
 clearSession();showAuth('You have signed out. Sign in to access private research data.');
});
$('#export-account').addEventListener('click',async()=>{
 const button=$('#export-account');button.disabled=true;
 try{
  const data=await api('/v1/account/export');
  const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob);
  const link=document.createElement('a');
  link.href=url;
  link.download='eti-account-export-'+new Date().toISOString().slice(0,10)+'.json';
  document.body.appendChild(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
  $('#account-message').textContent='Your account data export was prepared. Keep the downloaded JSON file private.';
 }catch(error){$('#account-message').textContent='Export failed: '+error.message}
 finally{button.disabled=false}
});
$('#delete-workspace').addEventListener('click',async()=>{
 if(!activeWorkspace)return;
 const select=$('#workspace-select');
 const name=select.options[select.selectedIndex]?.textContent||'this workspace';
 if(!window.confirm('Permanently delete "'+name+'" and its memory, tasks and messages? The deletion is audited and cannot be undone.'))return;
 const button=$('#delete-workspace');button.disabled=true;
 try{
  await api('/v1/workspaces/'+encodeURIComponent(activeWorkspace),{method:'DELETE'});
  activeWorkspace='';await loadWorkspaces();
  $('#account-message').textContent='Workspace deleted. Its deletion event remains in your audit history.';
 }catch(error){$('#account-message').textContent='Workspace deletion failed: '+error.message}
 finally{button.disabled=false}
});
$('#workspace-form').addEventListener('submit',async event=>{
 event.preventDefault();const form=new FormData(event.currentTarget);
 try{
  const data=await api('/v1/workspaces',{method:'POST',body:JSON.stringify({name:form.get('name')})});
  activeWorkspace=data.id;await loadWorkspaces();event.currentTarget.reset();
  $('#account-message').textContent='Workspace created.';
 }catch(error){$('#account-message').textContent=error.message}
});
$('#workspace-select').addEventListener('change',async event=>{
 activeWorkspace=event.target.value;
 if(activeWorkspace){try{await loadWorkspaceData()}catch(error){$('#account-message').textContent=error.message}}
 else $('#workspace-tools').hidden=true;
});
$('#memory-form').addEventListener('submit',async event=>{
 event.preventDefault();if(!activeWorkspace)return;
 const form=new FormData(event.currentTarget);let value;
 try{value=JSON.parse(form.get('value'))}catch(error){$('#memory-list').textContent='Value must be valid JSON.';return}
 try{
  await api('/v1/workspaces/'+encodeURIComponent(activeWorkspace)+'/memory/'+encodeURIComponent(form.get('namespace'))+'/'+encodeURIComponent(form.get('key')),{method:'PUT',body:JSON.stringify({value})});
  await loadWorkspaceData();$('#account-message').textContent='Memory saved to this workspace.';
 }catch(error){$('#memory-list').textContent=error.message}
});
$('#refresh-models').addEventListener('click',loadModels);
$('#model-register-form').addEventListener('submit',async event=>{
 event.preventDefault();
 let payload;
 try{payload=JSON.parse(new FormData(event.currentTarget).get('payload'))}
 catch(error){$('#model-message').textContent='Registration payload must be valid JSON.';return}
 try{
  const result=await api('/v1/models',{method:'POST',body:JSON.stringify(payload)});
  $('#model-message').textContent='Candidate registered: '+result.model_id+' '+result.version+'. It is not approved for inference.';
  await loadModels();
 }catch(error){$('#model-message').textContent='Candidate registration failed: '+error.message}
});
$('#model-approve-form').addEventListener('submit',async event=>{
 event.preventDefault();const form=new FormData(event.currentTarget);
 const modelId=encodeURIComponent(form.get('model_id'));
 const version=encodeURIComponent(form.get('version'));
 try{
  const result=await api('/v1/models/'+modelId+'/'+version+'/approve',{method:'POST'});
  $('#model-message').textContent='Model approved: '+result.model_id+' '+result.version+'.';
  await loadModels();
 }catch(error){$('#model-message').textContent='Approval failed: '+error.message+'. The reviewer must differ from the creator and the artifact must pass integrity, schema and calibration checks.'}
});
$('#model-retire-form').addEventListener('submit',async event=>{
 event.preventDefault();const form=new FormData(event.currentTarget);
 const modelId=encodeURIComponent(form.get('model_id'));
 const version=encodeURIComponent(form.get('version'));
 try{
  const result=await api('/v1/models/'+modelId+'/'+version+'/retire',{method:'POST'});
  $('#model-message').textContent='Model retired: '+result.model_id+' '+result.version+'.';
  await loadModels();
 }catch(error){$('#model-message').textContent='Retirement failed: '+error.message}
});
$('#task-form').addEventListener('submit',async event=>{
 event.preventDefault();if(!activeWorkspace)return;
 const form=new FormData(event.currentTarget);let input;
 try{input=JSON.parse(form.get('input'))}catch(error){$('#task-list').textContent='Task input must be valid JSON.';return}
 try{
  await api('/v1/workspaces/'+encodeURIComponent(activeWorkspace)+'/agents/tasks',{method:'POST',body:JSON.stringify({
   target_capability:form.get('capability'),input
  })});
  await loadWorkspaceData();
  $('#account-message').textContent='Task queued in the durable ledger. A registered worker can process it when online.';
 }catch(error){$('#task-list').textContent=error.message}
});
async function restoreSession(){
 if(!accessToken){showAuth();return}
 try{const data=await api('/v1/profile');showAccount(data.account);await loadWorkspaces()}
 catch(error){clearSession();showAuth('Please sign in to continue.')}
}
async function loadStatus(){
 try{
  const [h,e,m,p]=await Promise.all([
   fetch('/health',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('health');return r.json()}),
   fetch('/execution/status',{cache:'no-store'}).then(r=>r.json()),
   fetch('/markets/supported',{cache:'no-store'}).then(r=>r.json()),
   fetch('/providers/capabilities',{cache:'no-store'}).then(r=>r.json())
  ]);
  $('#health').textContent=h.status+' · '+h.mode;
  $('#execution').textContent=e.enabled?'Enabled':'Disabled';
  $('#markets').textContent=m.asset_classes.join(', ')+' · '+m.venues.join(', ');
  $('#providers').textContent=JSON.stringify(p,null,2);
 }catch(err){
  $('#health').textContent='API unavailable';
  $('#providers').textContent='Connect to the ETI research backend to view provider status.';
 }
}
$('#decision-form').addEventListener('submit',async ev=>{
 ev.preventDefault();const f=new FormData(ev.target);const num=k=>Number(f.get(k));
 const out=$('#decision-result');out.textContent='Assessing…';
 let positions;
 try{positions=JSON.parse(f.get('positions'))}
 catch(e){out.textContent='Positions must be valid JSON, for example {"BTC/USDT":50}';return}
 const body={
  symbol:f.get('symbol'),equity:num('equity'),entry_price:num('entry_price'),
  stop_price:num('stop_price'),requested_notional:num('requested_notional'),
  gross_exposure:num('gross_exposure'),current_positions:positions,
  correlated_group_exposure:num('correlated_group_exposure'),
  probabilities:{short:num('short'),flat:num('flat'),long:num('long')}
 };
 try{
  const response=await fetch('/decision/assess',{method:'POST',
   headers:{'Content-Type':'application/json'},body:JSON.stringify(body),cache:'no-store'});
  const data=await response.json();
  if(!response.ok)throw new Error(data.detail||'Assessment failed');
  out.textContent=JSON.stringify(data,null,2);
 }catch(error){out.textContent='Could not complete assessment: '+error.message+'. No order was placed.'}
});
const navLinks=[...document.querySelectorAll('.bottom-nav a')];
const updateNav=()=>{
 const hash=location.hash||'#home';
 navLinks.forEach(a=>a.classList.toggle('active',a.getAttribute('href')===hash));
};
window.addEventListener('hashchange',updateNav);updateNav();
if('serviceWorker'in navigator){
 window.addEventListener('load',()=>navigator.serviceWorker.register('/sw.js').catch(()=>{}));
}
loadStatus();restoreSession();
