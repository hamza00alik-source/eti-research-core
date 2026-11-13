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
 showAccount(data.account);await loadWorkspaces();
}
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
async function loadWorkspaces(){
 await loadCapabilities();
 const data=await api('/v1/workspaces');
 const select=$('#workspace-select');select.replaceChildren(new Option('Select workspace',''));
 for(const workspace of data.items)select.add(new Option(workspace.name,workspace.id));
 if(data.items.length){
  const saved=data.items.find(item=>item.id===activeWorkspace);
  activeWorkspace=(saved||data.items[0]).id;select.value=activeWorkspace;
  await loadWorkspaceData();
 }else{
  activeWorkspace='';$('#workspace-tools').hidden=true;
  $('#memory-list').textContent='Create a workspace to begin storing research memory.';
  $('#task-list').textContent='No workspace selected.';
  $('#audit-list').textContent='No audit events yet.';
 }
}
async function loadWorkspaceData(){
 if(!activeWorkspace)return;
 $('#workspace-tools').hidden=false;
 const [memory,tasks,audit]=await Promise.all([
  api('/v1/workspaces/'+encodeURIComponent(activeWorkspace)+'/memory'),
  api('/v1/workspaces/'+encodeURIComponent(activeWorkspace)+'/agents/tasks'),
  api('/v1/audit/events?limit=100')
 ]);
 $('#memory-list').textContent=JSON.stringify(memory.items,null,2);
 $('#task-list').textContent=JSON.stringify(tasks.items,null,2);
 $('#audit-list').textContent=JSON.stringify(audit.items,null,2);
}
$('#login-form').addEventListener('submit',async event=>{
 event.preventDefault();const form=new FormData(event.currentTarget);
 $('#account-message').textContent='Signing in…';
 try{
  const data=await api('/v1/auth/token',{method:'POST',body:JSON.stringify({
   username:form.get('username'),password:form.get('password')
  })});
  await acceptSession(data);event.currentTarget.reset();
 }catch(error){showAuth(error.message)}
});
$('#register-form').addEventListener('submit',async event=>{
 event.preventDefault();const form=new FormData(event.currentTarget);
 $('#account-message').textContent='Creating invited account…';
 try{
  const data=await api('/v1/auth/register',{method:'POST',body:JSON.stringify({
   username:form.get('username'),password:form.get('password'),invite_code:form.get('invite_code')
  })});
  await acceptSession(data);event.currentTarget.reset();
 }catch(error){showAuth(error.message)}
});
$('#signout').addEventListener('click',()=>{
 clearSession();showAuth('You have signed out. Sign in to access private research data.');
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
$('#task-form').addEventListener('submit',async event=>{
 event.preventDefault();if(!activeWorkspace)return;
 const form=new FormData(event.currentTarget);let input;
 try{input=JSON.parse(form.get('input'))}catch(error){$('#task-list').textContent='Task input must be valid JSON.';return}
 try{
  await api('/v1/workspaces/'+encodeURIComponent(activeWorkspace)+'/agents/tasks',{method:'POST',body:JSON.stringify({
   target_capability:form.get('capability'),input
  })});
  await loadWorkspaceData();
  $('#account-message').textContent='Task recorded in the durable queue. No agent has executed it yet.';
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
