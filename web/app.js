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
let autonomyProfile=null;
let modelTrainingEnabled=false;
let modelTrainingIntervalSeconds=86400;
async function saveAutonomyConfig(patch={}){
 const current=autonomyProfile||{};
 const next={...current,...patch};
 const data=await api('/v1/autonomy',{method:'PUT',body:JSON.stringify({
  enabled:next.enabled??autonomyEnabled,
  workspace_id:activeWorkspace||null,
  venues:next.venues||['mexc'],
  symbols:next.symbols||['BTC/USDT','ETH/USDT','SOL/USDT','XRP/USDT'],
  timeframes:next.timeframes||['1h'],
  interval_seconds:next.interval_seconds||900,
  model_training_enabled:next.model_training_enabled??modelTrainingEnabled,
  model_training_interval_seconds:next.model_training_interval_seconds||modelTrainingIntervalSeconds
 })});
 autonomyProfile=data.profile;
 autonomyEnabled=Boolean(data.profile.enabled);
 modelTrainingEnabled=Boolean(data.profile.model_training_enabled);
 modelTrainingIntervalSeconds=Number(data.profile.model_training_interval_seconds||86400);
 return data.profile;
}
async function loadAutonomy(){
 const status=$('#autonomy-status'),button=$('#autonomy-toggle');
 if(!status||!button)return;
 try{
  const data=await api('/v1/autonomy');autonomyProfile=data.profile||{};
  autonomyEnabled=Boolean(autonomyProfile.enabled);
  modelTrainingEnabled=Boolean(autonomyProfile.model_training_enabled);
  modelTrainingIntervalSeconds=Number(autonomyProfile.model_training_interval_seconds||86400);
  status.textContent=!autonomyEnabled?'Autonomous operation is OFF':modelTrainingEnabled?'Autonomous operation is ON · market scans + daily model training · research only':'Market scans are ON · daily model training is OFF';
  button.textContent=autonomyEnabled&&modelTrainingEnabled?'Pause autonomous operation':autonomyEnabled?'Complete autonomous setup':'Start autonomous operation';
  const trainingButton=$('#model-training-toggle');
  const trainingStatus=$('#model-training-status');
  const cadence=$('#model-training-interval');
  if(trainingButton){
   trainingButton.disabled=!autonomyEnabled;
   trainingButton.textContent=modelTrainingEnabled?'Pause daily model training':'Enable daily model training';
  }
  if(trainingStatus)trainingStatus.textContent=!autonomyEnabled
   ?'Enable autonomous research first'
   :modelTrainingEnabled?'Scheduled training is ON · research candidates only':'Scheduled training is OFF';
  if(cadence)cadence.value=String(modelTrainingIntervalSeconds);
 }catch(error){status.textContent='Autonomy status unavailable: '+error.message;button.disabled=true}
}
$('#autonomy-toggle')?.addEventListener('click',async()=>{
 const button=$('#autonomy-toggle'),status=$('#autonomy-status');
 // A single owner action starts/stops the complete research + daily training loop.
 // This never enables paper or live order execution.
 const nextEnabled=!(autonomyEnabled&&modelTrainingEnabled);
 button.disabled=true;status.textContent='Updating autonomous operation…';
 try{
  await saveAutonomyConfig({enabled:nextEnabled,model_training_enabled:nextEnabled});
  await loadAutonomy();
  await loadWorkspaceData();
 }catch(error){status.textContent='Could not update autonomous operation: '+error.message}
 finally{button.disabled=false}
});
$('#model-training-toggle')?.addEventListener('click',async()=>{
 const button=$('#model-training-toggle'),status=$('#model-training-status');
 if(!autonomyEnabled){status.textContent='Enable autonomous research first';return}
 button.disabled=true;status.textContent='Updating…';
 try{
  await saveAutonomyConfig({model_training_enabled:!modelTrainingEnabled});
  await loadAutonomy();
  await loadWorkspaceData();
 }catch(error){status.textContent='Could not update model training: '+error.message}
 finally{button.disabled=false}
});
$('#model-training-interval')?.addEventListener('change',async event=>{
 modelTrainingIntervalSeconds=Number(event.target.value);
 if(!autonomyEnabled||!modelTrainingEnabled)return;
 const status=$('#model-training-status');
 status.textContent='Saving cadence…';
 try{await saveAutonomyConfig({model_training_interval_seconds:modelTrainingIntervalSeconds});await loadAutonomy()}
 catch(error){status.textContent='Could not update cadence: '+error.message}
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
async function loadPaperSessions(){
 const list=$('#paper-session-list');
 if(!list||!activeWorkspace)return;
 try{
  const data=await api('/v1/paper/sessions?workspace_id='+encodeURIComponent(activeWorkspace));
  list.replaceChildren();
  if(!data.items.length){list.textContent='No paper sessions yet.';return}
  for(const session of data.items){
   const card=document.createElement('section');card.className='paper-session';
   const title=document.createElement('h4');title.textContent=session.symbol+' · '+session.timeframe+' · '+session.status.toUpperCase();
   const details=document.createElement('p');
   details.textContent='Model '+session.model_id+' '+session.model_version+
    ' · Equity '+Number(session.equity).toFixed(2)+' · Cash '+Number(session.cash).toFixed(2)+
    ' · Realized P/L '+Number(session.realized_pnl).toFixed(2)+
    ' · Max drawdown '+(Number(session.max_drawdown)*100).toFixed(2)+'%';
   const historyView=document.createElement('pre');historyView.textContent='Trade history not loaded.';
   card.append(title,details,historyView);
   if(session.status!=='stopped'){
    const button=document.createElement('button');
    button.type='button';button.className='secondary-button';
    button.textContent=session.status==='running'?'Pause paper session':'Start paper session';
    button.addEventListener('click',async()=>{
     button.disabled=true;
     try{
      await api('/v1/paper/sessions/'+encodeURIComponent(session.id)+'/'+(session.status==='running'?'pause':'start'),{method:'POST'});
      await loadPaperSessions();
     }catch(error){$('#paper-message').textContent='Could not update paper session: '+error.message}
     finally{button.disabled=false}
    });
    card.append(button);
   }
   const trades=document.createElement('button');trades.type='button';trades.className='secondary-button';
   trades.textContent='Refresh trade history';
   trades.addEventListener('click',async()=>{
    try{
     const history=await api('/v1/paper/sessions/'+encodeURIComponent(session.id)+'/trades');
     historyView.textContent=JSON.stringify(history.items,null,2);
     trades.textContent='Trade history refreshed';
    }catch(error){$('#paper-message').textContent=error.message}
   });
   card.append(trades);list.append(card);
  }
 }catch(error){list.textContent='Paper sessions unavailable: '+error.message}
}
async function loadExchangeCredentials(){
 const list=$('#exchange-credential-list');
 if(!list)return;
 try{
  const data=await api('/v1/exchanges/credentials');
  list.replaceChildren();
  if(!data.items.length){list.textContent='No exchange credentials connected.';return}
  for(const item of data.items){
   const row=document.createElement('div');row.className='exchange-credential-item';
   const detail=document.createElement('span');
   detail.textContent=item.venue.toUpperCase()+' · '+item.masked_api_key+' · encrypted at rest';
   const remove=document.createElement('button');remove.type='button';remove.className='secondary-button';
   remove.textContent='Remove';remove.addEventListener('click',async()=>{
    if(!window.confirm('Remove the saved '+item.venue.toUpperCase()+' API credentials from ETI?'))return;
    remove.disabled=true;
    try{await api('/v1/exchanges/credentials/'+encodeURIComponent(item.venue),{method:'DELETE'});await loadExchangeCredentials()}
    catch(error){$('#exchange-credential-message').textContent='Could not remove credentials: '+error.message}
    finally{remove.disabled=false}
   });
   row.append(detail,remove);list.append(row);
  }
 }catch(error){list.textContent='Exchange connection list unavailable: '+error.message}
}

async function loadLiveExecution(){
 const status=$('#live-execution-status'),form=$('#live-execution-form');
 if(!form)return;
 try{
  const data=await api('/v1/execution/profile'),profile=data.profile;
  if(profile){
   form.elements.venue.value=profile.venue;
   form.elements.timeframe.value=profile.timeframe;
   form.elements.symbols.value=profile.symbols.join(', ');
   form.elements.capital_allocation_usdt.value=profile.capital_allocation_usdt;
   const risk=profile.risk_config;
   form.elements.max_risk_per_trade.value=(risk.max_risk_per_trade*100).toFixed(1);
   form.elements.max_position_fraction.value=(risk.max_position_fraction*100).toFixed(0);
   form.elements.max_gross_exposure.value=(risk.max_gross_exposure*100).toFixed(0);
   form.elements.max_daily_loss.value=(risk.max_daily_loss*100).toFixed(1);
   form.elements.max_drawdown.value=(risk.max_drawdown*100).toFixed(1);
   form.elements.confidence_threshold.value=(risk.confidence_threshold*100).toFixed(0);
  }
  const enabled=Boolean(profile&&profile.enabled);
  status.textContent=enabled
   ?'LIVE TRADING AUTHORIZED · ETI manages entries and exits automatically within your saved limits. No per-trade approval is required.'
   :profile?'Settings saved · live trading is disabled until you authorize once.':'Live trading is disabled. Save your watchlist and risk limits to continue.';
  form.querySelectorAll('input,select,button').forEach(el=>el.disabled=enabled);
  $('#authorize-live-execution').disabled=!profile||enabled;
  $('#live-kill-switch').hidden=!enabled;
  ['confirm-live-trading','confirm-spot-permission','confirm-withdrawals-disabled','confirm-ip-restrictions'].forEach(id=>{
   const input=$('#'+id);input.checked=false;input.disabled=enabled;input.parentElement.hidden=enabled;
  });
  const orders=await api('/v1/execution/orders');
  const list=$('#live-order-list');list.replaceChildren();
  if(!orders.items.length){list.textContent='No live orders recorded.'}
  else{
   for(const order of orders.items){
    const row=document.createElement('div');row.className='exchange-credential-item';
    const label=document.createElement('span');
    label.textContent=order.symbol+' · '+order.side.toUpperCase()+' · '+order.status.toUpperCase()+' · filled '+Number(order.filled||0);
    row.append(label);list.append(row);
   }
  }
 }catch(error){status.textContent='Live execution controls unavailable: '+error.message}
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
 await loadExchangeCredentials();
 await loadLiveExecution();
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
 await loadPaperSessions();
}
$('#login-form').addEventListener('submit',async event=>{
 event.preventDefault();const formElement=event.currentTarget;const form=new FormData(formElement);
 $('#account-message').textContent='Signing in…';
 try{
  const data=await api('/v1/auth/token',{method:'POST',body:JSON.stringify({
   username:form.get('username'),password:form.get('password')
  })});
  await acceptSession(data);formElement?.reset?.();
 }catch(error){showAuth(error.message)}
});
$('#register-form').addEventListener('submit',async event=>{
 event.preventDefault();const formElement=event.currentTarget;const form=new FormData(formElement);
 $('#account-message').textContent='Creating invited account…';
 try{
  const data=await api('/v1/auth/register',{method:'POST',body:JSON.stringify({
   username:form.get('username'),password:form.get('password'),invite_code:form.get('invite_code')
  })});
  await acceptSession(data);formElement?.reset?.();
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
 event.preventDefault();const formElement=event.currentTarget;const form=new FormData(formElement);
 try{
  const data=await api('/v1/workspaces',{method:'POST',body:JSON.stringify({name:form.get('name')})});
  activeWorkspace=data.id;await loadWorkspaces();formElement?.reset?.();
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
$('#live-execution-form')?.addEventListener('submit',async event=>{
 event.preventDefault();
 const form=event.currentTarget,values=new FormData(form),button=form.querySelector('button[type="submit"]');
 button.disabled=true;$('#live-execution-status').textContent='Saving live trading limits…';
 try{
  const symbols=String(values.get('symbols')).split(',').map(value=>value.trim()).filter(Boolean);
  const result=await api('/v1/execution/profile',{method:'PUT',body:JSON.stringify({
   venue:values.get('venue'),symbols,timeframe:values.get('timeframe'),
   capital_allocation_usdt:Number(values.get('capital_allocation_usdt')),
   max_risk_per_trade:Number(values.get('max_risk_per_trade'))/100,
   max_position_fraction:Number(values.get('max_position_fraction'))/100,
   max_gross_exposure:Number(values.get('max_gross_exposure'))/100,
   max_daily_loss:Number(values.get('max_daily_loss'))/100,
   max_drawdown:Number(values.get('max_drawdown'))/100,
   confidence_threshold:Number(values.get('confidence_threshold'))/100
  })});
  $('#live-execution-status').textContent=result.note;await loadLiveExecution();
 }catch(error){$('#live-execution-status').textContent='Could not save live settings: '+error.message}
 finally{button.disabled=$('#live-execution-status').textContent.startsWith('LIVE TRADING AUTHORIZED')}
});
$('#authorize-live-execution')?.addEventListener('click',async()=>{
 const confirmations={
  confirm_live_trading:$('#confirm-live-trading').checked,
  confirm_spot_trading_permission:$('#confirm-spot-permission').checked,
  confirm_withdrawals_disabled:$('#confirm-withdrawals-disabled').checked,
  confirm_ip_restrictions:$('#confirm-ip-restrictions').checked
 };
 if(!Object.values(confirmations).every(Boolean)){
  $('#live-execution-status').textContent='Complete all four one-time safety confirmations first.';return;
 }
 if(!window.confirm('Authorize ETI to place real Binance/MEXC spot orders automatically within your saved capital and risk limits? Individual trades will not require confirmation.'))return;
 const button=$('#authorize-live-execution');button.disabled=true;
 try{
  const result=await api('/v1/execution/authorize',{method:'POST',body:JSON.stringify(confirmations)});
  $('#live-execution-status').textContent=result.note;await loadLiveExecution();
 }catch(error){$('#live-execution-status').textContent='Live authorization failed: '+error.message;button.disabled=false}
});
$('#live-kill-switch')?.addEventListener('click',async()=>{
 if(!window.confirm('Stop all new ETI live trading tasks immediately? Open exchange orders may still need reconciliation at the exchange.'))return;
 const button=$('#live-kill-switch');button.disabled=true;
 try{
  const result=await api('/v1/execution/kill-switch',{method:'POST'});
  $('#live-execution-status').textContent=result.message;await loadLiveExecution();
 }catch(error){$('#live-execution-status').textContent='Kill switch failed: '+error.message}
 finally{button.disabled=false}
});

$('#exchange-credential-form')?.addEventListener('submit',async event=>{
 event.preventDefault();
 const formElement=event.currentTarget,form=new FormData(formElement);
 const message=$('#exchange-credential-message'),button=formElement.querySelector('button[type="submit"]');
 button.disabled=true;message.textContent='Verifying private read access and encrypting credentials…';
 try{
  const result=await api('/v1/exchanges/credentials',{method:'POST',body:JSON.stringify({
   venue:form.get('venue'),api_key:form.get('api_key'),api_secret:form.get('api_secret')
  })});
  formElement.reset();
  message.textContent=result.verification.read_access_verified
   ?'Private read access verified and credentials encrypted. Trading permission remains unverified; live orders are still disabled.'
   :'Credentials saved, but exchange permissions need review.';
  await loadExchangeCredentials();
 }catch(error){message.textContent='Could not connect exchange: '+error.message}
 finally{button.disabled=false}
});

$('#paper-session-form')?.addEventListener('submit',async event=>{
 event.preventDefault();
 const form=new FormData(event.currentTarget);
 const payload={
  model_id:form.get('model_id'),model_version:form.get('model_version'),
  venue:form.get('venue'),symbol:form.get('symbol'),timeframe:form.get('timeframe'),
  initial_cash:Number(form.get('initial_cash')),workspace_id:activeWorkspace||null
 };
 const message=$('#paper-message');message.textContent='Creating paused paper session…';
 try{
  const result=await api('/v1/paper/sessions',{method:'POST',body:JSON.stringify(payload)});
  message.textContent='Paper session created in PAUSED mode. Review it, then start it when ready.';
  await loadPaperSessions();
 }catch(error){message.textContent='Paper session creation failed: '+error.message}
});
const taskInputTemplates={
 'market.snapshot':{venue:'mexc',symbol:'BTC/USDT',timeframe:'1h',limit:200},
 'strategy.walk_forward':{venue:'mexc',symbol:'BTC/USDT',timeframe:'1h',limit:500,min_history:120,validation_size:80,test_size:80,step_size:80,purge_bars:2,fee_bps:6,slippage_bps:2},
 'model.train_research':{venue:'mexc',symbol:'BTC/USDT',timeframe:'1h',limit:3000,horizon:6}
};
function updateTaskInputTemplate(){
 const capability=$('#capability-select')?.value;
 const input=$('#task-form textarea[name="input"]');
 if(!input)return;
 input.value=JSON.stringify(taskInputTemplates[capability]||{},null,2);
}
$('#capability-select')?.addEventListener('change',updateTaskInputTemplate);
$('#task-form').addEventListener('submit',async event=>{
 event.preventDefault();if(!activeWorkspace)return;
 const form=new FormData(event.currentTarget),capability=form.get('capability');let input;
 try{input=JSON.parse(form.get('input'))}catch(error){$('#task-list').textContent='Task input must be valid JSON.';return}
 if(!input||typeof input!=='object'||Array.isArray(input)){
  $('#task-list').textContent='Task input must be a JSON object.';return;
 }
 if(capability==='model.train_research'){
  if(!['binance','mexc'].includes(input.venue)||typeof input.symbol!=='string'||!input.symbol.includes('/')||
     !['1h','4h','1d'].includes(input.timeframe)||!Number.isInteger(input.limit)||input.limit<1500||input.limit>5000||
     !Number.isInteger(input.horizon)||input.horizon<2||input.horizon>24){
   $('#task-list').textContent='Model training requires venue binance/mexc, a spot pair, timeframe 1h/4h/1d, limit 1500–5000 candles, and horizon 2–24. Use the suggested template.';
   return;
  }
 }
 if(capability==='strategy.walk_forward'&&(!Number.isInteger(input.limit)||input.limit<300||input.limit>500)){
  $('#task-list').textContent='Strategy walk-forward requires 300–500 candles. Use the suggested template.';return;
 }
 try{
  await api('/v1/workspaces/'+encodeURIComponent(activeWorkspace)+'/agents/tasks',{method:'POST',body:JSON.stringify({
   target_capability:capability,input
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

// Live public Binance spot chart. No API keys or trading permissions are used.
(()=>{
 const canvas=$('#market-chart');
 if(!canvas)return;
 const ctx=canvas.getContext('2d');
 const symbolSelect=$('#chart-symbol'),timeframeSelect=$('#chart-timeframe');
 const status=$('#chart-feed-status'),priceLabel=$('#chart-last-price'),changeLabel=$('#chart-price-change');
 let candles=[],socket=null,reconnectTimer=null,pollTimer=null,activeSymbol='',activeTimeframe='',destroyed=false;
 const fmtPrice=value=>value>=1000?value.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}):value>=1?value.toFixed(4):value.toPrecision(5);
 function setStatus(message,connected=false){status.textContent=message;status.classList.toggle('connected',connected)}
 function updatePrice(){
  if(!candles.length)return;
  const last=candles[candles.length-1],first=candles[Math.max(0,candles.length-2)];
  priceLabel.textContent=fmtPrice(last.c);
  const pct=first.o?((last.c-first.o)/first.o)*100:0;
  changeLabel.textContent=(pct>=0?'+':'')+pct.toFixed(2)+'% · '+activeSymbol.replace('USDT',' / USDT');
  changeLabel.classList.toggle('positive',pct>=0);changeLabel.classList.toggle('negative',pct<0);
 }
 function draw(){
  if(!canvas||!ctx)return;
  const rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
  const width=Math.max(300,rect.width),height=Math.max(220,rect.height);
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,width,height);
  if(!candles.length){ctx.fillStyle='#9eacc2';ctx.font='13px system-ui';ctx.fillText('Loading market candles…',18,32);return}
  const left=10,right=72,top=14,volumeHeight=48,bottom=24,plotBottom=height-volumeHeight-bottom,plotHeight=plotBottom-top;
  const visible=candles.slice(-100),high=Math.max(...visible.map(x=>x.h)),low=Math.min(...visible.map(x=>x.l));
  const spread=Math.max(high-low,high*0.0005),max=high+spread*.06,min=low-spread*.06;
  const y=value=>top+(max-value)/(max-min)*plotHeight;
  ctx.font='10px system-ui';ctx.textAlign='right';ctx.textBaseline='middle';
  for(let i=0;i<=4;i++){
   const value=max-(max-min)*i/4,yy=top+plotHeight*i/4;
   ctx.strokeStyle='#263750';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(width-right,yy);ctx.stroke();
   ctx.fillStyle='#9eacc2';ctx.fillText(fmtPrice(value),width-7,yy);
  }
  const maxVol=Math.max(1,...visible.map(x=>x.v)),slot=(width-left-right)/visible.length,body=Math.max(2,Math.min(11,slot*.62));
  visible.forEach((c,i)=>{
   const x=left+i*slot+slot/2,up=c.c>=c.o,color=up?'#72d6bd':'#f07883';
   ctx.strokeStyle=color;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(x,y(c.h));ctx.lineTo(x,y(c.l));ctx.stroke();
   ctx.fillStyle=color;const topBody=Math.min(y(c.o),y(c.c)),bodyHeight=Math.max(1.5,Math.abs(y(c.o)-y(c.c)));
   ctx.fillRect(x-body/2,topBody,body,bodyHeight);
   const vh=(c.v/maxVol)*(volumeHeight-5);ctx.globalAlpha=.55;ctx.fillRect(x-body/2,height-bottom-vh,body,vh);ctx.globalAlpha=1;
  });
  const last=visible[visible.length-1],lastY=y(last.c);
  ctx.setLineDash([4,4]);ctx.strokeStyle='#86a9c7';ctx.beginPath();ctx.moveTo(left,lastY);ctx.lineTo(width-right,lastY);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#86a9c7';ctx.textAlign='left';ctx.fillText(new Date(last.t).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),left,height-8);
  ctx.textAlign='right';ctx.fillText('VOL',width-right,height-8);
 }
 function closeFeed(){
  if(reconnectTimer)clearTimeout(reconnectTimer);if(pollTimer)clearInterval(pollTimer);
  reconnectTimer=null;pollTimer=null;
  if(socket){socket.onclose=null;socket.close();socket=null}
 }
 async function loadHistory(){
  const symbol=symbolSelect.value,timeframe=timeframeSelect.value;
  activeSymbol=symbol;activeTimeframe=timeframe;setStatus('Loading history…');
  const url='https://api.binance.com/api/v3/klines?symbol='+encodeURIComponent(symbol)+'&interval='+encodeURIComponent(timeframe)+'&limit=180';
  const response=await fetch(url,{cache:'no-store'});
  if(!response.ok)throw new Error('Binance history request failed ('+response.status+')');
  const rows=await response.json();
  candles=rows.map(row=>({t:Number(row[0]),o:Number(row[1]),h:Number(row[2]),l:Number(row[3]),c:Number(row[4]),v:Number(row[5])}));
  updatePrice();draw();
 }
 function connectFeed(){
  closeFeed();if(destroyed)return;
  const stream=activeSymbol.toLowerCase()+'@kline_'+activeTimeframe;
  setStatus('Connecting live stream…');
  socket=new WebSocket('wss://stream.binance.com:9443/ws/'+stream);
  socket.onopen=()=>setStatus('Live · Binance WebSocket',true);
  socket.onmessage=event=>{
   try{
    const msg=JSON.parse(event.data),k=msg.k;if(!k)return;
    const candle={t:Number(k.t),o:Number(k.o),h:Number(k.h),l:Number(k.l),c:Number(k.c),v:Number(k.v)};
    const last=candles[candles.length-1];
    if(last&&last.t===candle.t)candles[candles.length-1]=candle;
    else if(!last||candle.t>last.t)candles.push(candle);
    if(candles.length>240)candles=candles.slice(-240);
    updatePrice();draw();
   }catch(error){setStatus('Live data message error')}
  };
  socket.onerror=()=>setStatus('Live stream interrupted · retrying');
  socket.onclose=()=>{
   if(destroyed)return;
   setStatus('Reconnecting live stream…');
   reconnectTimer=setTimeout(connectFeed,3500);
  };
  pollTimer=setInterval(async()=>{
   if(socket&&socket.readyState===WebSocket.OPEN)return;
   try{await loadHistory();setStatus('Polling public data · live stream unavailable')}
   catch(error){setStatus('Feed unavailable: '+error.message)}
  },15000);
 }
 async function start(){
  closeFeed();
  try{await loadHistory();connectFeed()}
  catch(error){setStatus('Could not load chart: '+error.message);priceLabel.textContent='—'}
 }
 symbolSelect.addEventListener('change',start);timeframeSelect.addEventListener('change',start);
 $('#chart-refresh').addEventListener('click',start);
 if('ResizeObserver'in window)new ResizeObserver(draw).observe(canvas.parentElement);
 else window.addEventListener('resize',draw);
 document.addEventListener('visibilitychange',()=>{if(document.hidden){closeFeed()}else if(!destroyed)connectFeed()});
 window.addEventListener('beforeunload',()=>{destroyed=true;closeFeed()},{once:true});
 start();
})();

loadStatus();restoreSession();
