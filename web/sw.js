const CACHE='eti-shell-v14';
const ASSETS=['/','/style.css','/app.js','/manifest.webmanifest','/eti-icon.svg'];
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(ASSETS)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key!==CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',event=>{
 if(event.request.method!=='GET'||new URL(event.request.url).origin!==self.location.origin)return;
 const url=new URL(event.request.url);
 // Authenticated API responses must never enter the offline shell cache.
 if(url.pathname.startsWith('/v1/')){
  event.respondWith(fetch(new Request(event.request,{cache:'no-store'})));
  return;
 }
 // Cache only the static app shell; all API/status routes remain network-only.
 if(!ASSETS.includes(url.pathname))return;
 event.respondWith(caches.match(event.request).then(cached=>cached||fetch(event.request).then(response=>{
  if(response.ok){const copy=response.clone();caches.open(CACHE).then(cache=>cache.put(event.request,copy))}
  return response;
 }).catch(()=>caches.match('/'))));
});
