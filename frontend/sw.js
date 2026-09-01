const CACHE='navdrift-v1';
const API_HOST='navdrift0-api.onrender.com';
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(['/','index.html','mobile.html']))));
self.addEventListener('fetch',e=>{
  const url=new URL(e.request.url);
  if(url.hostname===API_HOST){
    e.respondWith(fetch(e.request,{signal:AbortSignal.timeout(3000)}).catch(()=>new Response('{"error":"offline","demo_mode":true}',{headers:{'Content-Type':'application/json'}})));
  } else {
    e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request)));
  }
});
