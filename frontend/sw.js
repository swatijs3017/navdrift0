const CACHE = 'navdrift-v2';
const API_HOST = 'navdrift0-api.onrender.com';

// Only cache these static assets, never HTML pages.
// HTML pages must always go to the network so redirects work correctly.
const STATIC = [
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(STATIC)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  const url = new URL(req.url);

  // 1. Navigation requests (loading a page): always go to network.
  //    This is the critical fix — Safari rejects redirect responses from SW
  //    for navigate requests, so we must never intercept them.
  if (req.mode === 'navigate') {
    e.respondWith(fetch(req));
    return;
  }

  // 2. API calls: network-first, 3 s timeout, offline fallback.
  if (url.hostname === API_HOST) {
    e.respondWith(
      fetch(req, { signal: AbortSignal.timeout(3000) }).catch(() =>
        new Response(
          JSON.stringify({ error: 'offline', demo_mode: true }),
          { headers: { 'Content-Type': 'application/json' } }
        )
      )
    );
    return;
  }

  // 3. Static assets (Leaflet, fonts): cache-first.
  e.respondWith(
    caches.match(req).then(cached => cached || fetch(req))
  );
});
