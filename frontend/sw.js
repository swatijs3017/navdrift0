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

  // 0. Anything that isn't a plain GET (our Overpass road-graph query is a POST, so is any
  //    future write-style request) is never touched by this service worker at all. Passing a
  //    Request through caches.match() and then re-fetching the SAME Request object can throw
  //    once its body has already been read once — that's exactly the "FetchEvent.respondWith
  //    received an error: TypeError: Load failed" seen when the Overpass fetch got routed
  //    through here by accident. Not calling respondWith() at all lets the browser handle the
  //    request completely natively, which is the only path that is guaranteed safe for a
  //    request with a body.
  if (req.method !== 'GET') {
    return;
  }

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

  // 3. Only the two specific static assets this service worker actually knows how to cache go
  //    through cache-first. Everything else GET (Overpass's own node/way responses if a future
  //    version ever GETs it, any other third-party host, anything not explicitly listed) goes
  //    straight to the network untouched, rather than being run through caches.match() for
  //    requests this worker was never meant to own.
  if (STATIC.includes(req.url)) {
    e.respondWith(
      caches.match(req).then(cached => cached || fetch(req))
    );
  }
});
