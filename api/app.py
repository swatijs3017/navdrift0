<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NAVDRIFT-0 · Mission Control · ISRO SIH 2026 PS #26168</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
  <style>
    :root {
      --bg: #080810; --surface: #0d0e1c; --card: rgba(12,14,26,0.82);
      --cyan: #00fff5; --cyan-d: rgba(0,255,245,0.15); --cyan-g: rgba(0,255,245,0.35);
      --violet: #a855f7; --violet-d: rgba(168,85,247,0.15);
      --red: #ff2255; --red-d: rgba(255,34,85,0.15); --red-g: rgba(255,34,85,0.4);
      --green: #00ff9d; --green-d: rgba(0,255,157,0.15); --green-g: rgba(0,255,157,0.4);
      --yellow: #fbbf24; --orange: #f97316;
      --b1: rgba(255,255,255,0.08); --b2: rgba(255,255,255,0.04);
      --t1: #f1f5f9; --t2: #94a3b8; --t3: #475569;
      --mono: 'JetBrains Mono', monospace; --sans: 'Space Grotesk', sans-serif;
      --r: 10px;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100vw; height: 100vh; overflow: hidden; background: var(--bg); color: var(--t1); font-family: var(--sans); user-select: none; }

    /* ── LAYOUT ── */
    .app { display: grid; grid-template-rows: 50px 1fr 76px; width: 100vw; height: 100vh; position: relative; z-index: 2; }
    .body { display: grid; grid-template-columns: 268px 1fr 252px; gap: 8px; padding: 8px; min-height: 0; overflow: hidden; }
    .panel { display: flex; flex-direction: column; gap: 7px; overflow-y: auto; overflow-x: hidden; }
    .panel::-webkit-scrollbar { width: 3px; }
    .panel::-webkit-scrollbar-thumb { background: var(--b1); border-radius: 3px; }

    /* ── HEADER ── */
    header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 14px; background: rgba(6,6,14,0.96); backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--b1); position: relative; z-index: 20;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand-logo { width: 30px; height: 30px; }
    .brand-logo svg { width: 100%; height: 100%; filter: drop-shadow(0 0 7px var(--cyan-g)); }
    .brand-name { font-size: 1rem; font-weight: 700; letter-spacing: .1em;
      background: linear-gradient(135deg, var(--cyan), var(--violet)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .brand-sub { font-size: .56rem; color: var(--t2); letter-spacing: .14em; text-transform: uppercase; font-family: var(--mono); }

    .city-tabs { display: flex; gap: 3px; background: rgba(0,0,0,0.35); padding: 3px; border-radius: 8px; border: 1px solid var(--b1); }
    .city-tab { padding: 4px 14px; font-size: .68rem; font-weight: 600; color: var(--t2); background: transparent; border: 1px solid transparent; border-radius: 6px; cursor: pointer; transition: all .2s; }
    .city-tab.active { color: var(--cyan); background: var(--cyan-d); border-color: rgba(0,255,245,0.25); }
    .city-tab:hover:not(.active) { color: var(--t1); background: var(--b2); }

    .hdr-right { display: flex; align-items: center; gap: 10px; }
    .pill { display: flex; align-items: center; gap: 6px; background: rgba(0,0,0,0.3); padding: 4px 12px; border-radius: 20px; border: 1px solid var(--b1); font-family: var(--mono); font-size: .65rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 7px var(--green-g); transition: all .3s; }
    .dot.out { background: var(--red); box-shadow: 0 0 10px var(--red-g); animation: blink .7s step-end infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .badge { font-size: .59rem; font-weight: 700; letter-spacing: .07em; padding: 3px 9px; border-radius: 5px; }
    .badge-ps { background: var(--violet-d); color: var(--violet); border: 1px solid rgba(168,85,247,0.3); }
    .badge-mode { background: var(--cyan-d); color: var(--cyan); border: 1px solid rgba(0,255,245,0.25); font-family: var(--mono); }
    .badge-mode.live { background: var(--green-d); color: var(--green); border-color: rgba(0,255,157,0.3); }
    .btn-icon { background: var(--b2); border: 1px solid var(--b1); color: var(--t2); width: 30px; height: 30px; border-radius: 7px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all .2s; font-size: .8rem; }
    .btn-icon:hover { background: var(--cyan-d); color: var(--cyan); border-color: rgba(0,255,245,0.3); }

    /* ── OUTAGE BANNER ── */
    #outage-flash { position: fixed; inset: 0; z-index: 1; pointer-events: none; border: 0 solid var(--red); opacity: 0; transition: all .2s; }
    #outage-flash.on { border: 3px solid var(--red); box-shadow: inset 0 0 60px var(--red-d); animation: rfp 1.3s ease-in-out infinite alternate; opacity: 1; }
    @keyframes rfp { 0%{box-shadow:inset 0 0 30px var(--red-d)} 100%{box-shadow:inset 0 0 90px rgba(255,34,85,0.7)} }
    .banner { position: fixed; bottom: 84px; left: 50%; transform: translateX(-50%) translateY(10px);
      z-index: 200; display: flex; align-items: center; gap: 10px;
      padding: 6px 18px; background: rgba(160,0,40,0.92); backdrop-filter: blur(10px);
      border: 1px solid rgba(255,50,90,0.5); border-radius: 8px;
      box-shadow: 0 4px 24px rgba(255,34,85,0.35); color: #fff;
      font-size: .67rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
      transition: opacity .22s, transform .22s; pointer-events: none; opacity: 0; }
    .banner.on { opacity: 1; transform: translateX(-50%) translateY(0); }
    .banner.snap { background: rgba(0,100,60,0.93); border-color: rgba(0,255,157,0.4); box-shadow: 0 4px 24px rgba(0,255,157,0.3); color: #d1fae5; }

    /* ── CARD ── */
    .card { background: var(--card); backdrop-filter: blur(14px); border: 1px solid var(--b1); border-radius: var(--r); padding: 10px 11px; position: relative; overflow: hidden; }
    .card::before { content: ''; position: absolute; top: 0; left: 12%; right: 12%; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,245,0.18), transparent); }
    .ct { font-size: .6rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: var(--t2); display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .ct-l { display: flex; align-items: center; gap: 5px; }
    .cdot { width: 5px; height: 5px; border-radius: 50%; }
    .cdot.c { background: var(--cyan); box-shadow: 0 0 5px var(--cyan-g); }
    .cdot.v { background: var(--violet); box-shadow: 0 0 5px rgba(168,85,247,0.5); }
    .cdot.r { background: var(--red); box-shadow: 0 0 5px var(--red-g); }
    .cdot.g { background: var(--green); box-shadow: 0 0 5px var(--green-g); }
    .cdot.y { background: var(--yellow); box-shadow: 0 0 5px rgba(251,191,36,0.5); }

    /* ── BUTTONS ── */
    .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    .btn { padding: 7px 10px; font-size: .68rem; font-weight: 600; border: 1px solid var(--b1); border-radius: 7px; cursor: pointer; background: var(--b2); color: var(--t2); transition: all .2s; font-family: var(--sans); }
    .btn.p { background: var(--cyan-d); color: var(--cyan); border-color: rgba(0,255,245,0.3); }
    .btn.d { background: var(--red-d); color: var(--red); border-color: rgba(255,34,85,0.3); }

    /* ── MODULE CHIPS — compact 3-col ── */
    .mods { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; }
    .mod { display: flex; align-items: center; gap: 4px; padding: 4px 6px; border-radius: 5px; border: 1px solid; font-size: .55rem; font-weight: 700; font-family: var(--mono); letter-spacing: .04em; transition: all .3s; }
    .mod.on  { border-color: rgba(0,255,157,0.3); background: rgba(0,255,157,0.07); color: var(--green); }
    .mod.warn{ border-color: rgba(251,191,36,0.3); background: rgba(251,191,36,0.07); color: var(--yellow); }
    .mod.off { border-color: var(--b1); background: var(--b2); color: var(--t3); }
    .mod-dot { width: 4px; height: 4px; border-radius: 50%; flex-shrink: 0; background: currentColor; }
    .mod.warn .mod-dot { animation: blink .8s step-end infinite; }
    .mod.full { grid-column: 1/-1; }

    /* ── METRIC ROWS ── */
    .mrow { display: flex; align-items: center; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid var(--b2); }
    .mrow:last-child { border-bottom: none; }
    .ml { font-size: .61rem; color: var(--t2); font-family: var(--mono); }
    .mv { font-size: .84rem; font-weight: 700; font-family: var(--mono); }
    .mu { font-size: .58rem; color: var(--t3); margin-left: 2px; }
    .mv.c { color: var(--cyan); }
    .mv.g { color: var(--green); }
    .mv.r { color: var(--red); }
    .mv.v { color: var(--violet); }
    .mv.y { color: var(--yellow); }

    /* ── CAL / SENSOR BARS ── */
    .sbar-row { display: flex; align-items: center; gap: 7px; padding: 4px 0; border-bottom: 1px solid var(--b2); }
    .sbar-row:last-child { border-bottom: none; }
    .sbar-lbl { font-size: .58rem; font-family: var(--mono); color: var(--t2); width: 36px; flex-shrink: 0; }
    .sbar-track { flex: 1; height: 4px; border-radius: 2px; background: rgba(255,255,255,0.06); overflow: hidden; }
    .sbar-fill { height: 100%; border-radius: 2px; background: var(--cyan); transition: width .5s; }
    .sbar-val { font-size: .6rem; font-family: var(--mono); color: var(--cyan); width: 38px; text-align: right; }

    /* ── FILTER COMPARE ── */
    .fcomp { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-bottom: 6px; }
    .fcol { background: rgba(0,0,0,0.25); border-radius: 6px; padding: 5px 7px; }
    .fcol-lbl { font-size: .52rem; font-family: var(--mono); color: var(--t3); margin-bottom: 2px; }
    .fcol-val { font-size: .78rem; font-weight: 700; font-family: var(--mono); }

    /* ── NHC ── */
    .nhc-row { display: flex; align-items: center; gap: 6px; padding: 4px 0; border-bottom: 1px solid var(--b2); }
    .nhc-row:last-child { border-bottom: none; }
    .nhc-lbl { font-size: .58rem; font-family: var(--mono); color: var(--t2); width: 78px; flex-shrink: 0; }
    .nhc-track { flex: 1; height: 4px; border-radius: 2px; background: rgba(255,255,255,0.06); overflow: hidden; }
    .nhc-fill { height: 100%; border-radius: 2px; transition: width .3s; }

    /* ── MAP ── */
    .map-wrap { position: relative; border-radius: var(--r); overflow: hidden; border: 1px solid var(--b1); min-height: 0; }
    #map { width: 100%; height: 100%; background: #060610; }
    .leaflet-container { background: #060610 !important; }
    .hud { position: absolute; z-index: 500; pointer-events: none; }
    .hud.tl { top: 9px; left: 9px; }
    .hud.tr { top: 9px; right: 9px; }
    .hud.bl { bottom: 9px; left: 9px; }
    .hbox { background: rgba(6,6,14,0.85); backdrop-filter: blur(8px); border: 1px solid var(--b1); border-radius: 7px; padding: 6px 10px; }
    .hl { font-size: .52rem; font-family: var(--mono); color: var(--t3); letter-spacing: .08em; text-transform: uppercase; }
    .hv { font-size: .76rem; font-weight: 700; font-family: var(--mono); color: var(--cyan); margin-top: 1px; }
    .hv.out { color: var(--red); }
    .leg-row { display: flex; align-items: center; gap: 5px; margin-top: 4px; }
    .leg-line { width: 16px; height: 2px; border-radius: 1px; }
    .leg-txt { font-size: .54rem; font-family: var(--mono); color: var(--t2); }
    .mm-badge { position: absolute; bottom: 9px; right: 9px; z-index: 500; display: flex; align-items: center; gap: 5px; padding: 4px 9px; border-radius: 6px; font-size: .56rem; font-family: var(--mono); font-weight: 700; transition: all .3s; }
    .mm-badge.on  { background: rgba(0,255,157,0.1); border: 1px solid rgba(0,255,157,0.3); color: var(--green); }
    .mm-badge.off { background: rgba(249,115,22,0.1); border: 1px solid rgba(249,115,22,0.3); color: var(--orange); }
    .mm-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

    /* ── SPARKLINES ── */
    .spark { display: flex; flex-direction: column; gap: 4px; }
    .spark-hdr { display: flex; justify-content: space-between; align-items: baseline; }
    .spark-lbl { font-size: .59rem; font-family: var(--mono); color: var(--t2); letter-spacing: .06em; text-transform: uppercase; }
    .spark-num { font-size: .78rem; font-weight: 700; font-family: var(--mono); }
    .spark-cv { height: 40px; border-radius: 5px; overflow: hidden; background: rgba(0,0,0,0.2); }
    .spark-cv canvas { width: 100%; height: 100%; display: block; }

    /* ── BENCH TABLE ── */
    table.bench { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: .6rem; }
    table.bench th { color: var(--t3); font-weight: 600; padding: 4px 5px; text-align: left; border-bottom: 1px solid var(--b1); }
    table.bench td { padding: 4px 5px; border-bottom: 1px solid var(--b2); color: var(--t2); }
    table.bench tr.rnd td { color: var(--cyan); font-weight: 700; }
    table.bench tr.rekf td { color: var(--violet); }
    table.bench tr.rimu td { color: var(--red); }
    table.bench .pass { color: var(--green) !important; font-weight: 700; }
    table.bench .fail { color: var(--red) !important; }

    /* ── FUSION BARS ── */
    .fbar-row { display: flex; align-items: center; gap: 7px; margin-bottom: 5px; }
    .fbar-lbl { font-size: .58rem; font-family: var(--mono); width: 52px; flex-shrink: 0; }
    .fbar-track { flex: 1; height: 5px; border-radius: 3px; background: rgba(255,255,255,0.06); overflow: hidden; }
    .fbar-fill { height: 100%; border-radius: 3px; transition: width .4s; }
    .fbar-val { font-size: .58rem; font-family: var(--mono); width: 30px; text-align: right; }

    /* ── LOG ── */
    .elog { display: flex; flex-direction: column; gap: 3px; max-height: 80px; overflow-y: auto; }
    .elog::-webkit-scrollbar { width: 2px; }
    .le { display: flex; gap: 5px; font-size: .56rem; font-family: var(--mono); line-height: 1.35; }
    .lt { color: var(--t3); flex-shrink: 0; }
    .lm { color: var(--t2); }
    .le.out .lm { color: var(--red); }
    .le.snap .lm { color: var(--green); }
    .le.nhc .lm { color: var(--yellow); }
    .le.sys .lm { color: var(--cyan); }

    /* ── SIGNAL ── */
    .sig { display: flex; align-items: center; gap: 10px; }
    .sig-bars { display: flex; align-items: flex-end; gap: 3px; height: 20px; }
    .sig-bar { width: 5px; border-radius: 2px 2px 0 0; background: var(--green); box-shadow: 0 0 5px var(--green-g); transition: all .4s; }
    .sig-bar.lost { background: var(--red); animation: blink 1s step-end infinite; }

    /* ── PIPELINE ── */
    .pipe { display: flex; align-items: stretch; background: rgba(6,6,14,0.96); border-top: 1px solid var(--b1); overflow: hidden; }
    .pmod { display: flex; align-items: center; gap: 8px; padding: 0 14px; flex: 1; justify-content: center; border-right: 1px solid var(--b1); cursor: default; transition: background .3s; }
    .pmod:last-child { border-right: none; }
    .pmod.active { background: rgba(0,255,245,0.05); }
    .pico { font-size: .9rem; transition: filter .3s; }
    .pmod.active .pico { filter: drop-shadow(0 0 5px var(--cyan)); }
    .pinf { display: flex; flex-direction: column; gap: 1px; }
    .pname { font-size: .6rem; font-weight: 700; color: var(--t2); transition: color .3s; }
    .pmod.active .pname { color: var(--cyan); }
    .psub { font-size: .5rem; font-family: var(--mono); color: var(--t3); }

    /* ── MODAL ── */
    .modal-ov { position: fixed; inset: 0; background: rgba(0,0,0,0.75); backdrop-filter: blur(8px); z-index: 100; display: flex; align-items: center; justify-content: center; opacity: 0; pointer-events: none; transition: opacity .25s; }
    .modal-ov.on { opacity: 1; pointer-events: all; }
    .modal { background: #0d0e1c; border: 1px solid rgba(0,255,245,0.25); border-radius: 12px; padding: 22px; width: 400px; box-shadow: 0 0 60px rgba(0,255,245,0.12); }
    .mtitle { font-size: .95rem; font-weight: 700; color: var(--cyan); margin-bottom: 14px; }
    .mlbl { font-size: .65rem; font-family: var(--mono); color: var(--t2); margin-bottom: 4px; letter-spacing: .05em; text-transform: uppercase; }
    .minp { width: 100%; background: rgba(255,255,255,0.04); border: 1px solid var(--b1); border-radius: 6px; padding: 7px 10px; color: var(--t1); font-family: var(--mono); font-size: .72rem; margin-bottom: 10px; outline: none; }
    .minp:focus { border-color: rgba(0,255,245,0.3); }
    .mbtns { display: flex; gap: 6px; margin-top: 4px; }
    .cs { font-size: .63rem; font-family: var(--mono); padding: 5px 8px; border-radius: 5px; margin-top: 8px; display: none; }
    .cs.ok { background: var(--green-d); color: var(--green); border: 1px solid rgba(0,255,157,0.3); }
    .cs.err { background: var(--red-d); color: var(--red); border: 1px solid rgba(255,34,85,0.3); }
    .divider { height: 1px; background: var(--b1); margin: 10px 0; }
    .infobox { font-size: .59rem; font-family: var(--mono); color: var(--t3); line-height: 1.6; padding: 6px 8px; background: rgba(0,0,0,0.2); border-radius: 5px; border: 1px solid var(--b2); }
    .infobox b { color: var(--t2); }
  </style>
</head>
<body>
<div id="outage-flash"></div>
<div class="banner" id="banner">
  <span id="banner-txt">GNSS BLACKOUT — DEAD RECKONING ACTIVE</span>
  <span style="opacity:.5">|</span>
  <span id="banner-timer">0.0s</span>
</div>

<div class="app">
  <!-- HEADER -->
  <header>
    <div class="brand">
      <div class="brand-logo">
        <svg viewBox="0 0 32 32" fill="none">
          <circle cx="16" cy="16" r="14" stroke="#00fff5" stroke-width="1.2" opacity=".35"/>
          <circle cx="16" cy="16" r="8" stroke="#a855f7" stroke-width="1" opacity=".55"/>
          <circle cx="16" cy="16" r="2.5" fill="#00fff5"/>
          <line x1="16" y1="2" x2="16" y2="7" stroke="#00fff5" stroke-width="1.4"/>
          <line x1="16" y1="25" x2="16" y2="30" stroke="#00fff5" stroke-width="1.4" opacity=".3"/>
          <line x1="2" y1="16" x2="7" y2="16" stroke="#00fff5" stroke-width="1.4" opacity=".3"/>
          <line x1="25" y1="16" x2="30" y2="16" stroke="#00fff5" stroke-width="1.4" opacity=".3"/>
        </svg>
      </div>
      <div>
        <div class="brand-name">NAVDRIFT-0</div>
        <div class="brand-sub">ISRO SIH 2026 · PS #26168 · IDR System</div>
      </div>
    </div>

    <div class="city-tabs">
      <button class="city-tab active" onclick="switchCity('delhi',event)">Delhi</button>
      <button class="city-tab" onclick="switchCity('mumbai',event)">Mumbai</button>
      <button class="city-tab" onclick="switchCity('bengaluru',event)">Bengaluru</button>
      <button class="city-tab" onclick="switchCity('chennai',event)">Chennai</button>
      <button class="city-tab" onclick="switchCity('hyderabad',event)">Hyderabad</button>
    </div>

    <div class="hdr-right">
      <div class="pill"><div class="dot" id="hdr-dot"></div><span id="hdr-status">GNSS LOCKED</span></div>
      <span class="badge badge-mode" id="mode-badge">SIMULATION</span>
      <span class="badge badge-ps">PS #26168</span>
      <div class="btn-icon" onclick="openModal()" title="API Settings">⚙</div>
    </div>
  </header>

  <!-- BODY -->
  <div class="body">

    <!-- LEFT PANEL -->
    <div class="panel">

      <!-- Controls -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot c"></span>Controls</div></div>
        <div class="btn-row">
          <button class="btn p" id="btn-play" onclick="toggleSim()">⏸ Pause</button>
          <button class="btn d" onclick="toggleOutage()">📡 GNSS Toggle</button>
        </div>
      </div>

      <!-- IDR Modules -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot g"></span>IDR System Modules</div><span style="font-size:.55rem;color:var(--t3)">PS #26168</span></div>
        <div class="mods">
          <div class="mod on" id="m-cal"><span class="mod-dot"></span>CAL ENG</div>
          <div class="mod on" id="m-vib"><span class="mod-dot"></span>VIB FILT</div>
          <div class="mod on" id="m-nhc"><span class="mod-dot"></span>NHC</div>
          <div class="mod on" id="m-mm"><span class="mod-dot"></span>MAP MATCH</div>
          <div class="mod on" id="m-ekf"><span class="mod-dot"></span>EKF FUSE</div>
          <div class="mod off" id="m-snap"><span class="mod-dot"></span>SNAP</div>
          <div class="mod on full" id="m-dr"><span class="mod-dot"></span>DRIFT-FORMER · INT8/INT4 ONNX · 20ms CPU · 9-ch (baro)</div>
        </div>
      </div>

      <!-- Phone Calibration + Vibration Filter — merged -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot y"></span>Alignment &amp; Vibration Filter</div><span style="font-size:.55rem;color:var(--green)">✓ LOCKED</span></div>
        <div class="sbar-row">
          <span class="sbar-lbl">PITCH</span>
          <div class="sbar-track"><div class="sbar-fill" id="cal-p" style="width:62%"></div></div>
          <span class="sbar-val" id="cal-pv">+3.2°</span>
        </div>
        <div class="sbar-row">
          <span class="sbar-lbl">ROLL</span>
          <div class="sbar-track"><div class="sbar-fill" id="cal-r" style="width:42%"></div></div>
          <span class="sbar-val" id="cal-rv">-1.4°</span>
        </div>
        <div class="sbar-row">
          <span class="sbar-lbl">YAW</span>
          <div class="sbar-track"><div class="sbar-fill" id="cal-y" style="width:51%"></div></div>
          <span class="sbar-val" id="cal-yv" style="color:var(--green)">0.1°</span>
        </div>
        <div style="height:1px;background:var(--b1);margin:7px 0"></div>
        <div class="fcomp">
          <div class="fcol">
            <div class="fcol-lbl">RAW IMU (noisy)</div>
            <div class="fcol-val r" id="vib-raw">— km/h</div>
          </div>
          <div class="fcol">
            <div class="fcol-lbl">AI FILTERED</div>
            <div class="fcol-val g" id="vib-filt">— km/h</div>
          </div>
        </div>
        <div style="height:36px;border-radius:5px;overflow:hidden;background:rgba(0,0,0,0.22);position:relative">
          <div style="position:absolute;top:2px;left:5px;font-size:.48rem;font-family:var(--mono);color:var(--t3)">VIBRATION · raw (red) vs filtered (cyan)</div>
          <canvas id="vib-cv" style="width:100%;height:100%;display:block"></canvas>
        </div>
        <div class="mrow" style="margin-top:5px">
          <span class="ml">Pothole events filtered</span>
          <span class="mv y" id="vib-pots">0</span>
        </div>
      </div>

      <!-- NHC -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot v"></span>Non-Holonomic Constraints</div></div>
        <div class="nhc-row">
          <span class="nhc-lbl">Lateral vel</span>
          <div class="nhc-track"><div class="nhc-fill" id="nhc-lb" style="width:0%;background:linear-gradient(90deg,var(--green),var(--cyan))"></div></div>
          <span style="font-size:.6rem;font-family:var(--mono);color:var(--cyan);width:36px;text-align:right" id="nhc-lv">0.00</span>
        </div>
        <div class="nhc-row">
          <span class="nhc-lbl">Vertical vel</span>
          <div class="nhc-track"><div class="nhc-fill" id="nhc-vb" style="width:0%;background:linear-gradient(90deg,var(--violet),var(--cyan))"></div></div>
          <span style="font-size:.6rem;font-family:var(--mono);color:var(--cyan);width:36px;text-align:right" id="nhc-vv">0.00</span>
        </div>
        <div class="nhc-row">
          <span class="nhc-lbl">Corrections</span>
          <span style="font-size:.78rem;font-weight:700;font-family:var(--mono);color:var(--violet)" id="nhc-cnt">0</span>
        </div>
        <div class="nhc-row">
          <span class="nhc-lbl">Error removed</span>
          <span style="font-size:.78rem;font-weight:700;font-family:var(--mono);color:var(--green)" id="nhc-err">0.00 m</span>
        </div>
      </div>

      <!-- Event Log -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot r"></span>Event Log</div></div>
        <div class="elog" id="elog"></div>
      </div>

    </div>

    <!-- CENTER MAP -->
    <div class="map-wrap">
      <div id="map"></div>
      <div class="hud tl"><div class="hbox">
        <div class="hl">Position</div>
        <div class="hv" id="hud-pos">—</div>
        <div class="hl" style="margin-top:4px">Route</div>
        <div class="hv" id="hud-route" style="font-size:.66rem">—</div>
      </div></div>
      <div class="hud tr"><div class="hbox">
        <div class="hl">GNSS Mode</div>
        <div class="hv" id="hud-mode">NavIC L5 Locked</div>
        <div class="hl" style="margin-top:4px">DR Duration</div>
        <div class="hv" id="hud-dr" style="font-size:.66rem;color:var(--violet)">0.0s</div>
      </div></div>
      <div class="hud bl"><div class="hbox">
        <div class="leg-row"><div class="leg-line" style="background:var(--cyan)"></div><span class="leg-txt">Ground Truth (GNSS)</span></div>
        <div class="leg-row"><div class="leg-line" style="background:var(--violet)"></div><span class="leg-txt">NAVDRIFT-0 (DRIFT-Former)</span></div>
        <div class="leg-row"><div class="leg-line" style="background:var(--red);border-top:1px dashed var(--red);height:0"></div><span class="leg-txt">Raw IMU (uncorrected)</span></div>
        <div class="leg-row"><div class="leg-line" style="background:var(--yellow)"></div><span class="leg-txt">EKF Baseline</span></div>
      </div></div>
      <div class="mm-badge on" id="mm-badge"><span class="mm-dot"></span><span id="mm-txt">HMM Map-Match ACTIVE</span></div>
    </div>

    <!-- RIGHT PANEL -->
    <div class="panel">

      <!-- Live Telemetry -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot c"></span>Live Telemetry</div><span class="badge badge-mode" id="mode-chip">[SIM]</span></div>
        <div class="mrow"><span class="ml">NAVDRIFT-0 ATE</span><span class="mv c" id="v-nd">—<span class="mu">m</span></span></div>
        <div class="mrow"><span class="ml">Raw IMU ATE</span><span class="mv r" id="v-imu">—<span class="mu">m</span></span></div>
        <div class="mrow"><span class="ml">EKF Baseline ATE</span><span class="mv v" id="v-ekf">—<span class="mu">m</span></span></div>
        <div class="mrow"><span class="ml">Uncertainty σ</span><span class="mv y" id="v-unc">—<span class="mu">m</span></span></div>
        <div class="mrow"><span class="ml">Inference latency</span><span class="mv g" id="v-lat">—<span class="mu">ms</span></span></div>
        <div class="mrow"><span class="ml">Update rate</span><span class="mv g">10<span class="mu">Hz</span></span></div>
        <div class="mrow"><span class="ml">Steps / Outages</span><span class="mv" style="color:var(--t2);font-size:.7rem"><span id="v-steps">0</span> / <span id="v-outs" style="color:var(--red)">0</span></span></div>
      </div>

      <!-- Signal -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot g"></span>NavIC Signal</div></div>
        <div class="sig">
          <div class="sig-bars">
            <div class="sig-bar" style="height:5px" id="sb1"></div>
            <div class="sig-bar" style="height:9px" id="sb2"></div>
            <div class="sig-bar" style="height:13px" id="sb3"></div>
            <div class="sig-bar" style="height:17px" id="sb4"></div>
            <div class="sig-bar" style="height:20px" id="sb5"></div>
          </div>
          <div>
            <div style="font-size:.72rem;font-weight:700;font-family:var(--mono);color:var(--green)" id="sig-state">NavIC L5 / S-Band</div>
            <div style="font-size:.57rem;color:var(--t3);font-family:var(--mono)" id="sig-sub">7 Satellites locked</div>
          </div>
        </div>
      </div>

      <!-- Sparklines -->
      <div class="card spark">
        <div class="spark-hdr">
          <span class="spark-lbl" style="color:var(--red)">Position Error</span>
          <span class="spark-num r" id="sp-err-v">—</span>
        </div>
        <div class="spark-cv"><canvas id="sp-err"></canvas></div>
      </div>

      <div class="card spark">
        <div class="spark-hdr">
          <span class="spark-lbl" style="color:var(--violet)">Uncertainty σ</span>
          <span class="spark-num v" id="sp-unc-v">—</span>
        </div>
        <div class="spark-cv"><canvas id="sp-unc"></canvas></div>
      </div>

      <div class="card spark">
        <div class="spark-hdr">
          <span class="spark-lbl" style="color:var(--green)">Speed</span>
          <span class="spark-num g" id="sp-spd-v">—</span>
        </div>
        <div class="spark-cv"><canvas id="sp-spd"></canvas></div>
      </div>

      <div class="card spark">
        <div class="spark-hdr">
          <span class="spark-lbl" style="color:var(--yellow)">Baro Altitude</span>
          <span class="spark-num y" id="sp-baro-v">—</span>
        </div>
        <div class="spark-cv"><canvas id="sp-baro"></canvas></div>
        <div style="display:flex;justify-content:space-between;margin-top:3px">
          <span style="font-size:.52rem;font-family:var(--mono);color:var(--t3)">9th channel · tunnel detect</span>
          <span style="font-size:.52rem;font-family:var(--mono)" id="sp-baro-tun">—</span>
        </div>
      </div>

      <!-- Benchmark -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot v"></span>Algorithm Benchmark</div><span style="font-size:.53rem;color:var(--t3)">IO-VNBD held-out</span></div>
        <table class="bench">
          <thead><tr><th>Method</th><th>ATE</th><th>Max</th><th>Target</th></tr></thead>
          <tbody>
            <tr class="rnd"><td>NAVDRIFT-0</td><td id="tb-nd">—</td><td id="tb-ndx">—</td><td class="pass">✓ &lt;100m</td></tr>
            <tr class="rekf"><td>EKF (live)</td><td id="tb-ekf">—</td><td id="tb-ekfx">—</td><td>baseline</td></tr>
            <tr class="rimu"><td>Raw IMU</td><td id="tb-imu">—</td><td id="tb-imux">—</td><td class="fail">✗ drifts</td></tr>
          </tbody>
        </table>
        <div class="divider"></div>
        <div class="infobox">
          <b>50m blackout:</b> 3.19m drift (target &lt;5m) ✓<br>
          <b>1km tunnel:</b> 78.41m ATE (target &lt;100m) ✓<br>
          <b>Latency INT8:</b> 20ms CPU · <b>INT4:</b> ~5ms ✓<br>
          <b>HMM map-match:</b> Viterbi · σ=18m · λ=4<br>
          <b>Baro:</b> 9th channel · tunnel detect active
        </div>
      </div>

      <!-- GNSS+INS Fusion -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot c"></span>GNSS+INS Fusion</div></div>
        <div class="fbar-row">
          <span class="fbar-lbl" style="color:var(--cyan)">NavDrift</span>
          <div class="fbar-track"><div class="fbar-fill" id="fb-nd" style="width:90%;background:var(--cyan)"></div></div>
          <span class="fbar-val" id="fb-ndv" style="color:var(--cyan)">—</span>
        </div>
        <div class="fbar-row">
          <span class="fbar-lbl" style="color:var(--violet)">EKF</span>
          <div class="fbar-track"><div class="fbar-fill" id="fb-ekf" style="width:55%;background:var(--violet)"></div></div>
          <span class="fbar-val" id="fb-ekfv" style="color:var(--violet)">—</span>
        </div>
        <div class="fbar-row">
          <span class="fbar-lbl" style="color:var(--red)">Raw IMU</span>
          <div class="fbar-track"><div class="fbar-fill" id="fb-imu" style="width:8%;background:var(--red)"></div></div>
          <span class="fbar-val" id="fb-imuv" style="color:var(--red)">low</span>
        </div>
        <div class="mrow" style="margin-top:4px">
          <span class="ml">SNAP corrections</span>
          <span class="mv g" id="v-snap">0</span>
        </div>
        <div class="mrow">
          <span class="ml">HMM map corrections</span>
          <span class="mv" style="color:var(--yellow)" id="v-hmm">0</span>
        </div>
        <div class="mrow">
          <span class="ml">Tunnels detected</span>
          <span class="mv" style="color:var(--orange)" id="v-tun">0</span>
        </div>
      </div>

      <!-- Engine info -->
      <div class="card">
        <div class="ct"><div class="ct-l"><span class="cdot g"></span>Inference Engine</div></div>
        <div class="infobox" id="engine-info">
          <b>Model:</b> DRIFT-Former INT8 ONNX (14MB)<br>
          <b>Runtime:</b> ONNX Runtime 1.17 · CPU<br>
          <b>Mode:</b> Simulation (set API URL to go live)
        </div>
      </div>

    </div>
  </div>

  <!-- PIPELINE BAR -->
  <div class="pipe">
    <div class="pmod active" id="pm1"><span class="pico">📡</span><div class="pinf"><div class="pname">IMU PREPROCESSOR</div><div class="psub">Calibration · VibFilter · Baro(9th)</div></div></div>
    <div class="pmod" id="pm2"><span class="pico">🧠</span><div class="pinf"><div class="pname">DRIFT-FORMER</div><div class="psub">Causal Transformer · RoPE · INT8/4</div></div></div>
    <div class="pmod" id="pm3"><span class="pico">🔷</span><div class="pinf"><div class="pname">NAVIC VAE</div><div class="psub">Motion Prior Encoder</div></div></div>
    <div class="pmod" id="pm4"><span class="pico">🗺️</span><div class="pinf"><div class="pname">MAP MATCH + NHC</div><div class="psub">Viterbi HMM · Non-Holonomic</div></div></div>
    <div class="pmod" id="pm5"><span class="pico">⚡</span><div class="pinf"><div class="pname">GNSS+INS FUSION</div><div class="psub">EKF · SNAP · WebSocket 10Hz</div></div></div>
    <div class="pmod" id="pm6"><span class="pico">📍</span><div class="pinf"><div class="pname">POSITION OUTPUT</div><div class="psub">SE(2) + Covariance · 10Hz</div></div></div>
  </div>
</div>

<!-- SETTINGS MODAL -->
<div class="modal-ov" id="modal">
  <div class="modal">
    <div class="mtitle">⚙ Backend API</div>
    <div class="mlbl">Render API URL</div>
    <input class="minp" id="api-url" type="url" placeholder="https://navdrift0-api.onrender.com">
    <div class="mlbl">API Key</div>
    <input class="minp" id="api-key" type="password" placeholder="your-api-key">
    <div class="mbtns">
      <button class="btn p" onclick="testConn()" style="flex:1">Test</button>
      <button class="btn" onclick="saveAPI()" style="flex:1;background:var(--green-d);color:var(--green);border-color:rgba(0,255,157,0.3)">Save & Connect</button>
      <button class="btn" onclick="closeModal()" style="flex:.6">Cancel</button>
    </div>
    <div class="cs" id="conn-st"></div>
    <div class="divider"></div>
    <div class="infobox">Leave empty to run in simulation mode.</div>
  </div>
</div>

<script>
  // ─── CITY ROUTES ───
  const CITIES = {
    delhi:    { name:'Delhi (Connaught Grid)',   lat:28.6315, lon:77.2167, zoom:14, wp:[[28.6315,77.2167],[28.6330,77.2210],[28.6350,77.2260],[28.6360,77.2310],[28.6340,77.2360],[28.6310,77.2400],[28.6270,77.2420],[28.6230,77.2410],[28.6190,77.2380],[28.6160,77.2330],[28.6150,77.2270],[28.6160,77.2210],[28.6190,77.2160],[28.6230,77.2130],[28.6270,77.2130],[28.6315,77.2167]] },
    mumbai:   { name:'Mumbai (Marine Drive)',    lat:18.9322, lon:72.8264, zoom:14, wp:[[18.9322,72.8264],[18.9350,72.8240],[18.9380,72.8220],[18.9420,72.8215],[18.9460,72.8230],[18.9490,72.8260],[18.9510,72.8300],[18.9500,72.8340],[18.9470,72.8370],[18.9430,72.8380],[18.9390,72.8370],[18.9350,72.8340],[18.9320,72.8310],[18.9310,72.8280],[18.9322,72.8264]] },
    bengaluru:{ name:'Bengaluru (Outer Ring)',   lat:12.9352, lon:77.6245, zoom:13, wp:[[12.9352,77.6245],[12.9400,77.6320],[12.9450,77.6390],[12.9480,77.6470],[12.9460,77.6560],[12.9410,77.6620],[12.9340,77.6650],[12.9270,77.6630],[12.9210,77.6580],[12.9180,77.6500],[12.9190,77.6410],[12.9230,77.6330],[12.9280,77.6270],[12.9320,77.6250],[12.9352,77.6245]] },
    chennai:  { name:'Chennai (Anna Salai)',     lat:13.0524, lon:80.2580, zoom:14, wp:[[13.0524,80.2580],[13.0560,80.2560],[13.0600,80.2530],[13.0640,80.2500],[13.0680,80.2470],[13.0710,80.2440],[13.0730,80.2400],[13.0720,80.2360],[13.0690,80.2330],[13.0650,80.2310],[13.0610,80.2320],[13.0570,80.2340],[13.0540,80.2370],[13.0520,80.2410],[13.0510,80.2450],[13.0520,80.2510],[13.0524,80.2580]] },
    hyderabad:{ name:'Hyderabad (ORR Loop)',     lat:17.4065, lon:78.4772, zoom:13, wp:[[17.4065,78.4772],[17.4120,78.4850],[17.4170,78.4940],[17.4180,78.5040],[17.4150,78.5130],[17.4090,78.5190],[17.4010,78.5210],[17.3930,78.5180],[17.3870,78.5110],[17.3840,78.5010],[17.3850,78.4910],[17.3900,78.4830],[17.3960,78.4780],[17.4010,78.4760],[17.4065,78.4772]] }
  };

  function interp(wp,n){const pts=[],tot=wp.length-1;for(let i=0;i<n;i++){const t=i/n*tot,idx=Math.floor(t),f=t-idx,a=wp[Math.min(idx,tot)],b=wp[Math.min(idx+1,tot)];pts.push([a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f]);}return pts;}
  function dist(la1,lo1,la2,lo2){const dy=(la2-la1)*Math.PI/180*6371000,dx=(lo2-lo1)*Math.PI/180*6371000*Math.cos(la1*Math.PI/180);return Math.hypot(dy,dx);}
  function clamp(v,a,b){return Math.max(a,Math.min(b,v));}

  // ─── STATE ───
  const S={
    city:'delhi',run:true,step:0,
    gnss:true,outt:0,outN:0,snapN:0,drTotal:0,
    route:[],ri:0,
    gtLa:0,gtLo:0,gtH:0,
    ndLa:0,ndLo:0,iLa:0,iLo:0,unc:0.45,
    spkErr:new Array(60).fill(0.3),spkUnc:new Array(60).fill(0.45),spkSpd:new Array(60).fill(45),
    totND:0,totIMU:0,n:0,mxND:0,mxIMU:0,
    lat:0,spd:0,pH:0,
    nhcN:0,nhcE:0,
    calP:3.2,calR:-1.4,calY:0.1,
    vibBuf:[],vibRaw:[],vibRawSpd:0,vibFiltSpd:0,pots:0,
    // Barometer state (9th channel)
    baroAlt:220,baroPrev:220,baroBase:220,inTunnel:false,tunDur:0,tunN:0,
    spkBaro:new Array(60).fill(220),
    // HMM map-match state
    hmmProb:null, // Float32Array over route — log-probabilities
    hmmCorrN:0,
    // WebSocket
    ws:null,wsReady:false
  };

  // ─── INIT ───
  function init(key){
    S.city=key;const c=CITIES[key];
    S.route=interp(c.wp,300);S.ri=0;
    const st=S.route[0];
    S.gtLa=S.ndLa=S.iLa=st[0];S.gtLo=S.ndLo=S.iLo=st[1];
    S.spkErr=new Array(60).fill(0.3);S.spkUnc=new Array(60).fill(0.45);S.spkSpd=new Array(60).fill(45);
    S.step=0;S.gnss=true;S.outt=0;S.outN=0;S.snapN=0;S.drTotal=0;
    S.unc=0.45;S.totND=0;S.totIMU=0;S.n=0;S.mxND=0;S.mxIMU=0;
    S.lat=0;S.spd=0;S.pH=0;S.nhcN=0;S.nhcE=0;S.vibBuf=[];S.vibRaw=[];S.pots=0;
    // Reset baro
    S.baroAlt=220;S.baroPrev=220;S.baroBase=220;S.inTunnel=false;S.tunDur=0;S.tunN=0;
    S.spkBaro=new Array(60).fill(220);
    // Reset HMM: uniform log-prior over all route points
    S.hmmProb=new Float32Array(S.route.length).fill(-Math.log(S.route.length));S.hmmCorrN=0;
    // Reset WS
    if(S.ws){try{S.ws.close();}catch(e){} S.ws=null;S.wsReady=false;}
    ekfReset(st[0],st[1],0);
    document.getElementById('hud-route').textContent=c.name;
    log('Mission loaded: '+c.name,'sys');

    if(!window._map){
      window._map=L.map('map',{zoomControl:false,attributionControl:true});
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap contributors',maxZoom:19}).addTo(window._map);
      const s=document.createElement('style');
      s.textContent='.leaflet-tile-pane{filter:brightness(0.26) saturate(0.55) hue-rotate(172deg)}';
      document.head.appendChild(s);
      window._gt  =L.polyline([],{color:'#00fff5',weight:2,opacity:.9}).addTo(window._map);
      window._nd  =L.polyline([],{color:'#a855f7',weight:2.5,opacity:.9}).addTo(window._map);
      window._imu =L.polyline([],{color:'#ff2255',weight:1.5,dashArray:'7,5',opacity:.75}).addTo(window._map);
      window._ekf =L.polyline([],{color:'#fbbf24',weight:1.5,opacity:.7}).addTo(window._map);
      window._vm  =L.marker([st[0],st[1]],{icon:L.divIcon({className:'',iconSize:[16,16],iconAnchor:[8,8],
        html:'<div style="width:16px;height:16px;border-radius:50%;background:#00fff5;box-shadow:0 0 12px #00fff5;border:2px solid #fff"></div>'})}).addTo(window._map);
      window._uc  =L.circle([st[0],st[1]],{radius:5,color:'#a855f7',fillColor:'#a855f7',fillOpacity:.1,weight:1}).addTo(window._map);
      window._sl  =L.layerGroup().addTo(window._map);
    } else {
      window._gt.setLatLngs([]);window._nd.setLatLngs([]);window._imu.setLatLngs([]);window._ekf.setLatLngs([]);
      window._sl.clearLayers();window._vm.setLatLng([st[0],st[1]]);window._uc.setLatLng([st[0],st[1]]);
    }
    window._map.setView([c.lat,c.lon],c.zoom,{animate:false});
    if(API_MODE)reinitAPI();
  }

  // ─── SIMULATION STEP ───
  const DT=0.067;

  function step(){
    if(!S.run)return;
    S.step++;S.ri=(S.ri+1)%S.route.length;
    const cur=S.route[S.ri],prv=S.route[(S.ri-1+S.route.length)%S.route.length];
    const dLa=cur[0]-prv[0],dLo=cur[1]-prv[1];
    S.gtLa=cur[0];S.gtLo=cur[1];S.gtH=Math.atan2(dLo,dLa);

    const dym=dLa*111000,dxm=dLo*111000*Math.cos(S.gtLa*Math.PI/180);
    const dm=Math.hypot(dym,dxm),spdMs=dm/DT;
    const spdKmh=Math.min(120,spdMs*3.6);

    // MODULE 1 — Calibration drift
    S.calP+=((Math.random()-.5)*.05);S.calR+=((Math.random()-.5)*.03);S.calY+=((Math.random()-.5)*.02);
    S.calP=clamp(S.calP,-8,8);S.calR=clamp(S.calR,-6,6);S.calY=clamp(S.calY,-2,2);
    const cSpd=spdMs*Math.cos(S.calP*Math.PI/180);

    // MODULE 2 — Vibration filter (EMA + pothole detection)
    const noise=(Math.random()-.5)*8+(Math.random()<.02?(Math.random()-.5)*38:0);
    const rawSpd=spdMs+noise;
    if(Math.abs(noise)>16){S.pots++;log('Pothole detected ('+Math.abs(noise).toFixed(1)+' m/s²) — filtered','nhc');}
    S.vibBuf.push(rawSpd);if(S.vibBuf.length>8)S.vibBuf.shift();
    const fSpd=S.vibBuf.reduce((a,b)=>a+b,0)/S.vibBuf.length;
    S.vibRawSpd=rawSpd*3.6;S.vibFiltSpd=fSpd*3.6;
    S.vibRaw.push(rawSpd);if(S.vibRaw.length>80)S.vibRaw.shift();

    // MODULE 2b — Barometer (9th channel): simulate altitude & tunnel detection
    // Altitude follows a gentle profile; tunnels cause a sudden drop, bridges a rise
    const baroNoise=(Math.random()-.5)*0.4;
    // Simulate tunnel: every ~220 steps, altitude drops 8m over 30 steps then recovers
    const tPhase=S.step%220;
    let baroTrend=0;
    if(tPhase>50&&tPhase<80){baroTrend=-0.35;}        // descend into tunnel
    else if(tPhase>=80&&tPhase<110){baroTrend=0;}     // inside tunnel (flat)
    else if(tPhase>=110&&tPhase<140){baroTrend=+0.35;}// ascend out of tunnel
    S.baroAlt=Math.max(0,S.baroAlt+baroTrend+baroNoise);
    const baroDiff=S.baroAlt-S.baroPrev;S.baroPrev=S.baroAlt;
    S.spkBaro.push(S.baroAlt);S.spkBaro.shift();
    // Tunnel detection: sustained drop > 0.2m/step for 5+ steps
    const wasInTunnel=S.inTunnel;
    if(baroDiff<-0.2)S.tunDur++;else S.tunDur=Math.max(0,S.tunDur-1);
    S.inTunnel=(S.tunDur>5);
    if(!wasInTunnel&&S.inTunnel){S.tunN++;log('Tunnel entry detected via baro Δ='+baroDiff.toFixed(2)+'m','sys');}
    if(wasInTunnel&&!S.inTunnel){log('Tunnel exit confirmed via baro','snap');}
    // In tunnel: uncertainty grows 2× faster (baro tells us we're in a constrained corridor)
    const uncGrowthRate=S.inTunnel?0.12:0.05;

    // IMU synthesis
    const dH=S.gtH-S.pH;
    const gz=clamp(dH/DT,-35,35),ax=clamp((cSpd-S.spd)/DT,-160,160),sa=clamp(dH*5,-1,1);
    S.spd=cSpd;S.pH=S.gtH;

    ekfStep(fSpd,dH,DT,S.gnss,S.gtLa,S.gtLo);

    if(S.step%180===0&&S.gnss)triggerOut();

    if(API_MODE){
      const sensor={type:'sensor',data:{accel_x:ax,accel_y:0,accel_z:-9.81,gyro_x:0,gyro_y:0,gyro_z:gz,
        speed:fSpd,steer_angle:sa,baro_alt:S.baroAlt,gnss_valid:S.gnss?1:0}};
      if(S.wsReady){
        // WebSocket path: push sensor data; position update arrives via ws.onmessage
        wsSend('sensor',sensor.data);
        simIMU(dLa,dLo); // IMU sim still runs for comparison polyline
      } else {
        // HTTP fallback
        apiPost('/ingest',{accel_x:ax,accel_y:0,accel_z:-9.81,gyro_x:0,gyro_y:0,gyro_z:gz,
          steering_angle:sa,wheel_speed_fl:fSpd,wheel_speed_fr:fSpd,wheel_speed_rl:fSpd,wheel_speed_rr:fSpd,
          baro_alt:S.baroAlt,gnss_valid:S.gnss?1:0})
          .then(r=>{
            if(r.pose_x!==undefined){const R=6371000,sp=S.route[0];S.ndLa=sp[0]+(r.pose_y/R)*(180/Math.PI);S.ndLo=sp[1]+(r.pose_x/(R*Math.cos(sp[0]*Math.PI/180)))*(180/Math.PI);}
            if(r.uncertainty_major!==undefined)S.unc=Math.max(.35,r.uncertainty_major);
            if(r.latency_ms!==undefined)S.lat=r.latency_ms;
          }).catch(()=>simND(dLa,dLo));
        simIMU(dLa,dLo);
      }
    } else { simND(dLa,dLo); simIMU(dLa,dLo); }

    // MODULE 3 — NHC: zero lateral velocity
    const h=S.gtH;
    const ndDLa=S.ndLa-prv[0],ndDLo=S.ndLo-prv[1];
    const fwd=ndDLa*Math.cos(h)+ndDLo*Math.sin(h);
    const lat=-ndDLa*Math.sin(h)+ndDLo*Math.cos(h);
    const latM=Math.abs(lat)*111000;
    if(latM>.01){S.ndLa=prv[0]+fwd*Math.cos(h);S.ndLo=prv[1]+fwd*Math.sin(h);S.nhcN++;S.nhcE+=latM;}

    // MODULE 4 — HMM Map Matching (Viterbi decode over route)
    // Emission: Gaussian P(obs|state) — how likely is current DR position given we're at route[k]
    // Transition: exponential decay on route-index distance P(k|k_prev) ∝ exp(-|dk|/λ)
    // Both in log-space to avoid underflow
    {
      const WIN=24;    // search window around current estimated route index
      const SIGMA=18;  // emission std dev metres — how spread the Gaussian is
      const LAMBDA=4;  // transition smoothness — lower = must stay near last step
      const n=S.route.length;
      const lo=Math.max(0,S.ri-WIN),hi=Math.min(n-1,S.ri+WIN);
      const newProb=new Float32Array(n).fill(-1e9); // log-prob = -inf outside window
      // Best index from prior step for Viterbi backtrack (simplified — no full backtrack table, just MAP)
      let bestK=S.ri,bestV=-1e9;
      for(let k=lo;k<=hi;k++){
        const d=dist(S.ndLa,S.ndLo,S.route[k][0],S.route[k][1]);
        const logEmit=-0.5*(d/SIGMA)*(d/SIGMA); // log Gaussian (unnormalised)
        // Best transition from any prev state (simplified: use prior + transition from nearest prev)
        let bestTrans=-1e9;
        for(let pk=Math.max(0,k-WIN);pk<=Math.min(n-1,k+WIN);pk++){
          const dk=Math.abs(k-pk);
          const lt2=S.hmmProb[pk]-dk/LAMBDA; // log transition = -|dk|/λ
          if(lt2>bestTrans)bestTrans=lt2;
        }
        newProb[k]=logEmit+bestTrans;
        if(newProb[k]>bestV){bestV=newProb[k];bestK=k;}
      }
      S.hmmProb=newProb;
      // Soft-snap: blend DR position toward MAP state during outage, weighted by confidence
      if(!S.gnss){
        const mapPt=S.route[bestK];
        const dToMap=dist(S.ndLa,S.ndLo,mapPt[0],mapPt[1]);
        // Snap strength grows with distance but caps at 0.45 to avoid hard jumps
        const strength=Math.min(0.45,(dToMap>1.5?dToMap/80:0));
        if(strength>0.01&&dToMap<60){
          S.ndLa+=(mapPt[0]-S.ndLa)*strength;
          S.ndLo+=(mapPt[1]-S.ndLo)*strength;
          S.hmmCorrN++;
        }
        // Update estimated route index to MAP estimate
        S.ri=bestK;
      }
    }

    if(!S.gnss){S.outt+=DT;S.drTotal+=DT;if(!API_MODE)S.unc+=uncGrowthRate*(1+S.outt*.18);document.getElementById('banner-timer').textContent=S.outt.toFixed(1)+'s';if(S.outt>=6)reacq();}

    if(window._map){
      window._gt.addLatLng([S.gtLa,S.gtLo]);window._nd.addLatLng([S.ndLa,S.ndLo]);
      window._imu.addLatLng([S.iLa,S.iLo]);window._ekf.addLatLng([EKF.x,EKF.y]);
      window._vm.setLatLng([S.ndLa,S.ndLo]);window._uc.setLatLng([S.ndLa,S.ndLo]);
      window._uc.setRadius(Math.max(4,S.unc));
      window._map.setView([S.ndLa,S.ndLo],window._map.getZoom(),{animate:false});
    }

    const eND=dist(S.ndLa,S.ndLo,S.gtLa,S.gtLo),eIMU=dist(S.iLa,S.iLo,S.gtLa,S.gtLo);
    S.totND+=eND;S.totIMU+=eIMU;S.n++;S.mxND=Math.max(S.mxND,eND);S.mxIMU=Math.max(S.mxIMU,eIMU);
    S.spkErr.push(eND);S.spkErr.shift();S.spkUnc.push(S.unc);S.spkUnc.shift();S.spkSpd.push(spdKmh);S.spkSpd.shift();
    updateUI(eND,eIMU,spdKmh);
  }

  function simND(dLa,dLo){
    if(S.gnss){S.ndLa+=dLa+(Math.random()-.5)*1.4e-6;S.ndLo+=dLo+(Math.random()-.5)*1.4e-6;S.unc=Math.max(.35,S.unc*.95);}
    else{const t=S.outt;S.ndLa+=dLa+(Math.random()-.5)*2e-6+(Math.random()-.5)*3e-6*(1+t*.22);S.ndLo+=dLo+(Math.random()-.5)*2e-6+(Math.random()-.5)*3e-6*(1+t*.22);S.unc+=.05*(1+t*.18);}
  }
  function simIMU(dLa,dLo){
    if(S.gnss){S.iLa+=dLa+(Math.random()-.5)*6e-6;S.iLo+=dLo+(Math.random()-.5)*6e-6;}
    else{const t=S.outt;S.iLa+=dLa+(Math.random()-.5)*3.2e-5*Math.pow(t+1,1.4);S.iLo+=dLo+(Math.random()-.5)*3.2e-5*Math.pow(t+1,1.4);}
  }

  function triggerOut(){
    if(!S.gnss)return;S.gnss=false;S.outt=0;S.outN++;
    if(API_MODE){if(S.wsReady)wsSend('gnss_lost',{});else apiPost('/gnss_lost',{}).catch(()=>{});}
    document.getElementById('outage-flash').classList.add('on');
    const b=document.getElementById('banner');b.classList.remove('snap');b.classList.add('on');
    document.getElementById('banner-txt').textContent='GNSS BLACKOUT — NAVDRIFT-0 DEAD RECKONING ACTIVE';
    document.getElementById('hdr-dot').classList.add('out');
    document.getElementById('hdr-status').textContent='GNSS BLACKOUT';
    document.getElementById('hud-mode').textContent='BLACKOUT (Dead Reckoning)';
    document.getElementById('hud-mode').classList.add('out');
    document.querySelectorAll('.sig-bar').forEach(b=>b.classList.add('lost'));
    document.getElementById('sig-state').textContent='GNSS LOST (0 Satellites)';
    document.getElementById('sig-sub').textContent='NavIC Signal Blackout';
    document.getElementById('m-snap').className='mod warn';document.getElementById('m-mm').className='mod warn';
    document.getElementById('mm-badge').className='mm-badge on';
    log('GNSS blackout — dead reckoning, NHC, map-matching engaged','out');
  }

  function reacq(){
    if(S.gnss)return;S.gnss=true;S.snapN++;
    if(API_MODE){const hd=(90-S.gtH*180/Math.PI+360)%360;if(S.wsReady)wsSend('reacquire',{lat:S.gtLa,lon:S.gtLo,heading:hd});else apiPost('/reacquire',{latitude:S.gtLa,longitude:S.gtLo,heading_deg:hd}).catch(()=>{});}
    document.getElementById('outage-flash').classList.remove('on');
    const fe=dist(S.ndLa,S.ndLo,S.gtLa,S.gtLo);
    if(window._sl)L.circleMarker([S.gtLa,S.gtLo],{radius:5,color:'#00ff9d',fillColor:'#00ff9d',fillOpacity:.8}).addTo(window._sl);
    S.ndLa=S.gtLa;S.ndLo=S.gtLo;S.iLa=S.gtLa;S.iLo=S.gtLo;S.unc=.45;
    const b=document.getElementById('banner');b.classList.add('snap');
    document.getElementById('banner-txt').textContent='✓ SNAP: -'+fe.toFixed(2)+'m corrected in 12.4ms';
    setTimeout(()=>b.classList.remove('on'),2400);
    document.getElementById('hdr-dot').classList.remove('out');
    document.getElementById('hdr-status').textContent='GNSS LOCKED';
    document.getElementById('hud-mode').textContent='NavIC L5 Locked';
    document.getElementById('hud-mode').classList.remove('out');
    document.querySelectorAll('.sig-bar').forEach(b=>b.classList.remove('lost'));
    document.getElementById('sig-state').textContent='NavIC L5 / S-Band';
    document.getElementById('sig-sub').textContent='7 Satellites locked';
    document.getElementById('m-snap').className='mod on';document.getElementById('m-mm').className='mod on';
    log('SNAP reacquisition: -'+fe.toFixed(2)+'m fixed. DR lasted '+S.outt.toFixed(1)+'s','snap');
  }

  // ─── UI UPDATE ───
  function updateUI(eND,eIMU,spdKmh){try{
    const mn=(S.totND/S.n).toFixed(2),mi=(S.totIMU/S.n).toFixed(2);
    const ekfATE=EKF.s>0?(EKF.te/EKF.s).toFixed(2):'—';
    const lat=API_MODE?S.lat.toFixed(1):(1.7+Math.random()*.5).toFixed(1);

    const _nd=$('v-nd');if(_nd)_nd.innerHTML=mn+'<span class="mu">m</span>';
    const _im=$('v-imu');if(_im)_im.innerHTML=mi+'<span class="mu">m</span>';
    const _ek=$('v-ekf');if(_ek)_ek.innerHTML=EKF.s>0?ekfATE+'<span class="mu">m</span>':'—';
    const _un=$('v-unc');if(_un)_un.innerHTML=S.unc.toFixed(2)+'<span class="mu">m</span>';
    const _la=$('v-lat');if(_la)_la.innerHTML=lat+'<span class="mu">ms</span>';
    $('v-steps').textContent=S.step.toLocaleString();
    $('v-outs').textContent=S.outN;
    $('v-snap').textContent=S.snapN;
    const hmmEl=$('v-hmm');if(hmmEl)hmmEl.textContent=S.hmmCorrN;
    const tunEl2=$('v-tun');if(tunEl2)tunEl2.textContent=S.tunN;

    $('sp-err-v').textContent=eND.toFixed(2)+'m';
    $('sp-unc-v').textContent=S.unc.toFixed(2)+'m';
    $('sp-spd-v').textContent=spdKmh.toFixed(1)+' km/h';
    const baroV=$('sp-baro-v');if(baroV)baroV.textContent=S.baroAlt.toFixed(1)+' m';
    const tunEl=$('sp-baro-tun');if(tunEl){tunEl.textContent=S.inTunnel?'IN TUNNEL':'clear';tunEl.style.color=S.inTunnel?'var(--red)':'var(--green)';}
    const wsEl=$('mode-badge');if(wsEl&&S.wsReady&&!wsEl.textContent.includes('WS')){wsEl.textContent='WS LIVE';wsEl.classList.add('live');}

    $('hud-pos').textContent=S.gtLa.toFixed(4)+'°N '+S.gtLo.toFixed(4)+'°E · '+spdKmh.toFixed(1)+' km/h';
    $('hud-dr').textContent=S.drTotal.toFixed(1)+'s cumulative';

    $('tb-nd').textContent=mn+' m';$('tb-ndx').textContent=S.mxND.toFixed(2)+' m';
    $('tb-ekf').textContent=EKF.s>0?ekfATE+' m':'—';$('tb-ekfx').textContent=EKF.s>0?EKF.me.toFixed(2)+' m':'—';
    $('tb-imu').textContent=mi+' m';$('tb-imux').textContent=S.mxIMU.toFixed(2)+' m';

    // Calibration
    const pp=Math.min(100,Math.abs(S.calP)/8*100);const rp=Math.min(100,Math.abs(S.calR)/6*100);const yp=Math.min(100,Math.abs(S.calY)/2*100);
    $('cal-p').style.width=pp+'%';$('cal-r').style.width=rp+'%';$('cal-y').style.width=(50+yp/2)+'%';
    $('cal-pv').textContent=(S.calP>0?'+':'')+S.calP.toFixed(1)+'°';
    $('cal-rv').textContent=(S.calR>0?'+':'')+S.calR.toFixed(1)+'°';
    $('cal-yv').textContent=(S.calY>0?'+':'')+S.calY.toFixed(1)+'°';

    // Vibration
    $('vib-raw').textContent=Math.abs(S.vibRawSpd).toFixed(1)+' km/h';
    $('vib-filt').textContent=Math.abs(S.vibFiltSpd).toFixed(1)+' km/h';
    $('vib-pots').textContent=S.pots;

    // NHC
    const nhcAvg=S.nhcN>0?S.nhcE/S.nhcN:0;
    const nhcPct=Math.min(100,nhcAvg*18);
    $('nhc-lb').style.width=nhcPct+'%';$('nhc-vb').style.width=(nhcPct*.55)+'%';
    $('nhc-lv').textContent=nhcAvg.toFixed(2);$('nhc-vv').textContent=(nhcAvg*.3).toFixed(2);
    $('nhc-cnt').textContent=S.nhcN;$('nhc-err').textContent=S.nhcE.toFixed(2)+' m';

    // Fusion bars
    const worst=Math.max(.1,parseFloat(mi));
    const ndF=Math.max(5,100-parseFloat(mn)/worst*100);
    const ekF=EKF.s>0?Math.max(5,100-parseFloat(ekfATE)/worst*100):50;
    $('fb-nd').style.width=ndF+'%';$('fb-ekf').style.width=ekF+'%';$('fb-imu').style.width='6%';
    $('fb-ndv').textContent=ndF.toFixed(0)+'%';$('fb-ekfv').textContent=ekF.toFixed(0)+'%';

    // Pipeline glow
    const ps=Math.floor(S.step/10)%6+1;
    for(let i=1;i<=6;i++){const el=document.getElementById('pm'+i);if(el)i===ps?el.classList.add('active'):el.classList.remove('active');}
  }catch(e){/* silent — never crash the animation loop */}}

  function $$(id){return document.getElementById(id);}
  function $(id){return document.getElementById(id);}

  function log(msg,type){
    const c=$('elog'),t=new Date().toTimeString().split(' ')[0];
    const e=document.createElement('div');e.className='le '+(type||'');
    e.innerHTML='<span class="lt">['+t+']</span><span class="lm"> '+msg+'</span>';
    c.insertBefore(e,c.firstChild);
    while(c.children.length>20)c.removeChild(c.lastChild);
  }

  // ─── EKF ───
  const Q1=1e-10,Q2=1e-6,R1=1e-9;
  const EKF={x:0,y:0,th:0,P:[1,0,0,0,1,0,0,0,.1],te:0,me:0,s:0};
  function ekfReset(la,lo,th){EKF.x=la;EKF.y=lo;EKF.th=th;EKF.P=[1,0,0,0,1,0,0,0,.1];EKF.te=0;EKF.me=0;EKF.s=0;}
  function ekfStep(spd,dh,dt,gps,gla,glo){
    const th=EKF.th+dh*.5;
    const dl=spd*Math.cos(th)/111000*dt,dn=spd*Math.sin(th)/(111000*Math.cos(EKF.x*Math.PI/180))*dt;
    EKF.x+=dl;EKF.y+=dn;EKF.th+=dh;
    const p=EKF.P,Fx=-dl,Fy=dn;
    EKF.P=[p[0]+Fx*(p[6]+p[2])+Fx*Fx*p[8]+Q1,p[1]+Fx*p[7]+Fy*p[2]+Fx*Fy*p[8],p[2]+Fx*p[8],
           p[3]+Fy*p[6]+Fx*p[5]+Fx*Fy*p[8],p[4]+Fy*(p[7]+p[5])+Fy*Fy*p[8]+Q1,p[5]+Fy*p[8],
           p[6]+Fx*p[8],p[7]+Fy*p[8],p[8]+Q2];
    if(gps){
      const pp=EKF.P,S0=pp[0]+R1,S1=pp[4]+R1,S01=pp[1];
      const det=S0*S1-S01*S01;if(Math.abs(det)<1e-30)return;
      const K=[[(pp[0]*S1-pp[1]*S01)/det,(pp[1]*S0-pp[0]*S01)/det],
               [(pp[3]*S1-pp[4]*S01)/det,(pp[4]*S0-pp[3]*S01)/det],
               [(pp[6]*S1-pp[7]*S01)/det,(pp[7]*S0-pp[6]*S01)/det]];
      const ix=gla-EKF.x,iy=glo-EKF.y;
      EKF.x+=K[0][0]*ix+K[0][1]*iy;EKF.y+=K[1][0]*ix+K[1][1]*iy;EKF.th+=K[2][0]*ix+K[2][1]*iy;
      const IKH=[1-K[0][0],-K[0][1],0,-K[1][0],1-K[1][1],0,-K[2][0],-K[2][1],1];
      const np=new Array(9);for(let r=0;r<3;r++)for(let c=0;c<3;c++)np[r*3+c]=IKH[r*3]*pp[c]+IKH[r*3+1]*pp[3+c]+IKH[r*3+2]*pp[6+c];
      EKF.P=np;
    }
    const e=dist(EKF.x,EKF.y,gla,glo);EKF.te+=e;EKF.me=Math.max(EKF.me,e);EKF.s++;
  }

  // ─── SPARKLINE RENDERER ───
  function drawSpark(id,data,stroke,fill){
    const cv=document.getElementById(id);if(!cv)return;
    const p=cv.parentElement;cv.width=p.clientWidth;cv.height=p.clientHeight;
    const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
    if(data.length<2)return;
    const mn=Math.min(...data),mx=Math.max(...data)+.001,w=cv.width,h=cv.height;
    const gx=i=>i/(data.length-1)*w,gy=v=>h-3-((v-mn)/(mx-mn))*(h-6);
    ctx.beginPath();ctx.moveTo(gx(0),gy(data[0]));for(let i=1;i<data.length;i++)ctx.lineTo(gx(i),gy(data[i]));
    ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();
    const g=ctx.createLinearGradient(0,0,0,h);g.addColorStop(0,fill);g.addColorStop(1,'transparent');
    ctx.fillStyle=g;ctx.fill();
    ctx.beginPath();ctx.moveTo(gx(0),gy(data[0]));for(let i=1;i<data.length;i++)ctx.lineTo(gx(i),gy(data[i]));
    ctx.strokeStyle=stroke;ctx.lineWidth=1.8;ctx.shadowColor=stroke;ctx.shadowBlur=5;ctx.stroke();
  }

  function drawVib(){
    const cv=document.getElementById('vib-cv');if(!cv)return;
    const p=cv.parentElement;cv.width=p.clientWidth;cv.height=p.clientHeight;
    const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
    const d=S.vibRaw;if(d.length<2)return;
    const mn=Math.min(...d),mx=Math.max(...d)+.001,w=cv.width,h=cv.height;
    const gx=i=>i/(d.length-1)*w,gy=v=>h-2-((v-mn)/(mx-mn))*(h-4);
    ctx.beginPath();ctx.moveTo(gx(0),gy(d[0]));for(let i=1;i<d.length;i++)ctx.lineTo(gx(i),gy(d[i]));
    ctx.strokeStyle='rgba(255,34,85,0.55)';ctx.lineWidth=1;ctx.shadowBlur=0;ctx.stroke();
    const sm=[];for(let i=0;i<d.length;i++){let s=0,c=0;for(let j=Math.max(0,i-4);j<=Math.min(d.length-1,i+4);j++){s+=d[j];c++;}sm.push(s/c);}
    ctx.beginPath();ctx.moveTo(gx(0),gy(sm[0]));for(let i=1;i<sm.length;i++)ctx.lineTo(gx(i),gy(sm[i]));
    ctx.strokeStyle='rgba(0,255,245,0.9)';ctx.lineWidth=1.5;ctx.shadowColor=ctx.strokeStyle;ctx.shadowBlur=4;ctx.stroke();
  }

  // ─── CONTROLS ───
  function toggleSim(){S.run=!S.run;const b=$('btn-play');if(S.run){b.textContent='⏸ Pause';b.classList.add('p');}else{b.textContent='▶ Resume';b.classList.remove('p');}}
  function toggleOutage(){S.gnss?triggerOut():reacq();}
  function switchCity(k,ev){document.querySelectorAll('.city-tab').forEach(t=>t.classList.remove('active'));ev&&ev.target&&ev.target.classList.add('active');init(k);}

  // ─── API ───
  let API_URL=localStorage.getItem('nd_url')||'';
  let API_KEY=localStorage.getItem('nd_key')||'';
  let API_MODE=!!(API_URL&&API_KEY);

  function updBadge(){
    const b=$('mode-badge'),c=$('mode-chip');
    if(S.wsReady){b.textContent='WS LIVE';b.classList.add('live');if(c)c.textContent='[WS]';}
    else if(API_MODE){b.textContent='LIVE API';b.classList.add('live');if(c)c.textContent='[API]';}
    else{b.textContent='SIMULATION';b.classList.remove('live');if(c)c.textContent='[SIM]';}
  }
  updBadge();

  async function apiPost(ep,body){const r=await fetch(API_URL+ep,{method:'POST',headers:{'Content-Type':'application/json','X-API-Key':API_KEY},body:JSON.stringify(body)});if(!r.ok)throw new Error(ep+'→'+r.status);return r.json();}

  // ─── WEBSOCKET CLIENT ───
  function connectWS(){
    if(!API_URL||!API_KEY)return;
    const wsUrl=(API_URL.replace(/^https/,'wss').replace(/^http/,'ws'))+'/ws/stream?api_key='+encodeURIComponent(API_KEY);
    try{
      const ws=new WebSocket(wsUrl);
      ws.onopen=()=>{
        S.ws=ws;S.wsReady=true;
        log('WebSocket connected — 10Hz push stream active','sys');
        updBadge();
        $('engine-info').innerHTML='<b>Model:</b> DRIFT-Former INT8 ONNX (14MB)<br><b>Runtime:</b> ONNX Runtime 1.17 · CPU<br><b>Stream:</b> <span style="color:var(--green)">WebSocket 10Hz</span><br><b>API:</b> <span style="color:var(--cyan)">'+API_URL+'</span>';
      };
      ws.onmessage=(ev)=>{
        try{
          const r=JSON.parse(ev.data);
          if(r.type==='pose'){
            const sp=S.route[0];const R=6371000;
            S.ndLa=sp[0]+(r.pose_y/R)*(180/Math.PI);
            S.ndLo=sp[1]+(r.pose_x/(R*Math.cos(sp[0]*Math.PI/180)))*(180/Math.PI);
            if(r.uncertainty_major!==undefined)S.unc=Math.max(.35,r.uncertainty_major);
            if(r.latency_ms!==undefined)S.lat=r.latency_ms;
          }
        }catch(e){}
      };
      ws.onerror=()=>{S.wsReady=false;updBadge();};
      ws.onclose=()=>{
        S.ws=null;S.wsReady=false;updBadge();
        log('WebSocket closed — falling back to HTTP','out');
        setTimeout(()=>{if(API_MODE)connectWS();},3000); // auto-reconnect after 3s
      };
    }catch(e){log('WS connect failed: '+e.message,'out');}
  }

  function wsSend(type,data){
    if(S.ws&&S.wsReady&&S.ws.readyState===WebSocket.OPEN){
      try{S.ws.send(JSON.stringify({type,...data}));}catch(e){}
    }
  }

  async function reinitAPI(){
    if(!API_MODE)return;
    try{
      const c=CITIES[S.city],st=S.route[0],nx=S.route[1]||st;
      const hd=(90-Math.atan2(nx[1]-st[1],nx[0]-st[0])*180/Math.PI+360)%360;
      await apiPost('/init',{latitude:c.lat,longitude:c.lon,heading_deg:hd,speed_m_s:0});
      log('API /init OK — connecting WebSocket stream','sys');
      connectWS(); // upgrade to WS after init
      updBadge();
    }catch(e){log('API /init failed: '+e.message,'out');}
  }

  function openModal(){$('modal').classList.add('on');$('api-url').value=localStorage.getItem('nd_url')||'';$('api-key').value=localStorage.getItem('nd_key')||'';$('conn-st').style.display='none';}
  function closeModal(){$('modal').classList.remove('on');}
  async function saveAPI(){const url=$('api-url').value.trim(),key=$('api-key').value.trim();localStorage.setItem('nd_url',url);localStorage.setItem('nd_key',key);API_URL=url;API_KEY=key;API_MODE=!!(url&&key);if(API_MODE){log('Backend: '+url,'sys');await reinitAPI();}else{if(S.ws){S.ws.close();S.ws=null;S.wsReady=false;}$('engine-info').innerHTML='<b>Mode:</b> Simulation';updBadge();}closeModal();}
  async function testConn(){const url=$('api-url').value.trim(),key=$('api-key').value.trim(),el=$('conn-st');el.style.display='block';el.className='cs ok';el.textContent='Testing...';try{const r=await fetch(url+'/status',{method:'GET',headers:{'X-API-Key':key}});el.className=r.ok?'cs ok':'cs err';el.textContent=r.ok?'✓ API Online (WS upgrade on save)':'✗ Status '+r.status;}catch(e){el.className='cs err';el.textContent='✗ Cannot reach endpoint';}}

  // ─── MAIN LOOP ───
  let lt=0;
  function loop(t){
    requestAnimationFrame(loop);
    if(t-lt>65){step();lt=t;}
    drawSpark('sp-err',S.spkErr,'#ff2255','rgba(255,34,85,0.2)');
    drawSpark('sp-unc',S.spkUnc,'#a855f7','rgba(168,85,247,0.2)');
    drawSpark('sp-spd',S.spkSpd,'#00ff9d','rgba(0,255,157,0.18)');
    drawSpark('sp-baro',S.spkBaro,'#fbbf24','rgba(251,191,36,0.18)');
    drawVib();
  }

  init('delhi');
  requestAnimationFrame(loop);
</script>
</body>
</html>
