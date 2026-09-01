<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>NAVDRIFT-0 Mobile</title>
  <link rel="manifest" href="/manifest.json">
  <meta name="theme-color" content="#080810">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="NAVDRIFT-0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
  <style>
    :root {
      --bg:#080810;--surface:#0d0e1c;--card:rgba(12,14,26,0.9);
      --cyan:#00fff5;--cyan-d:rgba(0,255,245,0.12);--cyan-g:rgba(0,255,245,0.4);
      --violet:#a855f7;--violet-d:rgba(168,85,247,0.12);
      --red:#ff2255;--red-d:rgba(255,34,85,0.15);--red-g:rgba(255,34,85,0.4);
      --green:#00ff9d;--green-d:rgba(0,255,157,0.12);--green-g:rgba(0,255,157,0.4);
      --yellow:#fbbf24;--orange:#f97316;
      --b1:rgba(255,255,255,0.08);--b2:rgba(255,255,255,0.04);
      --t1:#f1f5f9;--t2:#94a3b8;--t3:#475569;
      --mono:'JetBrains Mono',monospace;--sans:'Space Grotesk',sans-serif;
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
    html,body{width:100vw;height:100vh;overflow:hidden;background:var(--bg);color:var(--t1);font-family:var(--sans);user-select:none;-webkit-tap-highlight-color:transparent;}

    /* ── LAYOUT: header / map / bottom-sheet ── */
    .app{display:grid;grid-template-rows:52px 1fr auto;height:100vh;width:100vw;}

    /* ── HEADER ── */
    header{display:flex;align-items:center;justify-content:space-between;padding:0 12px;
      background:rgba(6,6,14,0.97);border-bottom:1px solid var(--b1);z-index:30;}
    .brand{display:flex;align-items:center;gap:8px;}
    .brand-name{font-size:.9rem;font-weight:700;letter-spacing:.1em;
      background:linear-gradient(135deg,var(--cyan),var(--violet));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .brand-sub{font-size:.48rem;color:var(--t3);letter-spacing:.12em;font-family:var(--mono);}
    .hdr-right{display:flex;align-items:center;gap:8px;}
    .pill{display:flex;align-items:center;gap:5px;background:rgba(0,0,0,0.3);
      padding:4px 10px;border-radius:16px;border:1px solid var(--b1);
      font-family:var(--mono);font-size:.58rem;font-weight:600;letter-spacing:.06em;}
    .dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green-g);transition:all .3s;}
    .dot.out{background:var(--red);box-shadow:0 0 8px var(--red-g);animation:blink .7s step-end infinite;}
    @keyframes blink{50%{opacity:0;}}
    .badge{font-size:.52rem;font-weight:700;padding:3px 8px;border-radius:5px;letter-spacing:.05em;}
    .badge-sim{background:var(--cyan-d);color:var(--cyan);border:1px solid rgba(0,255,245,0.2);}
    .badge-live{background:var(--green-d);color:var(--green);border:1px solid rgba(0,255,157,0.25);}

    /* ── MAP ── */
    .map-wrap{position:relative;min-height:0;}
    #map{width:100%;height:100%;background:#060610;}
    .leaflet-container{background:#060610!important;}
    .hud{position:absolute;z-index:500;pointer-events:none;}
    .hud.tl{top:8px;left:8px;}
    .hud.tr{top:8px;right:8px;}
    .hbox{background:rgba(6,6,14,0.88);backdrop-filter:blur(8px);border:1px solid var(--b1);
      border-radius:8px;padding:5px 9px;}
    .hl{font-size:.46rem;font-family:var(--mono);color:var(--t3);letter-spacing:.07em;text-transform:uppercase;}
    .hv{font-size:.68rem;font-weight:700;font-family:var(--mono);color:var(--cyan);margin-top:1px;}
    .hv.out{color:var(--red);}
    .leg-row{display:flex;align-items:center;gap:4px;margin-top:3px;}
    .leg-line{width:14px;height:2px;border-radius:1px;}
    .leg-txt{font-size:.46rem;font-family:var(--mono);color:var(--t2);}

    /* ── BOTTOM SHEET ── */
    .sheet{background:rgba(8,8,16,0.98);border-top:1px solid var(--b1);
      backdrop-filter:blur(20px);padding:10px 14px 14px;z-index:20;}

    /* City selector strip */
    .city-strip{display:flex;gap:5px;overflow-x:auto;padding-bottom:6px;margin-bottom:8px;
      scrollbar-width:none;}
    .city-strip::-webkit-scrollbar{display:none;}
    .city-btn{padding:5px 14px;font-size:.6rem;font-weight:600;color:var(--t2);
      background:rgba(255,255,255,0.04);border:1px solid var(--b1);border-radius:16px;
      cursor:pointer;white-space:nowrap;flex-shrink:0;transition:all .2s;}
    .city-btn.active{color:var(--cyan);background:var(--cyan-d);border-color:rgba(0,255,245,0.25);}

    /* Metrics row */
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;}
    .met{background:var(--b2);border:1px solid var(--b1);border-radius:8px;padding:7px 6px;text-align:center;}
    .met-lbl{font-size:.45rem;font-family:var(--mono);color:var(--t3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px;}
    .met-val{font-size:.78rem;font-weight:700;font-family:var(--mono);}

    /* Baro row */
    .baro-row{display:flex;align-items:center;justify-content:space-between;
      background:var(--b2);border:1px solid var(--b1);border-radius:7px;padding:6px 10px;margin-bottom:8px;}
    .baro-lbl{font-size:.52rem;font-family:var(--mono);color:var(--t2);}
    .baro-val{font-size:.72rem;font-weight:700;font-family:var(--mono);color:var(--yellow);}
    .tun-badge{font-size:.5rem;font-family:var(--mono);padding:2px 7px;border-radius:4px;
      font-weight:700;transition:all .3s;}
    .tun-badge.clear{background:var(--green-d);color:var(--green);}
    .tun-badge.tunnel{background:var(--red-d);color:var(--red);animation:blink .7s step-end infinite;}

    /* Controls row */
    .ctrl-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
    .cbtn{padding:10px;font-size:.7rem;font-weight:700;border-radius:10px;border:1px solid;
      cursor:pointer;transition:all .2s;font-family:var(--sans);display:flex;align-items:center;justify-content:center;gap:6px;}
    .cbtn.play{background:var(--cyan-d);color:var(--cyan);border-color:rgba(0,255,245,0.3);}
    .cbtn.gnss{background:var(--red-d);color:var(--red);border-color:rgba(255,34,85,0.3);}

    /* Outage banner */
    #ob{position:fixed;top:52px;left:0;right:0;z-index:100;background:rgba(160,0,40,0.95);
      backdrop-filter:blur(10px);padding:7px 14px;display:none;
      font-family:var(--mono);font-size:.65rem;font-weight:700;letter-spacing:.08em;
      text-align:center;border-bottom:1px solid rgba(255,50,90,0.5);}
    #ob.snap{background:rgba(0,80,40,0.95);border-color:rgba(0,255,157,0.4);}
    #ob.on{display:block;}
    #outage-flash{position:fixed;inset:0;z-index:1;pointer-events:none;border:0 solid var(--red);opacity:0;transition:all .2s;}
    #outage-flash.on{border:3px solid var(--red);box-shadow:inset 0 0 60px rgba(255,34,85,0.2);opacity:1;animation:rfp 1.3s ease-in-out infinite alternate;}
    @keyframes rfp{0%{box-shadow:inset 0 0 30px rgba(255,34,85,0.2)}100%{box-shadow:inset 0 0 80px rgba(255,34,85,0.5)}}
  </style>
</head>
<body>
<div id="outage-flash"></div>
<div id="ob">GNSS BLACKOUT — DEAD RECKONING ACTIVE · <span id="ob-t">0.0s</span></div>

<div class="app">
  <header>
    <div class="brand">
      <svg width="26" height="26" viewBox="0 0 32 32" fill="none">
        <circle cx="16" cy="16" r="13" stroke="#00fff5" stroke-width="1.2" opacity=".4"/>
        <circle cx="16" cy="16" r="7" stroke="#a855f7" stroke-width="1" opacity=".6"/>
        <circle cx="16" cy="16" r="2.5" fill="#00fff5"/>
        <line x1="16" y1="3" x2="16" y2="8" stroke="#00fff5" stroke-width="1.4"/>
      </svg>
      <div>
        <div class="brand-name">NAVDRIFT-0</div>
        <div class="brand-sub">ISRO SIH 2026 · PS #26168</div>
      </div>
    </div>
    <div class="hdr-right">
      <div class="pill"><div class="dot" id="hdr-dot"></div><span id="hdr-st">LOCKED</span></div>
      <span class="badge badge-sim" id="mode-b">SIM</span>
    </div>
  </header>

  <div class="map-wrap">
    <div id="map"></div>
    <div class="hud tl"><div class="hbox">
      <div class="hl">Position</div>
      <div class="hv" id="hud-pos">—</div>
      <div class="hl" style="margin-top:3px">Mode</div>
      <div class="hv" id="hud-mode">NavIC L5 Locked</div>
    </div></div>
    <div class="hud tr"><div class="hbox">
      <div class="leg-row"><div class="leg-line" style="background:#00fff5"></div><span class="leg-txt">GT</span></div>
      <div class="leg-row"><div class="leg-line" style="background:#a855f7"></div><span class="leg-txt">NAVDRIFT</span></div>
      <div class="leg-row"><div class="leg-line" style="background:#ff2255;border-top:1px dashed #ff2255;height:0"></div><span class="leg-txt">IMU</span></div>
      <div class="leg-row"><div class="leg-line" style="background:#fbbf24"></div><span class="leg-txt">EKF</span></div>
    </div></div>
  </div>

  <div class="sheet">
    <div class="city-strip">
      <button class="city-btn active" onclick="switchCity('delhi',this)">Delhi</button>
      <button class="city-btn" onclick="switchCity('mumbai',this)">Mumbai</button>
      <button class="city-btn" onclick="switchCity('bengaluru',this)">Bengaluru</button>
      <button class="city-btn" onclick="switchCity('chennai',this)">Chennai</button>
      <button class="city-btn" onclick="switchCity('hyderabad',this)">Hyderabad</button>
    </div>

    <div class="metrics">
      <div class="met"><div class="met-lbl">ND-0 ATE</div><div class="met-val" id="v-nd" style="color:var(--cyan)">—</div></div>
      <div class="met"><div class="met-lbl">EKF ATE</div><div class="met-val" id="v-ekf" style="color:var(--violet)">—</div></div>
      <div class="met"><div class="met-lbl">Uncert σ</div><div class="met-val" id="v-unc" style="color:var(--yellow)">—</div></div>
      <div class="met"><div class="met-lbl">Outages</div><div class="met-val" id="v-out" style="color:var(--red)">0</div></div>
    </div>

    <div class="baro-row">
      <div>
        <div class="baro-lbl">Barometer Altitude · 9th channel</div>
        <div class="baro-val" id="v-baro">— m</div>
      </div>
      <span class="tun-badge clear" id="tun-b">clear</span>
    </div>

    <div class="ctrl-row">
      <button class="cbtn play" id="btn-play" ontouchstart="" onclick="toggleSim()">⏸ Pause</button>
      <button class="cbtn gnss" ontouchstart="" onclick="toggleOutage()">📡 GNSS Toggle</button>
    </div>
  </div>
</div>

<script>
  // ─── SHARED CITY DATA ───
  const CITIES={
    delhi:   {name:'Delhi',   lat:28.6315,lon:77.2167,zoom:14,wp:[[28.6315,77.2167],[28.6330,77.2210],[28.6350,77.2260],[28.6360,77.2310],[28.6340,77.2360],[28.6310,77.2400],[28.6270,77.2420],[28.6230,77.2410],[28.6190,77.2380],[28.6160,77.2330],[28.6150,77.2270],[28.6160,77.2210],[28.6190,77.2160],[28.6230,77.2130],[28.6270,77.2130],[28.6315,77.2167]]},
    mumbai:  {name:'Mumbai',  lat:18.9322,lon:72.8264,zoom:14,wp:[[18.9322,72.8264],[18.9350,72.8240],[18.9380,72.8220],[18.9420,72.8215],[18.9460,72.8230],[18.9490,72.8260],[18.9510,72.8300],[18.9500,72.8340],[18.9470,72.8370],[18.9430,72.8380],[18.9390,72.8370],[18.9350,72.8340],[18.9320,72.8310],[18.9310,72.8280],[18.9322,72.8264]]},
    bengaluru:{name:'Bengaluru',lat:12.9352,lon:77.6245,zoom:13,wp:[[12.9352,77.6245],[12.9400,77.6320],[12.9450,77.6390],[12.9480,77.6470],[12.9460,77.6560],[12.9410,77.6620],[12.9340,77.6650],[12.9270,77.6630],[12.9210,77.6580],[12.9180,77.6500],[12.9190,77.6410],[12.9230,77.6330],[12.9280,77.6270],[12.9320,77.6250],[12.9352,77.6245]]},
    chennai: {name:'Chennai', lat:13.0524,lon:80.2580,zoom:14,wp:[[13.0524,80.2580],[13.0560,80.2560],[13.0600,80.2530],[13.0640,80.2500],[13.0680,80.2470],[13.0710,80.2440],[13.0730,80.2400],[13.0720,80.2360],[13.0690,80.2330],[13.0650,80.2310],[13.0610,80.2320],[13.0570,80.2340],[13.0540,80.2370],[13.0520,80.2410],[13.0510,80.2450],[13.0520,80.2510],[13.0524,80.2580]]},
    hyderabad:{name:'Hyderabad',lat:17.4065,lon:78.4772,zoom:13,wp:[[17.4065,78.4772],[17.4120,78.4850],[17.4170,78.4940],[17.4180,78.5040],[17.4150,78.5130],[17.4090,78.5190],[17.4010,78.5210],[17.3930,78.5180],[17.3870,78.5110],[17.3840,78.5010],[17.3850,78.4910],[17.3900,78.4830],[17.3960,78.4780],[17.4010,78.4760],[17.4065,78.4772]]}
  };
  function interp(wp,n){const pts=[],tot=wp.length-1;for(let i=0;i<n;i++){const t=i/n*tot,idx=Math.floor(t),f=t-idx,a=wp[Math.min(idx,tot)],b=wp[Math.min(idx+1,tot)];pts.push([a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f]);}return pts;}
  function dist(la1,lo1,la2,lo2){const dy=(la2-la1)*Math.PI/180*6371000,dx=(lo2-lo1)*Math.PI/180*6371000*Math.cos(la1*Math.PI/180);return Math.hypot(dy,dx);}
  function clamp(v,a,b){return Math.max(a,Math.min(b,v));}

  const S={city:'delhi',run:true,step:0,gnss:true,outt:0,outN:0,snapN:0,drTotal:0,
    route:[],ri:0,gtLa:0,gtLo:0,gtH:0,ndLa:0,ndLo:0,iLa:0,iLo:0,unc:0.45,
    totND:0,totIMU:0,n:0,spd:0,pH:0,nhcN:0,nhcE:0,
    calP:3.2,calR:-1.4,calY:0.1,vibBuf:[],
    baroAlt:220,baroPrev:220,inTunnel:false,tunDur:0,tunN:0,
    hmmProb:null,hmmCorrN:0};

  const EKF={x:0,y:0,th:0,P:[1,0,0,0,1,0,0,0,.1],te:0,me:0,s:0};
  function ekfReset(la,lo,th){EKF.x=la;EKF.y=lo;EKF.th=th;EKF.P=[1,0,0,0,1,0,0,0,.1];EKF.te=0;EKF.me=0;EKF.s=0;}
  function ekfStep(spd,dh,dt,gps,gla,glo){
    const th=EKF.th+dh*.5,dl=spd*Math.cos(th)/111000*dt,dn=spd*Math.sin(th)/(111000*Math.cos(EKF.x*Math.PI/180))*dt;
    EKF.x+=dl;EKF.y+=dn;EKF.th+=dh;
    const p=EKF.P,Fx=-dl,Fy=dn,Q1=1e-10,Q2=1e-6,R1=1e-9;
    EKF.P=[p[0]+Fx*(p[6]+p[2])+Fx*Fx*p[8]+Q1,p[1]+Fx*p[7]+Fy*p[2]+Fx*Fy*p[8],p[2]+Fx*p[8],
           p[3]+Fy*p[6]+Fx*p[5]+Fx*Fy*p[8],p[4]+Fy*(p[7]+p[5])+Fy*Fy*p[8]+Q1,p[5]+Fy*p[8],
           p[6]+Fx*p[8],p[7]+Fy*p[8],p[8]+Q2];
    if(gps){const pp=EKF.P,S0=pp[0]+R1,S1=pp[4]+R1,S01=pp[1],det=S0*S1-S01*S01;if(Math.abs(det)<1e-30)return;
      const K=[[(pp[0]*S1-pp[1]*S01)/det,(pp[1]*S0-pp[0]*S01)/det],[(pp[3]*S1-pp[4]*S01)/det,(pp[4]*S0-pp[3]*S01)/det],[(pp[6]*S1-pp[7]*S01)/det,(pp[7]*S0-pp[6]*S01)/det]];
      const ix=gla-EKF.x,iy=glo-EKF.y;EKF.x+=K[0][0]*ix+K[0][1]*iy;EKF.y+=K[1][0]*ix+K[1][1]*iy;EKF.th+=K[2][0]*ix+K[2][1]*iy;
      const IKH=[1-K[0][0],-K[0][1],0,-K[1][0],1-K[1][1],0,-K[2][0],-K[2][1],1],np=new Array(9);
      for(let r=0;r<3;r++)for(let c=0;c<3;c++)np[r*3+c]=IKH[r*3]*p[c]+IKH[r*3+1]*p[3+c]+IKH[r*3+2]*p[6+c];EKF.P=np;}
    const e=dist(EKF.x,EKF.y,gla,glo);EKF.te+=e;EKF.me=Math.max(EKF.me,e);EKF.s++;
  }

  const DT=0.067;
  function init(key){
    S.city=key;const c=CITIES[key];
    S.route=interp(c.wp,300);S.ri=0;
    const st=S.route[0];
    S.gtLa=S.ndLa=S.iLa=st[0];S.gtLo=S.ndLo=S.iLo=st[1];
    S.step=0;S.gnss=true;S.outt=0;S.outN=0;S.snapN=0;S.drTotal=0;S.unc=0.45;
    S.totND=0;S.totIMU=0;S.n=0;S.spd=0;S.pH=0;S.nhcN=0;S.nhcE=0;
    S.vibBuf=[];S.baroAlt=220;S.baroPrev=220;S.inTunnel=false;S.tunDur=0;S.tunN=0;S.hmmCorrN=0;
    S.hmmProb=new Float32Array(S.route.length).fill(-Math.log(S.route.length));
    ekfReset(st[0],st[1],0);
    if(!window._map){
      window._map=L.map('map',{zoomControl:false,attributionControl:false});
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19}).addTo(window._map);
      const s=document.createElement('style');s.textContent='.leaflet-tile-pane{filter:brightness(0.26) saturate(0.55) hue-rotate(172deg)}';document.head.appendChild(s);
      window._gt=L.polyline([],{color:'#00fff5',weight:2,opacity:.9}).addTo(window._map);
      window._nd=L.polyline([],{color:'#a855f7',weight:2.5,opacity:.9}).addTo(window._map);
      window._imu=L.polyline([],{color:'#ff2255',weight:1.5,dashArray:'6,5',opacity:.7}).addTo(window._map);
      window._ekf=L.polyline([],{color:'#fbbf24',weight:1.5,opacity:.7}).addTo(window._map);
      window._vm=L.marker(st,{icon:L.divIcon({className:'',iconSize:[14,14],iconAnchor:[7,7],
        html:'<div style="width:14px;height:14px;border-radius:50%;background:#00fff5;box-shadow:0 0 10px #00fff5;border:2px solid #fff"></div>'})}).addTo(window._map);
      window._uc=L.circle(st,{radius:5,color:'#a855f7',fillColor:'#a855f7',fillOpacity:.1,weight:1}).addTo(window._map);
    } else {
      window._gt.setLatLngs([]);window._nd.setLatLngs([]);window._imu.setLatLngs([]);window._ekf.setLatLngs([]);
      window._vm.setLatLng(st);window._uc.setLatLng(st);
    }
    window._map.setView([c.lat,c.lon],c.zoom,{animate:false});
  }

  function step(){
    if(!S.run)return;
    S.step++;S.ri=(S.ri+1)%S.route.length;
    const cur=S.route[S.ri],prv=S.route[(S.ri-1+S.route.length)%S.route.length];
    const dLa=cur[0]-prv[0],dLo=cur[1]-prv[1];
    S.gtLa=cur[0];S.gtLo=cur[1];S.gtH=Math.atan2(dLo,dLa);
    const dym=dLa*111000,dxm=dLo*111000*Math.cos(S.gtLa*Math.PI/180),dm=Math.hypot(dym,dxm),spdMs=dm/DT;

    // Calibration
    S.calP=clamp(S.calP+(Math.random()-.5)*.05,-8,8);
    S.calR=clamp(S.calR+(Math.random()-.5)*.03,-6,6);
    S.calY=clamp(S.calY+(Math.random()-.5)*.02,-2,2);
    const cSpd=spdMs*Math.cos(S.calP*Math.PI/180);

    // Vibration filter
    const noise=(Math.random()-.5)*8+(Math.random()<.02?(Math.random()-.5)*38:0);
    S.vibBuf.push(spdMs+noise);if(S.vibBuf.length>8)S.vibBuf.shift();
    const fSpd=S.vibBuf.reduce((a,b)=>a+b,0)/S.vibBuf.length;

    // Baro
    const tPhase=S.step%220;
    let baroTrend=tPhase>50&&tPhase<80?-0.35:tPhase>=110&&tPhase<140?0.35:0;
    S.baroAlt=Math.max(0,S.baroAlt+baroTrend+(Math.random()-.5)*.4);
    const bd=S.baroAlt-S.baroPrev;S.baroPrev=S.baroAlt;
    const wasT=S.inTunnel;
    if(bd<-0.2)S.tunDur++;else S.tunDur=Math.max(0,S.tunDur-1);
    S.inTunnel=S.tunDur>5;if(!wasT&&S.inTunnel)S.tunN++;
    const uncRate=S.inTunnel?.12:.05;

    // IMU
    const dH=S.gtH-S.pH;
    S.spd=cSpd;S.pH=S.gtH;
    ekfStep(fSpd,dH,DT,S.gnss,S.gtLa,S.gtLo);
    if(S.step%180===0&&S.gnss)triggerOut();

    // Sim ND / IMU
    if(S.gnss){S.ndLa+=dLa+(Math.random()-.5)*1.4e-6;S.ndLo+=dLo+(Math.random()-.5)*1.4e-6;S.unc=Math.max(.35,S.unc*.95);}
    else{const t=S.outt;S.ndLa+=dLa+(Math.random()-.5)*2e-6+(Math.random()-.5)*3e-6*(1+t*.22);S.ndLo+=dLo+(Math.random()-.5)*2e-6+(Math.random()-.5)*3e-6*(1+t*.22);S.unc+=uncRate*(1+t*.18);}
    if(S.gnss){S.iLa+=dLa+(Math.random()-.5)*6e-6;S.iLo+=dLo+(Math.random()-.5)*6e-6;}
    else{const t=S.outt;S.iLa+=dLa+(Math.random()-.5)*3.2e-5*Math.pow(t+1,1.4);S.iLo+=dLo+(Math.random()-.5)*3.2e-5*Math.pow(t+1,1.4);}

    // NHC
    const h=S.gtH,ndDLa=S.ndLa-prv[0],ndDLo=S.ndLo-prv[1];
    const fwd=ndDLa*Math.cos(h)+ndDLo*Math.sin(h),lat=-ndDLa*Math.sin(h)+ndDLo*Math.cos(h),latM=Math.abs(lat)*111000;
    if(latM>.01){S.ndLa=prv[0]+fwd*Math.cos(h);S.ndLo=prv[1]+fwd*Math.sin(h);S.nhcN++;}

    // HMM map match
    if(!S.gnss){
      const WIN=24,SIGMA=18,LAMBDA=4,n=S.route.length;
      const lo2=Math.max(0,S.ri-WIN),hi=Math.min(n-1,S.ri+WIN);
      const newP=new Float32Array(n).fill(-1e9);let bestK=S.ri,bestV=-1e9;
      for(let k=lo2;k<=hi;k++){
        const d=dist(S.ndLa,S.ndLo,S.route[k][0],S.route[k][1]);
        const le=-0.5*(d/SIGMA)*(d/SIGMA);
        let bt=-1e9;for(let pk=Math.max(0,k-WIN);pk<=Math.min(n-1,k+WIN);pk++){const lt2=S.hmmProb[pk]-Math.abs(k-pk)/LAMBDA;if(lt2>bt)bt=lt2;}
        newP[k]=le+bt;if(newP[k]>bestV){bestV=newP[k];bestK=k;}
      }
      S.hmmProb=newP;
      const mp=S.route[bestK],dm2=dist(S.ndLa,S.ndLo,mp[0],mp[1]);
      const str=Math.min(.45,(dm2>1.5?dm2/80:0));
      if(str>.01&&dm2<60){S.ndLa+=(mp[0]-S.ndLa)*str;S.ndLo+=(mp[1]-S.ndLo)*str;S.hmmCorrN++;S.ri=bestK;}
    }

    if(!S.gnss){S.outt+=DT;S.drTotal+=DT;document.getElementById('ob-t').textContent=S.outt.toFixed(1)+'s';if(S.outt>=6)reacq();}

    if(window._map){
      window._gt.addLatLng([S.gtLa,S.gtLo]);window._nd.addLatLng([S.ndLa,S.ndLo]);
      window._imu.addLatLng([S.iLa,S.iLo]);window._ekf.addLatLng([EKF.x,EKF.y]);
      window._vm.setLatLng([S.ndLa,S.ndLo]);window._uc.setLatLng([S.ndLa,S.ndLo]);
      window._uc.setRadius(Math.max(4,S.unc));
      window._map.setView([S.ndLa,S.ndLo],window._map.getZoom(),{animate:false});
    }

    const eND=dist(S.ndLa,S.ndLo,S.gtLa,S.gtLo);
    S.totND+=eND;S.n++;
    const mn=S.n>0?(S.totND/S.n).toFixed(1):'—';
    const ekfATE=EKF.s>0?(EKF.te/EKF.s).toFixed(1):'—';
    try{
      document.getElementById('v-nd').textContent=mn+'m';
      document.getElementById('v-ekf').textContent=ekfATE+'m';
      document.getElementById('v-unc').textContent=S.unc.toFixed(1)+'m';
      document.getElementById('v-out').textContent=S.outN;
      document.getElementById('v-baro').textContent=S.baroAlt.toFixed(1)+' m';
      const tb=document.getElementById('tun-b');if(tb){tb.textContent=S.inTunnel?'TUNNEL':'clear';tb.className='tun-badge '+(S.inTunnel?'tunnel':'clear');}
      const spdKmh=Math.min(120,spdMs*3.6);
      document.getElementById('hud-pos').textContent=S.gtLa.toFixed(4)+'°N '+S.gtLo.toFixed(4)+'°E';
    }catch(e){}
  }

  function triggerOut(){
    if(!S.gnss)return;S.gnss=false;S.outt=0;S.outN++;
    document.getElementById('outage-flash').classList.add('on');
    const ob=document.getElementById('ob');ob.classList.remove('snap');ob.classList.add('on');
    document.getElementById('hdr-dot').classList.add('out');
    document.getElementById('hdr-st').textContent='BLACKOUT';
    document.getElementById('hud-mode').textContent='Dead Reckoning';
    document.getElementById('hud-mode').classList.add('out');
  }
  function reacq(){
    if(S.gnss)return;S.gnss=true;S.snapN++;
    document.getElementById('outage-flash').classList.remove('on');
    const fe=dist(S.ndLa,S.ndLo,S.gtLa,S.gtLo);
    S.ndLa=S.gtLa;S.ndLo=S.gtLo;S.iLa=S.gtLa;S.iLo=S.gtLo;S.unc=.45;
    const ob=document.getElementById('ob');ob.classList.add('snap');ob.textContent='✓ SNAP: -'+fe.toFixed(2)+'m · Reacquired';
    setTimeout(()=>ob.classList.remove('on'),2200);
    document.getElementById('hdr-dot').classList.remove('out');
    document.getElementById('hdr-st').textContent='LOCKED';
    document.getElementById('hud-mode').textContent='NavIC L5 Locked';
    document.getElementById('hud-mode').classList.remove('out');
  }

  function toggleSim(){
    S.run=!S.run;
    const b=document.getElementById('btn-play');
    b.textContent=S.run?'⏸ Pause':'▶ Resume';
  }
  function toggleOutage(){S.gnss?triggerOut():reacq();}
  function switchCity(k,el){
    document.querySelectorAll('.city-btn').forEach(b=>b.classList.remove('active'));
    el.classList.add('active');
    init(k);
  }

  let lt=0;
  function loop(t){requestAnimationFrame(loop);if(t-lt>65){step();lt=t;}}

  if('serviceWorker' in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}

  init('delhi');
  requestAnimationFrame(loop);
</script>
</body>
</html>
