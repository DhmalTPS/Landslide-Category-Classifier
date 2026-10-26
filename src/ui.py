"""Field app interface: one design system shared by the logo, type glyphs and page.

Palette (from the logo; complementary pair, analogous gradients):
  ground / interface   mint #B4F0D3 -> emerald #34B08C -> deep teal #084045
  signal / movement    amber #FFC862 -> coral #FF745E -> rose #E04282   (only for moving mass and warnings)
  neutrals             warm paper #F5F4EF / ink #13201C (light), #0D1311 / #E6EEEA (dark)
Glyphs: same 64x64 grammar as the logo (contour arcs, round caps): mint->teal ground, the moving
mass in the sunset gradient; transparent backgrounds, nothing painted behind the shapes.
The page works offline (system fonts, inline SVG, no external files) and has a demo mode:
  /?demo=ok | uncertain | family | bad | error   renders a sample result (no photo needed).
"""
import json

# ---------------------------------------------------------------- shared SVG definitions
# Glyphs reuse the logo's geometry (64x64 grid, contour arcs, round caps): one visual family.
from logo import C, GROUND, GROUND_AXIS, RADII, SIGNAL, W, _arc_d, _pt, _stops  # noqa: E402

DEFS = ('<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false"><defs>'
        f'<linearGradient id="g-ground" gradientUnits="userSpaceOnUse" x1="{GROUND_AXIS[0][0]}" y1="{GROUND_AXIS[0][1]}" '
        f'x2="{GROUND_AXIS[1][0]}" y2="{GROUND_AXIS[1][1]}">{_stops(GROUND)}</linearGradient>'
        f'<linearGradient id="g-signal" x1="0" y1="0" x2="1" y2="1">{_stops(SIGNAL)}</linearGradient>'
        '</defs></svg>')

_ST = f'fill="none" stroke-width="{W}" stroke-linecap="round" stroke-linejoin="round"'


def _arc(r, a0, a1, grad="g-ground", dx=0.0, dy=0.0, extra=""):
    return f'<path d="{_arc_d(r, a0, a1, dx, dy)}" stroke="url(#{grad})" {_ST}{extra}/>'


def _svg(body, label):
    return f'<svg viewBox="0 0 64 64" role="img" aria-label="{label}">{body}</svg>'


def _hill(outer_break=None):
    """The contour hill; optionally with a gap (degrees) in the outer contour."""
    if outer_break:
        out = _arc(RADII[0], 180, outer_break[1]) + _arc(RADII[0], outer_break[0], 0)
    else:
        out = _arc(RADII[0], 180, 0)
    return out + _arc(RADII[1], 180, 0) + _arc(RADII[2], 180, 0)


_rx, _ry = _pt(RADII[0], 18.0)          # lower end of the broken segment (pivot for rotation)
_e0, _e1 = _pt(RADII[0], 56.0), _pt(RADII[0], 8.0)
GLYPHS = {
    "translational_slide": _svg(_hill((18, 58)) + _arc(RADII[0], 58, 18, "g-signal", 5.6, 6.6), "Translational slide"),
    "rotational_slide": _svg(
        _hill((18, 62)) + _arc(RADII[0], 62, 18, "g-signal", 6.5, 5.0,
                               f' transform="rotate(26 {_rx + 6.5:.2f} {_ry + 5.0:.2f})"'), "Rotational slide"),
    "earth_mud_flow": _svg(
        _hill((8, 56)) +
        f'<path d="M{_e0[0]:.1f} {_e0[1]:.1f}C{_e0[0] + 7:.1f} {_e0[1] + 2:.1f} {_e0[0] + 2:.1f} {_e0[1] + 10:.1f} '
        f'{_e0[0] + 9:.1f} {_e0[1] + 13:.1f}S{_e1[0] + 4:.1f} {C[1] - 1:.1f} {_e1[0] + 6:.1f} {C[1] + 1.5:.1f}" '
        f'stroke="url(#g-signal)" {_ST}/>', "Earthflow or mudflow"),
    "fall_topple": _svg(
        f'<path d="M12 {C[1]}V17M21 {C[1]}V24M30 {C[1]}V33" stroke="url(#g-ground)" {_ST}/>'
        '<circle cx="39" cy="22" r="3.4" fill="url(#g-signal)"/><circle cx="45" cy="31.5" r="2.9" fill="url(#g-signal)"/>'
        f'<circle cx="50.5" cy="{C[1] - 2.2}" r="3.9" fill="url(#g-signal)"/>', "Rock fall or topple"),
    "debris_flow": _svg(
        _arc(22, 90, 0, dx=-21, dy=0) + _arc(13, 90, 0, dx=-21, dy=0) +
        _arc(22, 180, 90, dx=33, dy=0) + _arc(13, 180, 90, dx=33, dy=0) +
        '<circle cx="31" cy="20" r="2.4" fill="url(#g-signal)"/><circle cx="30.4" cy="28.5" r="2.8" fill="url(#g-signal)"/>'
        f'<circle cx="31.4" cy="36.5" r="3.1" fill="url(#g-signal)"/>' +
        f'<path d="M23 {C[1] + 1.5}Q31 {C[1] - 6} 39 {C[1] + 1.5}" stroke="url(#g-signal)" {_ST}/>', "Debris flow"),
    "no_landslide": _svg(_hill(), "No landslide visible"),
    "bad_photo": _svg(
        '<path d="M8 20V12Q8 8 12 8H20M44 8H52Q56 8 56 12V20M56 44V52Q56 56 52 56H44M20 56H12Q8 56 8 52V44" '
        f'stroke="url(#g-ground)" {_ST}/>'
        f'<line x1="21" y1="43" x2="43" y2="21" stroke="url(#g-signal)" {_ST}/>', "Unusable photo"),
}
FAMILY_GLYPH = {"flow": "debris_flow", "slide": "translational_slide"}

# small interface icons (stroke = currentColor)
UI_ICONS = {
    "camera": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8.5A2.5 2.5 0 0 1 6.5 6H8l1.4-2h5.2L16 6h1.5A2.5 2.5 0 0 1 20 8.5v8a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 16.5z"/><circle cx="12" cy="12.5" r="3.6"/></svg>',
    "image": '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><circle cx="9" cy="10" r="1.6"/><path d="M4 17l5-5 4 4 2.5-2.5L20 18"/></svg>',
    "check": '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M8.3 12.3l2.5 2.5 4.9-5.2"/></svg>',
    "alert": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4.2l8.8 15.3H3.2z"/><path d="M12 10v4.3M12 17.1v.1"/></svg>',
    "pos": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 19L13 6l8 13z"/><circle cx="7" cy="19.5" r="1"/></svg>',
    "frame": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16"/></svg>',
    "sun": '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.8"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4"/></svg>',
    "spark": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5l1.9 5.2 5.2 1.9-5.2 1.9L12 17.7l-1.9-5.2-5.2-1.9 5.2-1.9z"/></svg>',
}

CSS = """
:root{--teal-950:#082f2a;--teal-800:#0e4d45;--teal-600:#1f7a68;--teal-400:#4fa587;--sage-300:#8fc9a9;--sage-100:#e4f1e9;
--amber:#f7b74a;--orange:#e04e2b;--ruby:#c2412d;
--bg:#f5f4ef;--surface:#ffffff;--surface-2:#f9f8f4;--ink:#13201c;--ink-2:#3c4b45;--mute:#6c7973;--line:#e4e2da;
--g-ground:linear-gradient(135deg,#3dbb94 0%,#1b8a72 45%,#084045 100%);--g-signal:linear-gradient(135deg,#ffc862 0%,#ff745e 52%,#e04282 100%);
--g-track:linear-gradient(90deg,#b4f0d3,#34b08c);--tint-signal:rgba(255,116,94,.09);--tint-ground:rgba(52,176,140,.09);
--shadow:0 1px 2px rgba(16,33,26,.05),0 10px 30px -12px rgba(16,33,26,.18);--r:18px;
color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0c1210;--surface:#141c19;--surface-2:#18221e;--ink:#e6eeea;--ink-2:#bccac3;--mute:#8b9b94;
--line:#24312c;--g-track:linear-gradient(90deg,#1d4f43,#5fd1a8);--tint-signal:rgba(255,116,94,.11);--tint-ground:rgba(95,209,168,.08);
--shadow:0 1px 2px rgba(0,0,0,.4),0 14px 34px -14px rgba(0,0,0,.6);color-scheme:dark}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;min-height:100vh;color:var(--ink);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
background:radial-gradient(900px 560px at -12% -14%,rgba(180,240,211,.38),transparent 62%),
radial-gradient(760px 460px at 118% -6%,rgba(255,200,98,.16),transparent 60%),
radial-gradient(680px 520px at 110% 108%,rgba(224,66,130,.08),transparent 62%),var(--bg);background-attachment:fixed}
main{max-width:620px;margin:0 auto;padding:20px 16px 44px}
svg{display:block}
.ico{width:20px;height:20px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round;flex:none}
header{display:flex;align-items:center;gap:13px;margin:4px 0 22px}
header .logo svg{width:52px;height:52px}
.brand b{display:block;font-size:1.22rem;letter-spacing:-.015em;font-weight:720}
.brand span{display:block;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin-top:1px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);padding:18px;margin-top:14px}
.kicker{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);font-weight:650}
h2{font-size:1.06rem;margin:.15rem 0 .9rem;letter-spacing:-.01em}
.tips{display:grid;gap:10px;margin:0 0 16px;padding:0;list-style:none}
.tips li{display:flex;gap:10px;align-items:flex-start;color:var(--ink-2);font-size:.94rem}
.tips .ico{color:var(--teal-600);margin-top:2px}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.btn{position:relative;display:flex;align-items:center;justify-content:center;gap:9px;min-height:54px;padding:12px 14px;border-radius:14px;
font-weight:650;font-size:.98rem;cursor:pointer;user-select:none;transition:transform .12s ease,box-shadow .2s ease,filter .2s ease}
.btn.primary{color:#fff;background:var(--g-ground);box-shadow:0 10px 24px -12px rgba(8,64,69,.75),inset 0 1px 0 rgba(255,255,255,.18)}
.btn.primary:hover{filter:brightness(1.06)}.btn:active{transform:translateY(1px) scale(.99)}
.btn.ghost{color:var(--teal-600);background:var(--surface);border:1.5px solid rgba(31,122,104,.4)}
@media (prefers-color-scheme:dark){.btn.ghost{color:var(--sage-300);border-color:rgba(143,201,169,.45)}.tips .ico{color:var(--sage-300)}}
.btn input{position:absolute;inset:0;opacity:0;cursor:pointer}
.btn:focus-within{outline:3px solid rgba(255,200,98,.75);outline-offset:2px}
figure{position:relative;margin:14px 0 0;display:none}
figure img{width:100%;max-height:62vh;object-fit:cover;border-radius:var(--r);box-shadow:var(--shadow);display:block}
.chip{position:absolute;left:12px;top:12px;display:inline-flex;align-items:center;gap:7px;padding:6px 11px 6px 9px;border-radius:999px;
font-size:.8rem;font-weight:650;color:#fff;background:rgba(10,24,20,.62);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--sage-300)}
.chip.signal .dot{background:var(--amber)}.chip.bad .dot{background:#ff8a6b}
.spin{width:13px;height:13px;border-radius:50%;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.result{padding:20px}
.rhead{display:flex;gap:15px;align-items:center}
.glyph{width:68px;height:68px;flex:none;border-radius:20px;display:grid;place-items:center;background:var(--tint-ground)}
.glyph.signal{background:var(--tint-signal)}.glyph svg{width:52px;height:52px}
.title{font-size:1.42rem;font-weight:740;letter-spacing:-.02em;line-height:1.15;margin:.12rem 0 .18rem}
.hungr{font-size:.84rem;color:var(--mute)}
.meter{margin-top:16px}
.meter .row{display:flex;justify-content:space-between;align-items:baseline;font-size:.85rem;color:var(--mute)}
.meter .val{font-size:1.5rem;font-weight:750;color:var(--ink);font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.track{height:10px;border-radius:99px;background:var(--line);overflow:hidden;margin-top:6px}
.fill{height:100%;border-radius:99px;background:var(--g-ground);transform-origin:left;animation:grow .7s cubic-bezier(.2,.8,.2,1)}
.fill.signal{background:var(--g-signal)}
@keyframes grow{from{transform:scaleX(0)}}
.note{margin:14px 0 0;padding:12px 14px;border-radius:12px;background:var(--surface-2);color:var(--ink-2);font-size:.93rem;border:1px solid var(--line)}
.probs{display:grid;gap:9px;margin-top:4px}
.prob{display:grid;grid-template-columns:28px 1fr 46px;gap:10px;align-items:center}
.prob svg{width:28px;height:28px}
.prob .nm{font-size:.9rem;color:var(--ink-2);display:flex;flex-direction:column;gap:5px}
.prob .tr{height:7px;border-radius:99px;background:var(--line);overflow:hidden}
.prob .fl{height:100%;border-radius:99px;background:var(--g-track);animation:grow .7s cubic-bezier(.2,.8,.2,1)}
.prob.top .fl{background:var(--g-ground)}.prob.top .nm{color:var(--ink);font-weight:650}
.prob .pc{text-align:right;font-variant-numeric:tabular-nums;font-size:.88rem;color:var(--mute)}
.prob.top .pc{color:var(--ink);font-weight:700}
.list{list-style:none;padding:0;margin:0;display:grid;gap:10px}
.list li{display:flex;gap:10px;align-items:flex-start;font-size:.95rem;color:var(--ink-2)}
.list .ico{color:var(--teal-600);margin-top:1px}
@media (prefers-color-scheme:dark){.list .ico{color:var(--sage-300)}}
.safety{background:linear-gradient(var(--tint-signal),var(--tint-signal)),var(--surface);position:relative;overflow:hidden}
.safety::before{content:"";position:absolute;inset:0 auto 0 0;width:5px;background:var(--g-signal)}
.safety h2{display:flex;gap:8px;align-items:center}.safety .ico{color:var(--orange)}
.safety p{margin:0;color:var(--ink-2)}
.bad::before{content:"";position:absolute;inset:0 auto 0 0;width:5px;background:var(--g-signal)}
.bad{position:relative;overflow:hidden}
ol.steps{margin:.4rem 0 0;padding-left:1.2rem;color:var(--ink-2)}ol.steps li{margin:.25rem 0}
.skel{height:14px;border-radius:8px;background:linear-gradient(90deg,var(--line),var(--surface-2),var(--line));background-size:200% 100%;animation:sh 1.2s ease-in-out infinite;margin:10px 0}
@keyframes sh{to{background-position:-200% 0}}
.foot{margin-top:22px;font-size:.8rem;color:var(--mute);text-align:center;line-height:1.55}
.meta{display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:12px;font-size:.76rem;color:var(--mute)}
.meta span{padding:3px 9px;border-radius:99px;background:var(--surface);border:1px solid var(--line)}
@media (max-width:380px){.btns{grid-template-columns:1fr}.title{font-size:1.25rem}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""

JS = r"""
const D=__DATA__, G=D.glyphs, out=document.getElementById('out'), fig=document.getElementById('fig'),
 prev=document.getElementById('prev'), chip=document.getElementById('chip');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=p=>Math.round(p*100)+'%';
const ico=n=>D.icons[n].replace('<svg','<svg class="ico"');
function setChip(kind,text,busy){chip.className='chip '+kind;chip.innerHTML=(busy?'<span class="spin"></span>':'<span class="dot"></span>')+esc(text)}
async function shrink(file){
 try{const bmp=await createImageBitmap(file,{imageOrientation:'from-image'});
  const s=Math.min(1,1280/Math.max(bmp.width,bmp.height));const c=document.createElement('canvas');
  c.width=Math.round(bmp.width*s);c.height=Math.round(bmp.height*s);c.getContext('2d').drawImage(bmp,0,0,c.width,c.height);
  return await new Promise(r=>c.toBlob(r,'image/jpeg',0.9));}catch(e){return file;}}
function probs(r){const top=r.ranked[0][0];
 return r.ranked.map(([c,p])=>`<div class="prob ${c===top?'top':''}">${G[c]}<div class="nm">${esc(D.names[c])}<div class="tr"><div class="fl" style="width:${(p*100).toFixed(1)}%"></div></div></div><div class="pc">${pct(p)}</div></div>`).join('')}
function dims(d){
 if(!d.available)return `<section class="card"><div class="kicker">Dimensions</div><p class="hungr" style="margin:0">${esc(d.reason)}</p></section>`;
 const num=v=>v>=100?Math.round(v).toLocaleString():v, row=(k,v)=>`<div class="drow"><span>${esc(k)}</span><b>${v}</b></div>`;
 let h=`<section class="card"><div class="kicker">Estimated dimensions · depth trust: ${esc(d.depth_trust)}</div>`;
 if(d.overlay)h+=`<img class="dimg" src="${d.overlay}" alt="Measured area outlined">`;
 h+=d.measures.map(m=>row(m.name,`${num(m.value)} ${m.unit} <small>(90%: ${num(m.range[0])}–${num(m.range[1])})</small>`)).join('');
 if(d.size_class){const sp=d.size_class_prob||{};h+=row('Size class (Fell 1994)',esc(d.size_class)+(Object.keys(sp).length>1?' <small>('+Object.entries(sp).map(([k,v])=>esc(k)+' '+Math.round(v*100)+'%').join(', ')+')</small>':''));}
 if(d.block_count!=null)h+=row('Blocks counted',d.block_count+(d.block_size_class?' · largest: '+esc(d.block_size_class):''));
 if(d.grains)h+=d.grains.reliable?row('Grain size D16 / D50 / D84',`${d.grains.D16_m} / ${d.grains.D50_m} / ${d.grains.D84_m} m <small>(D50 90%: ${d.grains.D50_90ci_m.join('–')})</small>`):row('Grain size','not resolvable in this photo (take a closer photo of the deposit)');
 h+=row('Slope angle',d.slope_angle_deg+'°'+(d.tilt==='phone sensor'?' (phone tilt)':''));
 if(d.travel_angle_deg!=null)h+=row('Travel angle (H/L)',`${d.travel_angle_deg}° (${d.H_over_L})`);
 h+=row('Length / width',d.length_width_ratio)+row('Perimeter',num(d.perimeter_m)+' m')+row('Camera to slope',d.distance_m+' m');
 if(d.alternative){const a=d.alternative;h+=`<p class="hungr" style="margin:8px 0 0">If it is <b>${esc(D.names[a.label]||a.label)}</b> (${Math.round(a.prob*100)}%): ${a.measures.slice(0,3).map(m=>esc(m.name)+' '+num(m.value)+' '+m.unit).join(' · ')}</p>`;}
 const notes=[...d.notes,...d.depth_issues.map(x=>'Depth check: '+x)];
 h+=`<p class="hungr" style="margin:8px 0 0">Scale: ${esc(d.scale_source)}. ${notes.map(esc).join(' ')}</p></section>`;
 return h}
function render(r){
 const n=D.names;let h='';
 if(r.status==='bad_photo'){
  setChip('bad','Retake photo');
  h=`<section class="card result bad" aria-live="polite"><div class="rhead"><div class="glyph signal">${G.bad_photo}</div><div>
   <div class="kicker">Photo check</div><div class="title">Retake the photo</div>
   <div class="hungr">This does not look like a ground photo of a slope (${pct(r.usable)} usable)</div></div></div>
   <ol class="steps">${D.tips.map(t=>`<li>${esc(t)}</li>`).join('')}</ol>
   <p class="note">Best guess anyway: <b>${esc(n[r.label])}</b> (${pct(r.confidence)}). Not reliable for this photo.</p></section>`;
 }else{
  const i=D.info[r.label],unsure=r.status==='uncertain',fam=unsure&&r.family?D.families[r.family]:null;
  const gname=fam?D.familyGlyph[r.family]:r.label, sig=unsure&&!fam;
  setChip(sig?'signal':'', fam?'Landslide family':unsure?'Uncertain':'Landslide type');
  const kicker=fam?'Landslide family':unsure?'Uncertain — most likely':'Landslide type';
  const conf=fam?r.ranked[0][1]+r.ranked[1][1]:r.confidence;
  let note='';
  if(fam) note=`<p class="note">Clearly a ${esc(r.family)}; the exact type is close between <b>${esc(n[r.ranked[0][0]])}</b> and <b>${esc(n[r.ranked[1][0]])}</b>. Check the points below for both.</p>`;
  else if(unsure) note=`<p class="note">Not confident. Second choice: <b>${esc(n[r.ranked[1][0]])}</b>. Take another photo with the whole slope in view, or check the points below for both types.</p>`;
  h=`<section class="card result" aria-live="polite"><div class="rhead"><div class="glyph ${gname==='no_landslide'?'':'signal'}">${G[gname]}</div><div>
    <div class="kicker">${kicker}</div><div class="title">${esc(fam||i.title)}</div>
    <div class="hungr">${fam?'':'Hungr et al. 2014 · '+esc(i.hungr)}</div></div></div>
    <div class="meter"><div class="row"><span>Confidence${fam?' (family)':''}</span><span class="val">${pct(conf)}</span></div>
    <div class="track"><div class="fill ${sig?'signal':''}" style="width:${(conf*100).toFixed(1)}%"></div></div></div>${note}</section>
   <section class="card"><div class="kicker">All types</div><div class="probs" style="margin-top:12px">${probs(r)}</div></section>
   <section class="card"><div class="kicker">Check on site</div><h2>${esc(i.title)}</h2><ul class="list">${i.check.map(t=>`<li>${ico('check')}<span>${esc(t)}</span></li>`).join('')}</ul></section>
   <section class="card safety"><h2>${ico('alert')}Safety</h2><p>${esc(i.safety)}</p></section>`;
 }
 if(r.vlm){h+=`<section class="card"><div class="kicker">Second opinion · Qwen2.5-VL (zero-shot)</div><h2 style="display:flex;gap:8px;align-items:center">${ico('spark')}${esc(n[r.vlm.label]||r.vlm.label)} · ${pct(r.vlm.confidence)}</h2><p class="hungr" style="margin:0">${esc(r.vlm.description)}</p></section>`;}
 if(r.dimensions)h+=dims(r.dimensions);
 h+=`<div class="meta"><span>${esc(r.model)}</span><span>${r.ms} ms</span><span>confident ≥ ${pct(D.conf)}</span></div>`;
 out.innerHTML=h;}
function loading(){setChip('','Analysing…',true);
 out.innerHTML='<section class="card result" aria-busy="true"><div class="kicker">Analysing photo</div><div class="skel" style="width:60%;height:22px"></div><div class="skel" style="width:85%"></div><div class="skel" style="width:40%"></div></section>';}
function fail(msg){setChip('bad','Error');
 out.innerHTML=`<section class="card bad" role="alert"><div class="kicker">Something went wrong</div><h2>${esc(msg)}</h2><p class="hungr" style="margin:0">Check the connection to the laptop running the app, then try again.</p></section>`;}
// phone tilt (gravity in camera axes: x right, y down, z forward) for true slope angle and height
let tilt=null;
function onTilt(e){if(e.beta==null)return;const b=(e.beta-90)*Math.PI/180,g=(e.gamma||0)*Math.PI/180;
 tilt=[Math.sin(g)*Math.cos(b),Math.cos(g)*Math.cos(b),-Math.sin(b)].map(v=>v.toFixed(3)).join(',')}
async function askTilt(){try{if(window.DeviceOrientationEvent&&DeviceOrientationEvent.requestPermission)
 await DeviceOrientationEvent.requestPermission()}catch(_){}window.addEventListener('deviceorientation',onTilt)}
document.querySelectorAll('.btn').forEach(b=>b.addEventListener('click',askTilt,{once:true}));
async function go(e){const fromCam=e.target.id==='cam';const f=e.target.files[0];if(!f)return;e.target.value='';
 prev.src=URL.createObjectURL(f);fig.style.display='block';loading();fig.scrollIntoView({behavior:'smooth',block:'start'});
 try{const body=await shrink(f);
  const q=new URLSearchParams(location.search),dv=parseFloat(document.getElementById('dist').value);
  if(dv>0)q.set('d',dv);
  if(fromCam&&tilt)q.set('g',tilt);   // gallery photos were not taken at the current tilt
  const res=await fetch('predict?'+q,{method:'POST',body,headers:{'Content-Type':body.type||'application/octet-stream'}});
  const r=await res.json().catch(()=>({error:'Unexpected reply from the server ('+res.status+')'}));
  if(!res.ok)throw new Error(r.error||res.status);render(r);
 }catch(err){fail(err.message)}}
document.getElementById('cam').onchange=go;document.getElementById('gal').onchange=go;
const demo=new URLSearchParams(location.search).get('demo');
if(demo){const S=D.samples[demo]||D.samples.ok;   // sample result, no photo (presentations, UI checks)
 if(demo==='error')fail('could not read the image (use a JPEG or PNG photo under 40 megapixels)');else render(S);}
"""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Landslide Type Check</title>
<link rel="icon" type="image/svg+xml" href="favicon.svg">
<link rel="icon" type="image/png" sizes="48x48" href="favicon.ico">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<meta name="theme-color" content="#0e4d45">
<style>__CSS__.drow{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid rgba(127,127,127,.18);font-size:14px}.drow b{text-align:right;font-weight:600}.drow small{opacity:.65;font-weight:400}.dimg{width:100%;border-radius:10px;margin:10px 0}.dist{display:block;margin-top:14px;font-size:13px;opacity:.85}.dist input{display:block;width:100%;margin-top:6px;padding:10px;border-radius:10px;border:1px solid rgba(127,127,127,.4);font:inherit;background:transparent;color:inherit}
</style></head><body>__DEFS__<main>
<header><div class="logo">__LOGO__</div><div class="brand"><b>Landslide Type Check</b><span>Field classifier &middot; Hungr 2014</span></div></header>
<section class="card"><div class="kicker">New observation</div><h2>Photograph the slope from below</h2>
<ul class="tips">
<li>__I_pos__<span>Stand at the foot of the slope or across the road, facing up at it.</span></li>
<li>__I_frame__<span>Fit the whole slope: the top (scarp or source) and the bottom (deposit).</span></li>
<li>__I_sun__<span>Hold the phone level, in daylight. One slope per photo.</span></li></ul>
<div class="btns">
<label class="btn primary">__I_camera__Take photo<input id="cam" type="file" accept="image/*" capture="environment"></label>
<label class="btn ghost">__I_image__From gallery<input id="gal" type="file" accept="image/*"></label>
</div>
<label class="dist">Distance from you to the slope, metres (optional, makes sizes more accurate)
<input id="dist" type="number" min="1" max="3000" step="any" inputmode="decimal" placeholder="e.g. 40"></label></section>
<figure id="fig"><img id="prev" alt="Your photo"><div id="chip" class="chip"></div></figure>
<div id="out"></div>
<p class="foot">Decision support only &mdash; confirm with a geologist or geotechnical engineer before acting.<br>
Classification after Hungr, Leroueil &amp; Picarelli (2014).</p>
</main><script>__JS__</script></body></html>"""


def build_page(logo_svg, info, names, tips, confident_min):
    """The complete HTML page (inline CSS/JS/SVG; no external requests)."""
    top = lambda c, p: [[c, p]] + [[o, round((1 - p) / 5, 4)] for o in names if o != c]  # noqa: E731
    samples = {
        "ok": {"status": "ok", "usable": .94, "label": "translational_slide", "family": None, "confidence": .87,
               "dimensions": {"available": True, "label": "translational_slide", "depth_trust": "medium",
                              "measures": [{"key": "width", "name": "Width (across slope)", "value": 18.4, "unit": "m", "range": [12.0, 24.8]},
                                           {"key": "length", "name": "Length (down slope)", "value": 31.2, "unit": "m", "range": [20.3, 42.1]},
                                           {"key": "area", "name": "Area of the scar", "value": 452.0, "unit": "m²", "range": [203.4, 813.6]},
                                           {"key": "volume", "name": "Volume (Guzzetti 2009 area-volume law)", "unit": "m³", "value": 530.0, "range": [167.0, 1240.0]}],
                              "size_class": "very small", "slope_angle_deg": 34, "travel_angle_deg": 31, "H_over_L": 0.6,
                              "length_width_ratio": 1.7, "perimeter_m": 88.0, "distance_m": 42.0, "tilt": "assumed upright",
                              "grains": {"reliable": False}, "scale_source": "demo values", "notes": [], "depth_issues": []},
               "ranked": [["translational_slide", .87], ["debris_flow", .05], ["fall_topple", .04],
                          ["rotational_slide", .02], ["earth_mud_flow", .01], ["no_landslide", .01]],
               "model": "demo", "ms": 0},
        "uncertain": {"status": "uncertain", "usable": .9, "label": "fall_topple", "family": None, "confidence": .46,
                      "ranked": [["fall_topple", .46], ["no_landslide", .31], ["translational_slide", .12],
                                 ["rotational_slide", .05], ["debris_flow", .04], ["earth_mud_flow", .02]],
                      "model": "demo", "ms": 0},
        "family": {"status": "uncertain", "usable": .92, "label": "debris_flow", "family": "flow", "confidence": .48,
                   "ranked": [["debris_flow", .48], ["earth_mud_flow", .39], ["translational_slide", .07],
                              ["rotational_slide", .03], ["fall_topple", .02], ["no_landslide", .01]],
                   "model": "demo", "ms": 0},
        "bad": {"status": "bad_photo", "usable": .12, "label": "no_landslide", "family": None, "confidence": .5,
                "ranked": top("no_landslide", .5), "model": "demo", "ms": 0},
    }
    data = {"glyphs": GLYPHS, "icons": UI_ICONS, "info": info, "names": names, "tips": tips,
            "families": {"flow": "Flow-type landslide", "slide": "Slide (rotational or translational)"},
            "familyGlyph": FAMILY_GLYPH, "conf": confident_min, "samples": samples}
    page = PAGE.replace("__CSS__", CSS).replace("__DEFS__", DEFS).replace("__LOGO__", logo_svg)
    for k, v in UI_ICONS.items():
        page = page.replace(f"__I_{k}__", v.replace("<svg", '<svg class="ico"'))
    # JSON inside <script>: escape "</" so no string can close the script element
    return page.replace("__JS__", JS.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
