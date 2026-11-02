"""Guidance layer of the field app: it leads a junior engineer from "where do I stand?" to "what do I do next?".

Each stage of the journey has its own aid:

  before   first-run tour (4 illustrated cards, can be replayed), a per-session safety check,
           single-photo or 4-shot site-survey mode with a shot list, a distance helper (pace it out)
  capture  live viewfinder (needs https or localhost): framing bands for scarp and toe, rule-of-thirds
           grid, a level line and aim angle from the phone sensors, live light, sharpness and steadiness,
           a ready ring, haptic and optional voice cues. Without https the phone's own camera is used.
  check    an instant on-phone photo check before upload: light, shadows, highlights, haze, blur,
           resolution, level and aim. Thresholds come from the training photos
           (tools/calibrate_photo_check.py -> src/photo_check.json); it only warns and never blocks.
  wait     upload progress, then stages paced by the measured server time, field tips, cancel
  result   a "next steps" card (retake, add the distance and re-measure, close-up of the deposit, next
           survey shot), on-site signs that can be ticked (top-2 types compared when the model is
           unsure), a glossary of the numbers, spoken summary
  always   help sheet (photo guide, safety, reading results, glossary, limits, settings), recent
           observations, offline banner, retry, sun mode (high contrast), voice, keyboard and
           screen-reader support

Settings and history stay in this browser (localStorage); nothing extra is sent to the server.
"""
import json
from pathlib import Path

from guide_data import HI, QUESTIONS, QUIZ

_CHECK = Path(__file__).with_name("photo_check.json")
_DEFAULT = {"thresholds": {"lum_min": 71.1, "lum_max": 169.7, "dark_max": 0.157, "bright_max": 0.283,
                           "contrast_min": 28.9, "sharp_min": 414.2}, "percentile_q": 0.75}


def photo_check():
    try:
        return json.loads(_CHECK.read_text())
    except (OSError, ValueError):
        return _DEFAULT


GLOSSARY = {
    "Scarp / crown": "The steep break at the top where the ground tore away. Fresh scarps are bare and sharp-edged.",
    "Toe / deposit": "The bottom of the landslide, where the moved material piled up.",
    "Confidence": "How sure the model is about the type (0-100%). Below the 'confident' mark, the app says Uncertain.",
    "Uncertain / family": "Unsure about the exact type. 'Family' means it is sure of the group (flow or slide), not which member.",
    "90% range": "The size lies in this range with about 90% probability, given the photo. Wide range = rough estimate.",
    "Depth trust": "How well the 3-D depth checks agreed (high / medium / low). Low: treat sizes as order-of-magnitude.",
    "Grain size D16/D50/D84": "Block or boulder sizes: 16%, 50% and 84% of the counted pieces are smaller than these.",
    "Travel angle (H/L)": "Drop height over run-out length. A low angle means the material travelled far (mobile, dangerous).",
    "Size class (Fell 1994)": "Volume class from 'extremely small' (<500 m3) to 'very large', used in hazard reports.",
    "Slope angle": "Steepness of the ground surface in the outlined area, in degrees.",
}

SHOTS = [
    {"id": "overview", "t": "Whole slope", "frame": "full",
     "d": "From below, 20-80 m away. Get the top (scarp) and the bottom (deposit) in one frame."},
    {"id": "source", "t": "Source / scarp", "frame": "top",
     "d": "Zoom in (do not climb) so the upper break, cracks or bare face fill the frame."},
    {"id": "deposit", "t": "Deposit close-up", "frame": "bottom",
     "d": "5-15 m from the toe, boulders and soil filling the frame. This gives the grain size."},
    {"id": "scale", "t": "Something for scale", "frame": "full",
     "d": "The whole slope again with a person, car or pole near the toe. This fixes the metres."},
]

SAFETY = [
    "I am below or beside the slope, not on it or on the deposit.",
    "Nothing hangs over me and I am outside the path of falling rocks.",
    "No fresh cracking, rumbling, running muddy water or moving debris right now.",
    "On a road: I am out of traffic, or someone is watching for me.",
]

FIELD_TIPS = [
    "Cracks above the scarp mean the slope may still be moving: keep clear of the crown.",
    "Photos from two or three spots give a more reliable type than one photo.",
    "Typing the distance to the slope makes the sizes much more accurate.",
    "A person or car in the photo lets the app check its own scale.",
    "Morning or late-afternoon side light shows cracks and steps best.",
    "Wipe the lens: dust and rain drops make photos soft and hazy.",
    "Note the date, location and weather with your observation.",
]

# simple illustrations for the tour (stroke = currentColor, accents use the app gradients)
_IL = 'viewBox="0 0 240 130" class="il" aria-hidden="true"'
ILLUS = [
    f'<svg {_IL}><path d="M10 112H230" stroke="currentColor" stroke-width="2" opacity=".4"/>'
    '<path d="M120 112L188 26L230 26V112Z" fill="url(#g-ground)" opacity=".85"/>'
    '<path d="M150 74Q164 70 172 92L180 112" fill="none" stroke="url(#g-signal)" stroke-width="5" stroke-linecap="round"/>'
    '<circle cx="46" cy="84" r="6" fill="currentColor"/><path d="M46 90V104M46 104L40 112M46 104L52 112M46 95L56 92" '
    'stroke="currentColor" stroke-width="3" stroke-linecap="round" fill="none"/>'
    '<path d="M58 91L160 30M58 91L214 96" stroke="url(#g-signal)" stroke-width="1.6" stroke-dasharray="4 4" fill="none"/>'
    '<text x="22" y="126" font-size="10" fill="currentColor">you, below</text><text x="150" y="20" font-size="10" fill="currentColor">whole slope in view</text></svg>',
    f'<svg {_IL}><rect x="18" y="14" width="88" height="104" rx="10" fill="none" stroke="currentColor" stroke-width="2.5"/>'
    '<path d="M24 100L60 36L100 36V112H24Z" fill="url(#g-ground)" opacity=".85"/><path d="M24 28H100M24 100H100" stroke="url(#g-signal)" stroke-width="2" stroke-dasharray="5 4"/>'
    '<text x="38" y="26" font-size="9" fill="currentColor">scarp</text><text x="38" y="113" font-size="9" fill="currentColor">toe</text>'
    '<path d="M44 124l6 5 10-10" stroke="#34b08c" stroke-width="3" fill="none" stroke-linecap="round"/>'
    '<rect x="134" y="14" width="88" height="104" rx="10" fill="none" stroke="currentColor" stroke-width="2.5"/>'
    '<path d="M140 60L222 40V112H140Z" fill="url(#g-ground)" opacity=".45"/><circle cx="176" cy="80" r="16" fill="none" stroke="currentColor" stroke-width="1.5" opacity=".5"/>'
    '<path d="M168 120l12 8M180 120l-12 8" stroke="#e04e2b" stroke-width="3" stroke-linecap="round"/></svg>',
    f'<svg {_IL}><circle cx="52" cy="40" r="14" fill="url(#g-signal)"/><path d="M52 16V8M52 72V64M28 40H20M84 40H76" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
    '<rect x="120" y="24" width="56" height="92" rx="9" fill="none" stroke="currentColor" stroke-width="2.5"/>'
    '<path d="M104 70H192" stroke="url(#g-signal)" stroke-width="2.5"/><circle cx="148" cy="70" r="5" fill="#34b08c"/>'
    '<text x="196" y="74" font-size="10" fill="currentColor">level</text><text x="20" y="98" font-size="10" fill="currentColor">daylight, sun behind</text>'
    '<text x="20" y="112" font-size="10" fill="currentColor">two hands, hold still</text></svg>',
    f'<svg {_IL}><rect x="20" y="18" width="200" height="12" rx="6" fill="currentColor" opacity=".12"/><rect x="20" y="18" width="160" height="12" rx="6" fill="url(#g-ground)"/>'
    '<text x="20" y="48" font-size="10" fill="currentColor">confidence: how sure the type is</text>'
    '<path d="M40 86H200" stroke="currentColor" stroke-width="2" opacity=".35"/><path d="M70 86H170" stroke="url(#g-signal)" stroke-width="6" stroke-linecap="round"/>'
    '<circle cx="112" cy="86" r="6" fill="currentColor"/><text x="20" y="112" font-size="10" fill="currentColor">size and its 90% range (wide = rough)</text></svg>',
]
TOUR = [
    ("What this app does", "Photograph a landslide from the ground. The app names its type, shows the on-site signs "
     "to confirm, and estimates its size. It supports your judgement; it does not replace a geologist."),
    ("Stand below, see it all", "Stand at the foot of the slope or across the road, 20-80 m away, never on the slope "
     "or the deposit. Face up at it."),
    ("Frame top to bottom", "Fit the whole failure: the scarp at the top and the toe at the bottom. Not just a close-up "
     "of soil, not mostly sky."),
    ("Light, level, steady", "Daylight with the sun behind or beside you. Hold the phone level with two hands. The live "
     "guide turns green when the shot is good."),
    ("Read the answer", "Confidence says how sure the type is. Sizes come with a 90% range. Below each result, "
     "'Next steps' tells you what extra photo or detail would make it better."),
]

CSS = """
.iconbtn{margin-left:auto;width:42px;height:42px;border-radius:50%;border:1.5px solid var(--line);background:var(--surface);color:var(--ink);font:700 1.05rem/1 inherit;cursor:pointer;box-shadow:var(--shadow)}
.journey{display:flex;gap:6px;margin:2px 0 14px;padding:0;list-style:none;counter-reset:j}
.journey li{flex:1;font-size:.7rem;color:var(--mute);text-align:center;counter-increment:j}
.journey li::before{content:counter(j);display:grid;place-items:center;width:24px;height:24px;margin:0 auto 4px;border-radius:50%;border:1.5px solid var(--line);font-weight:700;background:var(--surface)}
.journey li.done::before{content:"\\2713";background:var(--tint-ground);border-color:var(--teal-400);color:var(--teal-600)}
.journey li.on{color:var(--ink);font-weight:650}.journey li.on::before{background:var(--g-ground);color:#fff;border-color:transparent}
.seg{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:4px;padding:4px;border-radius:12px;background:var(--surface-2);border:1px solid var(--line);margin:0 0 14px}
.seg button{border:0;border-radius:9px;padding:9px;font:600 .88rem inherit;background:transparent;color:var(--ink-2);cursor:pointer}
.seg button[aria-pressed=true]{background:var(--surface);color:var(--ink);box-shadow:var(--shadow)}
.shots{list-style:none;margin:0 0 14px;padding:0;display:grid;gap:8px}
.shots li{display:grid;grid-template-columns:26px 1fr;gap:10px;padding:10px 12px;border-radius:12px;border:1px solid var(--line);font-size:.88rem;color:var(--ink-2);cursor:pointer}
.shots li b{display:block;color:var(--ink);font-size:.93rem}
.shots li .n{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;font-weight:700;border:1.5px solid var(--line)}
.shots li.cur{border-color:var(--teal-400);background:var(--tint-ground)}.shots li.cur .n{background:var(--g-ground);color:#fff;border:0}
.shots li.done .n{background:var(--tint-ground);color:var(--teal-600);border-color:var(--teal-400)}
.pace{margin-top:8px;font-size:.85rem;color:var(--ink-2)}.pace summary{cursor:pointer;color:var(--teal-600);font-weight:600}
.pace .pr{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:8px}
.pace input{width:76px;padding:8px;border-radius:9px;border:1px solid var(--line);font:inherit;background:transparent;color:inherit}
.minibtn{border:1.5px solid rgba(31,122,104,.45);background:var(--surface);color:var(--teal-600);border-radius:10px;padding:8px 12px;font:650 .86rem inherit;cursor:pointer}
.minibtn.pri{background:var(--g-ground);color:#fff;border-color:transparent}
.fieldmsg{font-size:.8rem;color:var(--orange);margin-top:4px;min-height:1em}
.modal{position:fixed;inset:0;z-index:50;display:grid;place-items:end center;background:rgba(6,16,13,.55);backdrop-filter:blur(3px);-webkit-backdrop-filter:blur(3px)}
.modal[hidden]{display:none}
.sheet{width:min(620px,100%);max-height:92vh;overflow:auto;background:var(--surface);color:var(--ink);border-radius:22px 22px 0 0;padding:20px 18px calc(20px + env(safe-area-inset-bottom));box-shadow:var(--shadow)}
@media (min-width:700px){.modal{place-items:center}.sheet{border-radius:22px}}
.sheet h2{margin-top:0}.sheet h3{font-size:.95rem;margin:18px 0 6px}.sheet p,.sheet li{color:var(--ink-2);font-size:.92rem}
.sheet .row{display:flex;gap:8px;justify-content:space-between;align-items:center;margin-top:16px}
.il{width:100%;height:auto;max-height:150px;color:var(--ink-2);margin:4px 0 8px}
.dots{display:flex;gap:6px}.dots i{width:7px;height:7px;border-radius:50%;background:var(--line)}.dots i.on{background:var(--teal-600)}
.chk{display:flex;gap:10px;align-items:flex-start;padding:10px;border-radius:12px;border:1px solid var(--line);margin:8px 0;cursor:pointer;font-size:.92rem;color:var(--ink-2)}
.chk input{width:22px;height:22px;margin:0;flex:none;accent-color:#1f7a68}
.qc li{display:grid;grid-template-columns:22px 1fr;gap:10px;padding:7px 0;border-bottom:1px solid var(--line);font-size:.92rem;color:var(--ink-2)}
.qc{list-style:none;margin:8px 0 12px;padding:0}.qc b{color:var(--ink)}
.qc .ok{color:var(--teal-600)}.qc .warn{color:var(--orange)}
.prog{height:8px;border-radius:99px;background:var(--line);overflow:hidden;margin:12px 0 6px}
.prog i{display:block;height:100%;width:0;background:var(--g-ground);transition:width .4s linear}
.stages{list-style:none;padding:0;margin:10px 0 0;display:grid;gap:6px;font-size:.88rem;color:var(--mute)}
.stages li::before{content:"\\25CB  "}.stages li.done{color:var(--ink-2)}.stages li.done::before{content:"\\2713  ";color:var(--teal-600)}
.stages li.on{color:var(--ink);font-weight:650}.stages li.on::before{content:"\\25CF  ";color:var(--amber)}
.tipline{margin-top:12px;font-size:.85rem;color:var(--ink-2);padding:10px 12px;border-radius:12px;background:var(--surface-2);border:1px solid var(--line);min-height:3em}
.acts{list-style:none;padding:0;margin:8px 0 0;display:grid;gap:10px}
.acts li{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;font-size:.92rem;color:var(--ink-2);padding:10px 12px;border-radius:12px;background:var(--surface-2);border:1px solid var(--line)}
.acts li b{color:var(--ink)}
.tick{cursor:pointer;border-radius:10px;padding:4px;margin:-4px}.tick[aria-checked=true] .ico{color:#fff;background:#1f7a68;border-radius:50%}
.tick[aria-checked=false] .ico{opacity:.45}.tally{margin-top:12px;font-size:.88rem;color:var(--ink-2)}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}.cmp div{padding:10px;border-radius:12px;border:1px solid var(--line);text-align:center;font-size:.85rem}
.cmp b{display:block;font-size:1.3rem}.cmp .lead{border-color:var(--teal-400);background:var(--tint-ground)}
.gloss dt{font-weight:650;color:var(--ink);margin-top:8px;font-size:.9rem}.gloss dd{margin:2px 0 0;color:var(--ink-2);font-size:.88rem}
details.what{margin-top:10px;font-size:.88rem}details.what summary{cursor:pointer;color:var(--teal-600);font-weight:600}
.hist{list-style:none;padding:0;margin:8px 0 0;display:grid;gap:8px}
.hist li{display:grid;grid-template-columns:48px 1fr auto;gap:10px;align-items:center;font-size:.86rem;color:var(--ink-2)}
.hist img{width:48px;height:48px;object-fit:cover;border-radius:10px}.hist b{color:var(--ink)}
.netbar{position:sticky;top:0;z-index:40;margin:-8px -4px 10px;padding:9px 12px;border-radius:12px;background:#7a2b1d;color:#fff;font-size:.86rem;text-align:center}
.netbar[hidden]{display:none}
.switch{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--line);font-size:.92rem}
.switch input{width:22px;height:22px;accent-color:#1f7a68}
.vf{position:fixed;inset:0;z-index:60;background:#000;color:#fff;overflow:hidden}.vf[hidden]{display:none}
.vf video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.vfg{position:absolute;inset:7% 5% 20%;pointer-events:none}
.vfg i{position:absolute;background:rgba(255,255,255,.28)}.vfg .h1,.vfg .h2{left:0;right:0;height:1px}.vfg .h1{top:33.3%}.vfg .h2{top:66.6%}
.vfg .v1,.vfg .v2{top:0;bottom:0;width:1px}.vfg .v1{left:33.3%}.vfg .v2{left:66.6%}
.band{position:absolute;left:0;right:0;border:2px dashed rgba(255,200,98,.85);border-radius:12px;font-size:.75rem;font-weight:650;padding:4px 8px;color:#ffd58a;text-shadow:0 1px 2px #000}
.vfg .top{top:0;height:18%}.vfg .bot{bottom:0;height:18%;display:flex;align-items:flex-end}
.vfg[data-frame=top] .bot,.vfg[data-frame=bottom] .top{display:none}.vfg[data-frame=top] .top{height:70%}.vfg[data-frame=bottom] .bot{height:70%}
.lvl{position:absolute;left:50%;top:50%;width:46%;height:0;border-top:2px solid #ff745e;transform:translate(-50%,-50%);transition:transform .15s linear,border-color .2s}
.lvl.ok{border-color:#5fd1a8}.lvl::before,.lvl::after{content:"";position:absolute;top:-6px;width:2px;height:10px;background:inherit;border-left:2px solid currentColor;opacity:.6}
.vft{position:absolute;top:0;left:0;right:0;display:flex;justify-content:space-between;align-items:center;gap:8px;padding:calc(10px + env(safe-area-inset-top)) 12px 10px;background:linear-gradient(rgba(0,0,0,.6),transparent)}
.vft button{width:42px;height:42px;border-radius:50%;border:0;background:rgba(0,0,0,.45);color:#fff;font-size:1.1rem;cursor:pointer}
.vfs{font-size:.86rem;font-weight:650;text-align:center;flex:1}
.pills{position:absolute;left:0;right:0;top:calc(62px + env(safe-area-inset-top));display:flex;flex-wrap:wrap;gap:6px;justify-content:center;padding:0 10px}
.pills span{padding:4px 10px;border-radius:99px;font-size:.75rem;font-weight:650;background:rgba(0,0,0,.5)}
.pills .ok{box-shadow:inset 0 0 0 1.5px #5fd1a8}.pills .warn{box-shadow:inset 0 0 0 1.5px #ffc862;color:#ffd58a}
.vfmsg{position:absolute;left:12px;right:12px;bottom:calc(118px + env(safe-area-inset-bottom));text-align:center;font-weight:650;font-size:.95rem;text-shadow:0 1px 3px #000}
.vfb{position:absolute;left:0;right:0;bottom:0;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;padding:14px 16px calc(18px + env(safe-area-inset-bottom));background:linear-gradient(transparent,rgba(0,0,0,.7))}
.vfb .alt{justify-self:start;border:0;background:rgba(0,0,0,.45);color:#fff;border-radius:99px;padding:9px 12px;font:600 .8rem inherit;cursor:pointer}
.shutter{width:78px;height:78px;border-radius:50%;border:4px solid #ffc862;background:transparent;padding:5px;cursor:pointer;transition:border-color .2s}
.shutter span{display:block;width:100%;height:100%;border-radius:50%;background:#fff}.shutter.ready{border-color:#5fd1a8;box-shadow:0 0 0 6px rgba(95,209,168,.25)}
.vfr{justify-self:end;font-size:.78rem;font-weight:650}
.toast{position:fixed;left:50%;bottom:calc(18px + env(safe-area-inset-bottom));transform:translate(-50%,140%);z-index:70;max-width:min(92vw,520px);padding:11px 16px;border-radius:14px;background:#13201c;color:#fff;font-size:.88rem;box-shadow:var(--shadow);transition:transform .25s ease}
.toast.on{transform:translate(-50%,0)}.netbar.sync{background:#0e4d45}
.qa{display:flex;gap:8px;flex-wrap:wrap}.qa .minibtn{min-width:84px;min-height:44px}
.quiz{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:6px 0}.quiz button{border:2px solid var(--line);border-radius:14px;background:var(--surface-2);padding:6px;cursor:pointer;color:var(--ink-2);font:700 .9rem inherit}
.quiz svg{width:100%;height:auto;max-height:190px}.quiz .right{border-color:#34b08c;box-shadow:0 0 0 3px rgba(52,176,140,.25)}.quiz .wrong{opacity:.5}
#obsbox textarea,#obsbox select{width:100%;margin-top:8px;padding:9px;border-radius:10px;border:1px solid var(--line);font:inherit;background:transparent;color:inherit}#obsbox select{width:auto}
.hist li[role=button]{cursor:pointer;border-radius:10px}.hist li[role=button]:hover{background:var(--surface-2)}
.shots li.newsite{border-style:dashed}.switch select{font:inherit;padding:4px 6px;border-radius:8px;background:transparent;color:inherit;border:1px solid var(--line)}
.langbtn{margin-left:8px;width:auto;padding:0 12px;border-radius:21px;font-size:.85rem}
body.sun{--mute:var(--ink-2);--line:#9aa6a0;font-size:17.5px}body.sun .card{border-width:2px}body.sun .hungr{color:var(--ink-2)}
"""

JS = r"""
(()=>{const A=window.App,D=A.D,esc=A.esc,G=D.guide,T=G.check.thresholds,$=s=>document.querySelector(s),out=$('#out');
const store={get(k,d){try{const v=localStorage.getItem('lsg.'+k);return v==null?d:JSON.parse(v)}catch(_){return d}},
 set(k,v){try{localStorage.setItem('lsg.'+k,JSON.stringify(v))}catch(_){}}};
const sess={get(k){try{return sessionStorage.getItem('lsg.'+k)}catch(_){return null}},set(k,v){try{sessionStorage.setItem('lsg.'+k,v)}catch(_){}}};
const buzz=p=>{try{navigator.vibrate&&navigator.vibrate(p)}catch(_){}};
let voice=store.get('voice',false),lang=store.get('lang','en');
function say(t){if(!voice||!window.speechSynthesis)return;try{speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(lang==='hi'?tr(t):t);u.lang=lang==='hi'?'hi-IN':'en-IN';u.rate=1.03;speechSynthesis.speak(u)}catch(_){}}
const ICO=n=>D.icons[n].replace('<svg','<svg class="ico"');
const LIVE=!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia&&window.isSecureContext);
if(store.get('sun',false))document.body.classList.add('sun');

/* ---------- journey stepper, mode and survey shot list ---------- */
const J=['Safe spot','Frame','Check','Result'];
$('#journey').innerHTML=J.map(t=>`<li>${t}</li>`).join('');
function step(i){[...$('#journey').children].forEach((li,k)=>{li.className=k<i?'done':k===i?'on':''})}
step(sess.get('safe')?1:0);
let mode=store.get('mode','single'),shot=+(sess.get('shot')||0),done=new Set(JSON.parse(sess.get('done')||'[]'));
function saveShots(){sess.set('shot',shot);sess.set('done',JSON.stringify([...done]))}
function drawMode(){
 $('#mode').innerHTML=`<button type="button" data-m="single" aria-pressed="${mode==='single'}">Single photo</button><button type="button" data-m="survey" aria-pressed="${mode==='survey'}">Site survey (4 shots)</button><button type="button" data-m="pano" aria-pressed="${mode==='pano'}">Overlapping set</button>`;
 $('#shots').innerHTML=mode!=='survey'?'':G.shots.map((s,k)=>`<li data-k="${k}" class="${k===shot?'cur':''} ${done.has(s.id)?'done':''}" tabindex="0" role="button" aria-current="${k===shot}"><span class="n">${done.has(s.id)?'&#10003;':k+1}</span><span><b>${esc(s.t)}</b><span>${esc(s.d)}</span></span></li>`).join('')+(done.size?`<li class="newsite" data-new="1" role="button" tabindex="0"><span class="n">+</span><span><b>Start a new site</b><span>Clears the shot list, notes and location for the next landslide.</span></span></li>`:'');
 const h=$('#capH');if(h)h.textContent=mode==='survey'?'Shot '+(shot+1)+' of '+G.shots.length+': '+G.shots[shot].t:mode==='pano'?'Cover the slope with overlapping photos':'Photograph the slope from below';
 A.pano&&A.pano.show(mode==='pano')}
$('#mode').addEventListener('click',e=>{const m=e.target.dataset.m;if(m){mode=m;store.set('mode',m);drawMode()}});
$('#shots').addEventListener('click',e=>{const li=e.target.closest('li');if(!li)return;if(li.dataset.new){newSite();return}shot=+li.dataset.k;saveShots();drawMode()});
function newSite(){done.clear();shot=0;saveShots();['survey','site','notes','gps','weather'].forEach(k=>sess.set(k,''));sess.set('siteId',String(Date.now()));
 const n=$('#notes');if(n){n.value='';$('#weather').value='';gps=null;$('#gpsout').textContent=''}drawMode();$('#sitecard')?.remove();toast('New site started.')}
$('#shots').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.target.click()}});
drawMode();
const curShot=()=>mode==='survey'?G.shots[shot]:null;

/* ---------- distance helper ---------- */
const dist=$('#dist');
dist.insertAdjacentHTML('afterend',`<div class="fieldmsg" id="distmsg" aria-live="polite"></div>
<details class="pace"><summary>Don't know the distance?</summary>
<div class="pr">Pace it to the toe (only where safe): <input id="steps" type="number" min="1" inputmode="numeric" placeholder="steps"> &times;
<input id="stride" type="number" step="0.05" min="0.4" max="1.2" value="${store.get('stride',0.75)}" aria-label="metres per step"> m =
<b id="paced">&ndash;</b><button type="button" class="minibtn" id="usepace">Use</button></div>
<p style="margin:8px 0 0">Other ways: a laser rangefinder, or Google Maps &rarr; right-click &rarr; <i>Measure distance</i> from where you stand to the slope. A normal walking step is about 0.75 m; measure 10 of your steps once to know yours.</p></details>`);
const paced=()=>{const s=parseFloat($('#steps').value),k=parseFloat($('#stride').value);return s>0&&k>0?Math.round(s*k*10)/10:null};
$('#steps').oninput=$('#stride').oninput=()=>{const p=paced();$('#paced').textContent=p?p+' m':'–';if(+$('#stride').value)store.set('stride',+$('#stride').value)};
$('#usepace').onclick=()=>{const p=paced();if(p){dist.value=p;dist.dispatchEvent(new Event('input'))}};
dist.addEventListener('input',()=>{const v=parseFloat(dist.value);$('#distmsg').textContent=dist.value&&!(v>=1&&v<=3000)?'Use a distance between 1 and 3000 m (or leave it empty).':''});

/* ---------- modals: tour, safety, help ---------- */
document.body.insertAdjacentHTML('beforeend',`<div class="modal" id="mTour" hidden role="dialog" aria-modal="true" aria-labelledby="tourT"><div class="sheet"></div></div>
<div class="modal" id="mSafe" hidden role="dialog" aria-modal="true" aria-labelledby="safeT"><div class="sheet"></div></div>
<div class="modal" id="mHelp" hidden role="dialog" aria-modal="true" aria-labelledby="helpT"><div class="sheet"></div></div>
<div class="vf" id="vf" hidden role="dialog" aria-modal="true" aria-label="Camera with framing guide"><video playsinline muted autoplay></video>
<div class="vfg" data-frame="full"><i class="h1"></i><i class="h2"></i><i class="v1"></i><i class="v2"></i><div class="band top">Top of the slope (scarp)</div><div class="band bot">Bottom (toe / deposit)</div></div>
<div class="lvl" aria-hidden="true"></div>
<div class="vft"><button type="button" data-a="close" aria-label="Close camera">&#10005;</button><span class="vfs"></span><button type="button" data-a="voice" aria-label="Voice guidance"></button></div>
<div class="pills" aria-hidden="true"></div><div class="vfmsg" aria-live="polite"></div>
<div class="vfb"><button type="button" class="alt" data-a="native">Phone camera</button><button type="button" class="shutter" data-a="shoot" aria-label="Take photo"><span></span></button><span class="vfr"></span></div></div>`);
let lastFocus=null;
function openM(id,html){const m=$('#'+id);m.firstElementChild.innerHTML=html;m.hidden=false;lastFocus=document.activeElement;
 (m.querySelector('[data-f]')||m.querySelector('button,input'))?.focus();return m}
function closeM(id){$('#'+id).hidden=true;lastFocus?.focus?.()}
document.addEventListener('keydown',e=>{if(e.key!=='Escape')return;['mTour','mSafe','mHelp'].forEach(id=>{if(!$('#'+id).hidden)closeM(id)});if(!$('#vf').hidden)closeFinder()});
document.querySelectorAll('.modal').forEach(m=>m.addEventListener('click',e=>{if(e.target===m)closeM(m.id)}));

function tour(i=0){const [t,b]=G.tour[i],n=G.tour.length;
 openM('mTour',`<div class="kicker">Guide ${i+1} of ${n}</div><h2 id="tourT">${esc(t)}</h2>${i>0?G.illus[i-1]:''}<p>${esc(b)}</p>
 <div class="row"><button type="button" class="minibtn" data-a="skip">${i?'Back':'Skip'}</button><div class="dots">${G.tour.map((_,k)=>`<i class="${k===i?'on':''}"></i>`).join('')}</div>
 <button type="button" class="minibtn pri" data-f data-a="next">${i<n-1?'Next':'Start'}</button></div>`);
 $('#mTour').onclick=e=>{const a=e.target.dataset.a;if(e.target.id==='mTour'||a==='skip'&&!i){store.set('toured',1);closeM('mTour')}
  else if(a==='skip')tour(i-1);else if(a==='next'){if(i<n-1)tour(i+1);else{store.set('toured',1);quiz(0)}}}}
if(!store.get('toured',0)&&!new URLSearchParams(location.search).get('demo'))tour(0);

function safety(then){
 openM('mSafe',`<div class="kicker">Before you photograph</div><h2 id="safeT">${ICO('alert')} Are you in a safe spot?</h2>
 ${G.safety.map((s,k)=>`<label class="chk"><input type="checkbox" data-s="${k}"><span>${esc(s)}</span></label>`).join('')}
 <p class="hungr">If any of these is not true, move back first. If the slope is moving, leave and report it; no photo is worth it.</p>
 <div class="row"><button type="button" class="minibtn" data-a="cancel">Not now</button><button type="button" class="minibtn pri" data-a="ok" disabled>I am safe, continue</button></div>`);
 const m=$('#mSafe'),ok=m.querySelector('[data-a=ok]');
 m.onchange=()=>{ok.disabled=![...m.querySelectorAll('input')].every(c=>c.checked)};
 m.onclick=e=>{const a=e.target.dataset.a;if(a==='cancel'||e.target===m)closeM('mSafe');if(a==='ok'){sess.set('safe','1');closeM('mSafe');step(1);buzz(15);then()}}}

function help(){const g=G.glossary;
 openM('mHelp',`<div class="row" style="margin:0"><h2 id="helpT" style="margin:0">Help &amp; guide</h2><button type="button" class="minibtn" data-a="x" data-f aria-label="Close help">Close</button></div>
 <h3>Taking a good photo</h3><ol class="steps"><li>Stand below the slope or across the road, 20&ndash;80 m away; never on it.</li><li>Fit the scarp (top) and the toe (bottom) in one frame. Portrait or landscape both work.</li>
 <li>Daylight, sun behind or beside you. Wipe the lens. Hold level with two hands, tap the slope to focus.</li><li>Unsure result? Take more photos from other spots, or use <b>Site survey</b> mode.</li><li>Type the distance (or pace it) for better sizes; a person or car near the toe also helps.</li></ol>
 <h3>Safety</h3><ul class="steps">${G.safety.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>
 <h3>Reading the result</h3><dl class="gloss">${Object.entries(g).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>
 <h3>Limits</h3><p>Labels used to train the model were made by AI and the sizes are not yet checked against field measurements; treat sizes as estimates within their ranges. Always confirm with a geologist or geotechnical engineer before acting.</p>
 <h3>Settings</h3><label class="switch">Voice guidance (camera and results)<input type="checkbox" id="sVoice" ${voice?'checked':''}></label>
 <label class="switch">Sun mode (high contrast, bigger text)<input type="checkbox" id="sSun" ${document.body.classList.contains('sun')?'checked':''}></label>
 <label class="switch">Language / भाषा<select id="sLang"><option value="en" ${lang==='en'?'selected':''}>English</option><option value="hi" ${lang==='hi'?'selected':''}>हिन्दी</option></select></label>
 <div class="row"><button type="button" class="minibtn" data-a="tour">Replay the guide</button><button type="button" class="minibtn" data-a="quiz">Practice quiz</button><button type="button" class="minibtn" data-a="clear">Clear recent photos</button></div>
 <p class="hungr">Live camera guide: ${LIVE?'available':'needs a secure address (https or localhost); the phone camera app is used instead'}.</p>`);
 const m=$('#mHelp');
 m.onclick=e=>{const a=e.target.dataset.a;if(a==='x'||e.target===m)closeM('mHelp');if(a==='tour'){closeM('mHelp');tour(0)}if(a==='quiz'){closeM('mHelp');quiz(0)}if(a==='clear'){store.set('hist',[]);safe(async()=>{for(const x of await DB.all('results'))await DB.del('results',x.id)});drawHist();e.target.textContent='Cleared'}};
 m.onchange=e=>{if(e.target.id==='sVoice'){voice=e.target.checked;store.set('voice',voice);say('Voice guidance on')}
  if(e.target.id==='sSun'){document.body.classList.toggle('sun',e.target.checked);store.set('sun',e.target.checked)}
  if(e.target.id==='sLang'){store.set('lang',e.target.value);location.reload()}}}
function quiz(i=0,score=0){const Q=G.quiz,n=Q.length;
 if(i>=n){store.set('quizzed',score);openM('mTour',`<div class="kicker">Practice</div><h2 id="tourT">${score} of ${n} correct</h2><p>${score===n?'Well done. You are ready for the field.':'Read the photo tips once more; the live camera guide will also help on site.'}</p><div class="row"><span></span><button type="button" class="minibtn pri" data-f data-a="end">Done</button></div>`);
  $('#mTour').onclick=e=>{if(e.target.dataset.a==='end'||e.target.id==='mTour')closeM('mTour')};return}
 const q=Q[i];let answered=false;openM('mTour',`<div class="kicker">Practice ${i+1} of ${n}</div><h2 id="tourT">${esc(q.q)}</h2><div class="quiz"><button type="button" data-c="a" aria-label="Photo A">${q.a}<span>A</span></button><button type="button" data-c="b" aria-label="Photo B">${q.b}<span>B</span></button></div>
 <p id="qwhy" class="hungr" aria-live="polite"></p><div class="row"><button type="button" class="minibtn" data-a="skip">Skip practice</button><button type="button" class="minibtn pri" data-a="next" hidden>Next</button></div>`);
 $('#mTour').onclick=e=>{const c=e.target.closest('[data-c]')?.dataset.c,a=e.target.dataset.a;
  if(c&&!answered){answered=true;const ok=c===q.correct;if(ok)score++;buzz(ok?30:[40,60,40]);$('#mTour').querySelectorAll('[data-c]').forEach(b=>b.classList.add(b.dataset.c===q.correct?'right':'wrong'));
   $('#qwhy').innerHTML=`<b>${ok?'Correct.':'Not quite.'}</b> <span>${esc(q.why)}</span>`;const nx=$('#mTour').querySelector('[data-a=next]');nx.hidden=false;nx.focus();say((ok?'Correct. ':'Not quite. ')+q.why)}
  if(a==='next')quiz(i+1,score);if(a==='skip'||e.target.id==='mTour')closeM('mTour')}}
$('#helpBtn').onclick=help;

/* ---------- capture entry points ---------- */
const camIn=$('#cam'),galIn=$('#gal');
function openCamera(){if(LIVE)openFinder();else camIn.click()}
camIn.parentElement.addEventListener('click',e=>{if(e.target===camIn)return;   // the input's own click (label activation or programmatic) passes through
 if(!sess.get('safe')){e.preventDefault();safety(openCamera);return}
 if(LIVE){e.preventDefault();openFinder()}});
galIn.parentElement.addEventListener('click',()=>{if(sess.get('safe'))step(1)});

/* ---------- photo metrics (same as tools/calibrate_photo_check.py) ---------- */
let mc=null,prevG=null;
function pctl(a,p){a=[...a].sort((x,y)=>x-y);const i=(a.length-1)*p/100,l=Math.floor(i);return a[l]+(a[Math.min(l+1,a.length-1)]-a[l])*(i-l)}
// Pillow's antialiased bilinear resampling (triangle filter, support scaled by the reduction, horizontal pass
// then vertical, uint8 after each pass), so the phone computes the same numbers as the calibration script.
// Browser canvas scaling differs between browsers and made sharpness ~40% higher in tests.
function taps(inN,outN){const sc=inN/outN,fs=Math.max(1,sc),T=[];for(let i=0;i<outN;i++){const c=(i+.5)*sc,a=Math.max(0,Math.floor(c-fs+.5)),b=Math.min(inN,Math.floor(c+fs+.5)),w=[];let s=0;
 for(let x=a;x<b;x++){const t=Math.max(0,1-Math.abs((x-c+.5)/fs));w.push(t);s+=t}T.push([a,w.map(v=>s?v/s:0)])}return T}
function metrics(src,keep){const w0=src.videoWidth||src.width,h0=src.videoHeight||src.height,k0=Math.min(1,(keep?1024:4096)/Math.max(w0,h0)),W=Math.round(w0*k0),H=Math.round(h0*k0);
 mc=mc||document.createElement('canvas');mc.width=W;mc.height=H;const x=mc.getContext('2d',{willReadFrequently:true});x.drawImage(src,0,0,W,H);
 const S=x.getImageData(0,0,W,H).data,s=512/Math.max(W,H),w=Math.max(3,Math.round(W*s)),h=Math.max(3,Math.round(H*s)),tx=taps(W,w),ty=taps(H,h),M=new Uint8Array(w*H*3),R=v=>v<0?0:v>255?255:Math.floor(v+.5);
 for(let y=0;y<H;y++)for(let i=0;i<w;i++){const [a,wt]=tx[i];let r=0,gg=0,b=0;for(let k=0,p=(y*W+a)*4;k<wt.length;k++,p+=4){const c=wt[k];r+=S[p]*c;gg+=S[p+1]*c;b+=S[p+2]*c}const q=(y*w+i)*3;M[q]=R(r);M[q+1]=R(gg);M[q+2]=R(b)}
 const n=w*h,g=new Float32Array(n),C3=new Uint8Array(n*3);let su=0,sq=0,dk=0,br=0,sk=0,ni=0;
 for(let j=0;j<h;j++){const [a,wt]=ty[j];for(let i=0;i<w;i++){let r=0,gg=0,b=0;for(let k=0,q=(a*w+i)*3;k<wt.length;k++,q+=w*3){const c=wt[k];r+=M[q]*c;gg+=M[q+1]*c;b+=M[q+2]*c}
  const rr=R(r),g2=R(gg),bb=R(b),v=Math.fround(Math.fround(Math.fround(.299*rr)+Math.fround(.587*g2))+Math.fround(.114*bb)),o=(j*w+i)*3;C3[o]=rr;C3[o+1]=g2;C3[o+2]=bb;g[j*w+i]=v;su+=v;sq+=v*v;if(v<16)dk++;else if(v>239)br++}}
 const tv=[];for(let a=0;a<8;a++)for(let b=0;b<8;b++){const y0=1+Math.floor(a*(h-2)/8),y1=1+Math.floor((a+1)*(h-2)/8),x0=1+Math.floor(b*(w-2)/8),x1=1+Math.floor((b+1)*(w-2)/8);let s1=0,s2=0,k=0;
  for(let y=y0;y<y1;y++)for(let q=y*w+x0,e=y*w+x1;q<e;q++){const l=g[q-w]+g[q+w]+g[q-1]+g[q+1]-4*g[q];s1+=l;s2+=l*l;k++;
   const o=q*3,gv=g[q];ni++;if(Math.abs(l)<6&&((C3[o+2]>C3[o]+8&&C3[o+2]>=C3[o+1]&&gv>100)||gv>220))sk++}tv.push(k?s2/k-(s1/k)**2:0)}
 let motion=null;if(keep){if(prevG&&prevG.length===n){let m=0;for(let j=0;j<n;j+=7)m+=Math.abs(g[j]-prevG[j]);motion=m/(n/7)}prevG=g}
 const mean=su/n;return {lum:mean,dark:dk/n,bright:br/n,contrast:Math.sqrt(Math.max(0,sq/n-mean*mean)),sharp:pctl(tv,90),sky:ni?sk/ni:0,w:w0,h:h0,motion}}
function judge(m,t){const c=[];
 c.push(['Light',m.lum<T.lum_min?'Too dark. Face the slope with the sun behind or beside you, or wait for better light.':m.lum>T.lum_max?'Too bright. Point less at the sky; tap the slope on screen to set exposure.':null]);
 c.push(['Shadows',m.dark>T.dark_max?'Large black areas: deep shadow hides the surface. Step sideways so the slope is evenly lit.':null]);
 c.push(['Highlights',m.bright>T.bright_max?'Large white areas: sky or glare is blown out. Tilt down so less sky is in the frame.':null]);
 c.push(['Haze / contrast',m.contrast<T.contrast_min?'Flat, low-contrast image (fog, rain or a dirty lens). Wipe the lens; in fog, move closer.':null]);
 c.push(['Framing',m.sky>T.sky_max?'Mostly sky. Tilt down so the slope fills most of the frame.':null]);
 c.push(['Sharpness',m.sharp<T.sharp_min?'Blurred. Use two hands, tap the slope to focus, and hold still while the photo is taken.':null]);
 c.push(['Resolution',Math.min(m.w,m.h)<720?'Small image ('+m.w+'×'+m.h+'). Use the original camera photo, not a screenshot or a chat-app copy.':null]);
 if(t){c.push(['Level',Math.abs(t.roll)>10?'Phone tilted sideways by '+Math.round(Math.abs(t.roll))+'°. Hold it level.':null]);
  c.push(['Aim',t.elev<-15?'Camera points down. Stand below the slope and look up at it.':t.elev>60?'Camera points steeply up. Step back so the whole slope fits.':null])}
 return c}

/* ---------- phone orientation: gravity in image axes (x right, y down, z forward) ---------- */
let ori=null;
function onOri(e){if(e.beta==null)return;const b=e.beta*Math.PI/180,g=(e.gamma||0)*Math.PI/180,x=Math.cos(b)*Math.sin(g),y=Math.sin(b),z=Math.cos(b)*Math.cos(g),
 th=((screen.orientation&&screen.orientation.angle)??window.orientation??0)*Math.PI/180,ix=x*Math.cos(th)-y*Math.sin(th),iy=x*Math.sin(th)+y*Math.cos(th);
 ori={g:[ix,iy,z].map(v=>v.toFixed(3)).join(','),roll:Math.atan2(ix,iy)*180/Math.PI,elev:Math.atan2(-z,Math.hypot(ix,iy))*180/Math.PI}}
let oriOn=false;async function listenOri(){if(oriOn)return;oriOn=true;try{if(window.DeviceOrientationEvent&&DeviceOrientationEvent.requestPermission)await DeviceOrientationEvent.requestPermission()}catch(_){}
 window.addEventListener('deviceorientation',onOri)}
document.querySelectorAll('.btn').forEach(b=>b.addEventListener('click',listenOri,{once:true}));

/* ---------- live viewfinder ---------- */
let stream=null,loop=null,wasReady=false,lastSay=0,cap=null;
const vf=$('#vf'),vid=vf.querySelector('video');
async function openFinder(){listenOri();step(1);vf.hidden=false;const s=curShot(),pz=mode==='pano'&&A.pano;
 vf.querySelector('.vfg').dataset.frame=s?s.frame:pz?'pano':'full';vf.querySelector('.vfs').textContent=s?'Shot '+(shot+1)+': '+s.t:pz?A.pano.label():'Frame the whole slope';
 if(pz)A.pano.finder(true);
 vf.querySelector('[data-a=voice]').textContent=voice?'\u{1F50A}':'\u{1F507}';
 try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:3840},height:{ideal:2160}},audio:false});
  vid.srcObject=stream;await vid.play();
  try{const tr=stream.getVideoTracks()[0];cap=window.ImageCapture?new ImageCapture(tr):null}catch(_){cap=null}
  prevG=null;wasReady=false;loop=setInterval(liveTick,450);say(s?s.d:pz?A.pano.hint():'Frame the whole slope from below, top to bottom.');
  vf.querySelector('[data-a=shoot]').focus()}
 catch(err){closeFinder();camIn.click()}}
function closeFinder(){clearInterval(loop);loop=null;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;vf.hidden=true;A.pano&&A.pano.finder(false)}
function liveTick(){if(!vid.videoWidth)return;const m=metrics(vid,true),t=ori,c=judge(m,t).filter(x=>x[0]!=='Resolution'&&!(mode==='pano'&&x[0]==='Framing')),pills=[],bad=c.filter(x=>x[1]);
 const px=mode==='pano'&&A.pano?A.pano.tick(vid,t):null;if(px){bad.unshift(...px.bad)}
 const steady=m.motion==null||m.motion<G.steadyMax;if(!steady)bad.push(['Steady','Hold still for a moment.']);
 const lvl=vf.querySelector('.lvl');if(t){lvl.style.transform=`translate(-50%,-50%) rotate(${-t.roll}deg)`;lvl.classList.toggle('ok',Math.abs(t.roll)<=3)}else lvl.style.display='none';
 for(const [k,v] of c)pills.push(`<span class="${v?'warn':'ok'}">${k}</span>`);pills.push(`<span class="${steady?'ok':'warn'}">Steady</span>`);
 if(t)pills.push(`<span class="${t.elev>=-10&&t.elev<=45?'ok':'warn'}">Aim ${Math.round(t.elev)}°</span>`);
 if(px)pills.unshift(...px.pills);
 vf.querySelector('.pills').innerHTML=pills.join('');
 const ready=!bad.length,msg=ready?(px&&px.good)||'Good. Take the photo.':bad[0][1];vf.querySelector('.vfmsg').textContent=msg;
 vf.querySelector('.shutter').classList.toggle('ready',ready);vf.querySelector('.vfr').textContent=ready?'Ready':'';
 const now=Date.now();if(ready&&!wasReady){buzz(30);say(msg);lastSay=now}else if(!ready&&now-lastSay>5000){say(msg);lastSay=now}wasReady=ready}
async function shoot(){const t=ori;let blob=null;buzz([20,40,20]);
 try{if(cap&&cap.takePhoto)blob=await cap.takePhoto()}catch(_){blob=null}
 if(!blob){const c=document.createElement('canvas');c.width=vid.videoWidth;c.height=vid.videoHeight;c.getContext('2d').drawImage(vid,0,0);blob=await new Promise(r=>c.toBlob(r,'image/jpeg',.92))}
 if(mode==='pano'&&A.pano){gstate.live=true;A.pano.shot(blob,t,vid);return}
 closeFinder();start(blob,true,t,true)}
vf.addEventListener('click',e=>{const a=e.target.closest('[data-a]')?.dataset.a;if(a==='close')closeFinder();if(a==='shoot')shoot();
 if(a==='native'){closeFinder();camIn.click()}if(a==='voice'){voice=!voice;store.set('voice',voice);e.target.textContent=voice?'\u{1F50A}':'\u{1F507}';say('Voice guidance on')}});

/* ---------- language: English / Hindi (text nodes are translated as they appear) ---------- */
const HI=G.hi||{};
const RX=[[/^Shot (\d) of (\d): (.+)$/,(m,a,b,c)=>`शॉट ${a}/${b}: ${HI[c]||c}`],[/^Shot (\d): (.+)$/,(m,a,b)=>`शॉट ${a}: ${HI[b]||b}`],
 [/^Uploading… (\d+)%$/,(m,a)=>`अपलोड हो रहा है… ${a}%`],[/^(\d+) s · about (\d+) s left$/,(m,a,b)=>`${a} सेकंड · लगभग ${b} सेकंड बाकी`],
 [/^(\d+) s · taking longer than usual, still working$/,(m,a)=>`${a} सेकंड · सामान्य से अधिक समय, काम जारी है`],[/^Guide (\d) of (\d)$/,(m,a,b)=>`गाइड ${a}/${b}`],
 [/^Practice (\d) of (\d)$/,(m,a,b)=>`अभ्यास ${a}/${b}`],[/^(\d) of (\d) correct$/,(m,a,b)=>`${b} में से ${a} सही`],[/^Tip: (.+)$/,(m,a)=>'सुझाव: '+(HI[a]||a)],
 [/^Aim (-?\d+)°$/,(m,a)=>`दिशा ${a}°`],[/^(\d) things could be better$/,(m,a)=>`${a} चीज़ें बेहतर हो सकती हैं`],[/^(.+):$/,(m,a)=>HI[a]?HI[a]+':':m],
 [/^Estimated dimensions · depth trust: (\w+)$/,(m,a)=>`अनुमानित आयाम · गहराई भरोसा: ${HI[a]||a}`],[/^Site result · (\d) shots combined$/,(m,a)=>`स्थल परिणाम · ${a} शॉट मिलाकर`],
 [/^Recent observations \(this phone\)$/,()=>'हाल के अवलोकन (यह फ़ोन)'],[/^Volume \((.+)\)$/,(m,a)=>`आयतन (${a})`],
 [/^(.+) \(second choice\)$/,(m,a)=>`${HI[a]||a} (दूसरा विकल्प)`],[/^Measured with your distance of ([\d.]+) m\.$/,(m,a)=>`आपकी ${a} मीटर दूरी से मापा गया।`]];
function tr(s){if(lang!=='hi'||!s)return s;const k=s.trim();if(!k)return s;let v=HI[k];if(v==null)for(const [re,f] of RX){if(re.test(k)){v=k.replace(re,f);break}}return v==null?s:s.replace(k,v)}
function trTree(n){if(n.nodeType===3){const v=tr(n.nodeValue);if(v!==n.nodeValue)n.nodeValue=v;return}
 if(n.nodeType!==1||/^(SCRIPT|STYLE|svg|TEXTAREA)$/.test(n.tagName))return;
 for(const a of ['placeholder','aria-label','title']){const v=n.getAttribute(a);if(v){const t=tr(v);if(t!==v)n.setAttribute(a,t)}}
 n.childNodes.forEach(trTree)}
if(lang==='hi'){document.documentElement.lang='hi';trTree(document.body);
 new MutationObserver(ms=>{for(const m of ms)m.addedNodes.forEach(trTree)}).observe(document.body,{childList:true,subtree:true})}
$('#langBtn').textContent=lang==='hi'?'EN':'हिं';$('#langBtn').onclick=()=>{store.set('lang',lang==='hi'?'en':'hi');location.reload()};

/* ---------- local database: offline queue and full results (IndexedDB, this phone only) ---------- */
const DB=(()=>{let p=null;const open=()=>p||(p=new Promise((res,rej)=>{try{const r=indexedDB.open('landslide-guide',1);
 r.onupgradeneeded=()=>{const d=r.result;d.createObjectStore('queue',{keyPath:'id'});d.createObjectStore('results',{keyPath:'id'})};r.onsuccess=()=>res(r.result);r.onerror=()=>rej(r.error)}catch(e){rej(e)}}));
 const run=async(st,m,fn)=>{const d=await open();return new Promise((res,rej)=>{const t=d.transaction(st,m),q=fn(t.objectStore(st));let v;if(q)q.onsuccess=()=>{v=q.result};t.oncomplete=()=>res(v);t.onerror=()=>rej(t.error)})};
 return {put:(s,v)=>run(s,'readwrite',o=>o.put(v)),get:(s,k)=>run(s,'readonly',o=>o.get(k)),del:(s,k)=>run(s,'readwrite',o=>o.delete(k)),all:s=>run(s,'readonly',o=>o.getAll())}})();
const safe=async(f,d)=>{try{return await f()}catch(_){return d}};
function toast(t){let el=$('#toast');if(!el){document.body.insertAdjacentHTML('beforeend','<div id="toast" class="toast" role="status" aria-live="polite"></div>');el=$('#toast')}
 el.textContent=t;el.classList.add('on');clearTimeout(el._t);el._t=setTimeout(()=>el.classList.remove('on'),5000)}
async function photoURL(blob,max){try{const b=await createImageBitmap(blob),s=Math.min(1,max/Math.max(b.width,b.height)),c=document.createElement('canvas');c.width=Math.round(b.width*s);c.height=Math.round(b.height*s);
 c.getContext('2d').drawImage(b,0,0,c.width,c.height);return c.toDataURL('image/jpeg',max>200?.82:.6)}catch(_){return ''}}
async function urlShrink(u,max){if(!u)return '';try{return await photoURL(await (await fetch(u)).blob(),max)}catch(_){return ''}}

/* ---------- observation record: notes, weather, location ---------- */
$('.pace').insertAdjacentHTML('afterend',`<details class="pace" id="obsbox"><summary>Notes &amp; location (for the report)</summary>
<textarea id="notes" rows="2" placeholder="Site name, road / km, what you saw"></textarea>
<div class="pr"><select id="weather" aria-label="Weather"><option value="">Weather…</option><option value="Dry">Dry</option><option value="Raining now">Raining now</option><option value="Rain in the last 24 h">Rain in the last 24 h</option></select>
<button type="button" class="minibtn" id="gpsbtn">Add my location</button></div><div class="hungr" id="gpsout"></div></details>`);
const notes=$('#notes'),weather=$('#weather');notes.value=sess.get('notes')||'';weather.value=sess.get('weather')||'';
notes.oninput=()=>sess.set('notes',notes.value);weather.onchange=()=>sess.set('weather',weather.value);
let gps=null;try{gps=JSON.parse(sess.get('gps')||'null')}catch(_){}
const showGps=()=>{$('#gpsout').textContent=gps?`Location: ${gps.lat.toFixed(5)}, ${gps.lon.toFixed(5)} (±${Math.round(gps.acc)} m)`:''};showGps();
$('#gpsbtn').onclick=()=>{if(!navigator.geolocation||!window.isSecureContext){$('#gpsout').textContent='Location needs a secure address (start the app with --https). You can type it in the notes.';return}
 $('#gpsout').textContent='Finding your location…';navigator.geolocation.getCurrentPosition(p=>{gps={lat:p.coords.latitude,lon:p.coords.longitude,acc:p.coords.accuracy,t:Date.now()};sess.set('gps',JSON.stringify(gps));showGps();buzz(20)},
  e=>{$('#gpsout').textContent='Location not available ('+e.message+'). You can type it in the notes.'},{enableHighAccuracy:true,timeout:20000,maximumAge:60000})};
const meta=()=>({t:Date.now(),notes:notes.value.trim(),weather:weather.value,gps,shot:curShot()?.t||'',shotId:curShot()?.id||'',mode,dist:lastDist});

/* ---------- what the guidance did (sent with the photo, logged for tools/guidance_report.py) ---------- */
let gstate={flags:[],override:false,retakes:0,live:false};
const gx=()=>JSON.stringify({flags:gstate.flags,override:gstate.override,retakes:gstate.retakes,live:gstate.live,mode,shot:curShot()?.id||'',lang,set:mode==='pano'&&A.pano?A.pano.gx():''});
function params(o){const q=new URLSearchParams(location.search);q.delete('demo');for(const [k,v] of Object.entries(o))if(v!=null&&v!=='')q.set(k,v);return q}
const gArg=()=>lastSrc==='cam'&&lastTilt?lastTilt.g:null;
const ema=(k,v,d)=>Math.round(.6*store.get(k,d)+.4*v);

/* ---------- instant photo check ---------- */
let lastF35=null,lastBlob=null,lastTilt=null,lastSrc='cam',lastDist=null,xhr=null,job=null,cur=null,curMeta=null,curPhoto='';
A.onPick=(f,fromCam)=>start(f,fromCam,fromCam?ori:null,false);
async function start(file,fromCam,t,live){lastSrc=fromCam?'cam':'gal';gstate.live=!!live;gstate.override=false;step(2);
 lastF35=A.f35Of?await A.f35Of(file):null;
 const fig=$('#fig'),prev=$('#prev');prev.src=URL.createObjectURL(file);fig.style.display='block';A.setChip('','Checking photo…',true);
 fig.scrollIntoView({behavior:'smooth',block:'start'});
 let m=null;try{const bmp=await createImageBitmap(file,{imageOrientation:'from-image'});m=metrics(bmp);bmp.close&&bmp.close()}catch(_){}
 lastBlob=await A.shrink(file);lastTilt=t;
 const c=m?judge(m,t):[],bad=c.filter(x=>x[1]);gstate.flags=bad.map(b=>b[0]);
 if(!bad.length){send();return}
 A.setChip('signal','Check the photo');say(tr('Photo check')+': '+bad.map(b=>tr(b[0])).join(', ')+'. '+tr(bad[0][1]));buzz([40,60,40]);
 out.innerHTML=`<section class="card bad" aria-live="polite"><div class="kicker">Instant photo check</div><h2>${bad.length===1?'One thing could be better':bad.length+' things could be better'}</h2>
 <ul class="qc">${c.map(([k,v])=>`<li>${v?`<span class="warn">${ICO('alert')}</span><span><b>${k}:</b> <span>${esc(v)}</span></span>`:`<span class="ok">${ICO('check')}</span><span><b>${k}</b> <span>OK</span></span>`}</li>`).join('')}</ul>
 <p class="hungr" style="margin:0 0 12px">Flagged because fewer than 1 in ${Math.round(100/G.check.percentile_q)} training photos looks like this, so the answer would be less reliable. You can still send it.</p>
 <div class="btns"><button type="button" class="btn primary" data-a="retake">${ICO('camera')}Retake</button><button type="button" class="btn ghost" data-a="send">Analyse anyway</button></div></section>`;
 out.querySelector('[data-a=retake]').onclick=()=>{gstate.retakes++;fig.style.display='none';out.innerHTML='';step(1);lastSrc==='cam'?openCamera():galIn.click()};
 out.querySelector('[data-a=send]').onclick=()=>{gstate.override=true;send()}}

/* ---------- requests with progress: type first (seconds), then sizes ---------- */
function prog(el,stages,eta,withUpload){
 el.innerHTML=`<div class="prog"><i></i></div><div class="hungr ptime">${withUpload?'Uploading…':''}</div><ul class="stages">${stages.map(s=>`<li>${esc(s[1])}</li>`).join('')}</ul>`;
 const bar=el.querySelector('.prog i'),tm=el.querySelector('.ptime'),li=[...el.querySelectorAll('.stages li')],t0=Date.now(),U=withUpload?12:0;let up=0,ts=withUpload?0:Date.now();
 const iv=setInterval(()=>{if(!el.isConnected){clearInterval(iv);return}
  if(!ts){bar.style.width=(up*U).toFixed(1)+'%';tm.textContent='Uploading… '+Math.round(up*100)+'%';return}
  const f=(Date.now()-ts)/eta,p=f<1?f*.9:.9+.09*(1-Math.exp(-(f-1)*2)),left=Math.max(0,Math.round((eta-(Date.now()-ts))/1000));
  bar.style.width=(U+p*(100-U)).toFixed(1)+'%';tm.textContent=Math.round((Date.now()-t0)/1000)+' s'+(f<1?' · about '+left+' s left':' · taking longer than usual, still working');
  li.forEach((x,k)=>{const a=stages[k][0],b=k+1<stages.length?stages[k+1][0]:9;x.className=f>=b?'done':f>=a?'on':''})},400);
 return {up:v=>{up=v},srv:()=>{ts=ts||Date.now()},stop:()=>clearInterval(iv)}}
function tipRot(el){let i=Math.floor(Math.random()*G.tips.length);const put=()=>{el.textContent='Tip: '+G.tips[i]};put();
 const iv=setInterval(()=>{if(!el.isConnected){clearInterval(iv);return}i=(i+1)%G.tips.length;put()},7000)}
function post(url,body,P,h){const x=new XMLHttpRequest();xhr=x;x.open('POST',url);x.timeout=240000;
 if(body){x.setRequestHeader('Content-Type',body.type||'application/octet-stream');x.upload.onprogress=e=>{if(e.lengthComputable&&P)P.up(e.loaded/e.total)};x.upload.onload=()=>P&&P.srv()}
 x.onload=()=>{P&&P.stop();let r;try{r=JSON.parse(x.responseText)}catch(_){r={error:'Unexpected reply from the server ('+x.status+')'}}x.status===200?h.ok(r):h.err(r.error||('Server error '+x.status),x.status)};
 x.onerror=()=>{P&&P.stop();h.err('Could not reach the server. Check the Wi-Fi or hotspot connection to the laptop running the app.',0)};
 x.ontimeout=()=>{P&&P.stop();h.err('The server took more than 4 minutes. It may be busy or out of memory; try again.',-1)};
 x.onabort=()=>{P&&P.stop();h.err(null,-2)};x.send(body||null);return x}
function busy(title){out.innerHTML=`<section class="card result" aria-busy="true"><div class="kicker">${esc(title)}${curShot()?' · '+esc(curShot().t):''}</div><div id="pg"></div><div class="tipline" id="ptip"></div>
 <div style="margin-top:12px"><button type="button" class="minibtn" id="pcancel">Cancel</button></div></section>`;tipRot($('#ptip'));$('#pcancel').onclick=()=>xhr&&xhr.abort()}
function cancelled(){A.setChip('','Cancelled');out.innerHTML='';step(1)}
function send(){const dv=parseFloat(dist.value);lastDist=dv>=1&&dv<=3000?dv:null;curMeta=meta();job=null;
 if(!navigator.onLine){queueIt();return}
 A.setChip('','Identifying…',true);busy('Analysing photo');
 const P=prog($('#pg'),G.stagesType,store.get('etaT',4000),true);
 post('predict?'+params({stage:'type',d:lastDist,g:gArg(),f35:lastF35,gx:gx()}),lastBlob,P,{
  ok:r=>{if(r.ms)store.set('etaT',ema('etaT',r.ms,4000));cur=r;job=r.job||null;A.render(r);afterType(r);if(job)measure();else finish(r)},
  err:(m,s)=>s===-2?cancelled():fail(m,s===0)})}
function fullSend(d0){A.setChip('','Analysing…',true);busy('Sending the photo again');
 const P=prog($('#pg'),G.stagesType.concat(G.stagesDims.map(([f,t])=>[.12+f*.88,t])),store.get('etaT',4000)+store.get('etaD',28000),true);
 post('predict?'+params({stage:'full',d:d0??lastDist,g:gArg(),f35:lastF35,gx:gx()}),lastBlob,P,{
  ok:r=>{if(d0)lastDist=d0;const id=cur&&cur.__id;cur=r;if(id)r.__id=id;job=null;A.render(r);afterType(r);if(r.dimensions)afterDims(r);finish(r)},
  err:(m,s)=>s===-2?cancelled():fail(m,s===0)})}
function measure(d0){const slot=document.createElement('section');slot.className='card';slot.id='dimslot';
 slot.innerHTML=`<div class="kicker">Estimated dimensions</div><p class="hungr" style="margin:4px 0 0">The type is ready above. Measuring the size now; you can already tick the signs and answer the questions.</p><div id="dpg"></div>
 <button type="button" class="minibtn" id="dcancel" style="margin-top:10px">Skip sizes</button>`;
 const old=$('#dimcard');if(old)old.replaceWith(slot);else{const m=out.querySelector('.meta');m?out.insertBefore(slot,m):out.appendChild(slot)}
 A.setChip('','Measuring…',true);$('#dcancel').onclick=()=>xhr&&xhr.abort();
 const dv=d0??lastDist,P=prog($('#dpg'),G.stagesDims,store.get('etaD',28000),false);
 post('measure?'+params({job,d:dv,g:gArg()}),null,P,{
  ok:res=>{store.set('etaD',ema('etaD',res.ms||28000,28000));lastDist=dv||null;if(curMeta)curMeta.dist=lastDist;cur.dimensions=res.dimensions;placeDims(cur);afterDims(cur);finish(cur)},
  err:(m,s)=>{if(s===410&&lastBlob){fullSend(dv);return}
   slot.innerHTML=`<div class="kicker">Estimated dimensions</div><p class="hungr">${s===-2?'Sizes skipped.':'Sizes could not be measured: '+esc(m)}</p><button type="button" class="minibtn" id="dretry">Measure sizes</button>`;
   $('#dretry').onclick=()=>measure(dv);typeChip(cur);finish(cur)}})}
function typeChip(r){A.setChip(r.status==='uncertain'?'signal':r.status==='bad_photo'?'bad':'',r.status==='uncertain'?'Uncertain':r.status==='bad_photo'?'Retake photo':'Landslide type')}
function findDims(){return $('#dimcard')||[...out.querySelectorAll('.card')].find(c=>/^Estimated dimensions|^अनुमानित आयाम/.test((c.querySelector('.kicker')?.textContent||'').trim())&&c.id!=='dimslot')}
function placeDims(r){const w=document.createElement('div');w.innerHTML=A.dims(r.dimensions);const c=w.firstElementChild;c.id='dimcard';const s=$('#dimslot')||$('#dimcard');s?s.replaceWith(c):out.appendChild(c);typeChip(r)}
function fail(msg,net){A.fail(msg);const c=out.firstElementChild;
 c.insertAdjacentHTML('beforeend',`<div class="btns" style="margin-top:12px"><button type="button" class="btn primary" data-a="retry">Try again</button>${net?'<button type="button" class="btn ghost" data-a="later">Send later</button>':'<button type="button" class="btn ghost" data-a="new">New photo</button>'}</div>`);
 c.querySelector('[data-a=retry]').onclick=()=>lastBlob&&send();
 c.querySelector('[data-a=later]')?.addEventListener('click',queueIt);c.querySelector('[data-a=new]')?.addEventListener('click',()=>{out.innerHTML='';$('#fig').style.display='none';step(1)});
 say(tr('Something went wrong')+'. '+tr(msg));buzz([80,60,80])}

/* ---------- after the type: next steps, decision questions, tick-able signs ---------- */
function addAct(t,l,a){const nc=$('#nextcard');if(!nc)return;nc.hidden=false;const li=document.createElement('li');
 li.innerHTML=`<span>${t}</span>${l?`<button type="button" class="minibtn${a==='next'||a==='retake'?' pri':''}" data-a="${a}">${l}</button>`:''}`;
 nc.querySelector('.acts').appendChild(li);li.querySelector('button')?.addEventListener('click',()=>act(a))}
function afterType(r,replay){step(3);const N=D.names,top=r.ranked[0][0],sec=r.ranked[1][0],close=r.ranked[0][1]-r.ranked[1][1]<.2;
 out.firstElementChild.insertAdjacentHTML('afterend','<section class="card" id="nextcard" hidden><div class="kicker">Next steps</div><ul class="acts"></ul></section>');
 if(r.status==='bad_photo')addAct('Retake from below with the whole slope in view.','Retake','retake');
 else if(r.status==='uncertain')addAct(r.family?`Answer the questions or tick the signs below to tell <b>${esc(N[top])}</b> from <b>${esc(N[sec])}</b>.`:'Not sure. Answer the questions below, or take a second photo from another spot.','Another photo','retake');
 if(r.status!=='bad_photo'&&(r.status==='uncertain'||close))ask(r,replay);
 ticks(r);reportBar();
 if(!replay){buzz(r.status==='ok'?40:[40,80,40]);say(r.status==='bad_photo'?tr('Please retake the photo.'):(r.status==='uncertain'?tr('Uncertain. Most likely')+' ':'')+tr(N[r.label])+', '+Math.round(r.confidence*100)+' %.')}}
function afterDims(r,replay){const d=r.dimensions;if(!d)return;const c=findDims();if(c)c.id='dimcard';
 if(!d.available){if(r.status!=='bad_photo')addAct(esc(d.reason||'No size estimate for this photo.'),'','');return}
 const ce=d.cut_edges||[];if(ce.length)addAct(cutMsg(ce),'Retake','retake');
 if(!replay&&!lastDist&&!/object|person|car|calibrat/i.test(d.scale_source||''))addAct('Add the distance to the slope and re-measure: sizes get much tighter ranges.','Add distance','dist');
 if(!replay&&lastDist)addAct(`Measured with your distance of ${lastDist} m.`,'Change','dist');
 if(d.depth_trust==='low')addAct('Sizes are rough here. A person, car or pole near the toe lets the app check its scale.','Scale shot','scale');
 if(d.grains&&!d.grains.reliable&&['debris_flow','fall_topple','earth_mud_flow'].includes(d.label))addAct('Grain size not readable. A close-up of the deposit (5–15 m) gives it.','Close-up','deposit');
 gloss();const m=d.measures&&d.measures[0];if(!replay&&m)say(tr(m.name)+' '+m.value+' '+(m.unit==='m'?tr('metres'):m.unit))}
function cutMsg(e){const how={top:'tilt the phone up or step back so the scarp is inside the frame',bottom:'tilt down or step back so the toe and deposit are inside',left:'turn left or step back',right:'turn right or step back'};
 return `<b>${esc(tr('The landslide runs off the edge of the photo'))}</b> (${e.map(x=>esc(tr(x))).join(', ')}): ${esc(tr('sizes are minimums'))}. ${e.map(x=>esc(tr(how[x]))).join('; ')}.`}
function act(a){if(a==='retake'){gstate.retakes++;out.innerHTML='';$('#fig').style.display='none';step(1);lastSrc==='cam'?openCamera():galIn.click();return}
 if(a==='dist'){dist.scrollIntoView({behavior:'smooth',block:'center'});dist.focus();$('.pace').open=!dist.value;
  if(!$('#remeasure')){dist.insertAdjacentHTML('afterend','<button type="button" class="minibtn pri" id="remeasure" style="margin-top:8px">Re-measure this photo</button>');
   $('#remeasure').onclick=()=>{const v=parseFloat(dist.value);if(!(v>=1&&v<=3000)){$('#distmsg').textContent='Type a distance between 1 and 3000 m first.';return}
    $('#remeasure').remove();if(job){findDims()?.scrollIntoView({behavior:'smooth',block:'center'});measure(v)}else if(lastBlob)fullSend(v);else toast('Re-measuring needs the photo: take it again.')}}
  return}
 if(a==='sitereport'){siteReport();return}
 const k={scale:3,deposit:2,next:shot}[a];if(k!=null){if(a!=='next'){mode='survey';store.set('mode','survey')}shot=k;saveShots();drawMode();out.innerHTML='';$('#fig').style.display='none';step(1);openCamera()}}

/* decision questions: Bayes update of the model's probabilities with yes/no field answers; the next question is
   the one with the largest expected information gain. Likelihoods are expert rules of thumb (G.questions). */
function ask(r,replay){let p={};r.ranked.forEach(([c,v])=>p[c]=Math.max(v,1e-4));const Q=G.questions,asked=[],ans=r.answers||[];r.answers=ans;
 const card=document.createElement('section');card.className='card';card.id='askcard';$('#nextcard').after(card);
 const H=o=>-Object.values(o).reduce((s,v)=>s+(v>0?v*Math.log2(v):0),0);
 const upd=(o,q,y)=>{const n={};let z=0;for(const c in o){const L=q.L[c]??.5;n[c]=o[c]*(y?L:1-L);z+=n[c]}for(const c in n)n[c]/=z||1;return n};
 const gain=q=>{const py=Object.keys(p).reduce((s,c)=>s+p[c]*(q.L[c]??.5),0);return H(p)-(py*H(upd(p,q,1))+(1-py)*H(upd(p,q,0)))};
 const pick=()=>{let best=null,bg=0;for(const q of Q){if(asked.includes(q.id))continue;const g=gain(q);if(g>bg){bg=g;best=q}}return bg>=.05?best:null};
 for(const a of ans){const q=Q.find(x=>x.t===a.q);if(q){asked.push(q.id);if(a.a!=='not sure')p=upd(p,q,a.a==='yes')}}
 const draw=()=>{const rk=Object.entries(p).sort((a,b)=>b[1]-a[1]),q=!replay&&asked.length<5&&rk[0][1]<.85?pick():null;
  card.innerHTML=`<div class="kicker">Help decide · answer what you can see</div>${q?`<h2>${esc(q.t)}</h2>${q.hint?`<p class="hungr" style="margin:-6px 0 10px">${esc(q.hint)}</p>`:''}
  <div class="qa"><button type="button" class="minibtn pri" data-y="1">Yes</button><button type="button" class="minibtn pri" data-y="0">No</button><button type="button" class="minibtn" data-y="-">Not sure</button></div>`
  :`<h2>${ans.some(a=>a.a!=='not sure')?esc(tr('Your answers point to'))+' '+esc(tr(D.names[rk[0][0]]))+' ('+Math.round(rk[0][1]*100)+'%)':esc(tr('No further question would change the answer much'))}</h2>`}
  <div class="probs" style="margin-top:12px">${rk.slice(0,3).map(([c,v],k)=>`<div class="prob ${k?'':'top'}">${D.glyphs[c]}<div class="nm">${esc(D.names[c])}<div class="tr"><div class="fl" style="width:${(v*100).toFixed(1)}%"></div></div></div><div class="pc">${Math.round(v*100)}%</div></div>`).join('')}</div>
  <p class="hungr" style="margin:10px 0 0">${ans.length?esc(tr('Model result combined with your answers.')):esc(tr('Starts from the model result.'))} ${esc(tr('The answer weights are expert rules of thumb from the Hungr 2014 type descriptions, not trained on data; a geologist has the final word.'))}</p>
  ${ans.length&&!replay?'<button type="button" class="minibtn" data-y="undo" style="margin-top:8px">Undo last answer</button>':''}`;
  card.querySelectorAll('[data-y]').forEach(b=>b.onclick=()=>{const y=b.dataset.y;
   if(y==='undo'){const l=ans.pop();asked.pop();p=l.prev||p;draw();return}
   ans.push({q:q.t,a:y==='1'?'yes':y==='0'?'no':'not sure',prev:p});asked.push(q.id);if(y!=='-')p=upd(p,q,y==='1');r.fieldRanked=Object.entries(p).sort((a,b)=>b[1]-a[1]);draw();buzz(15);
   if(cur&&cur.__id)finish(cur)})};
 draw()}
function tickList(ul,cb,saved){const lis=[...ul.querySelectorAll('li')];lis.forEach((li,i)=>{li.classList.add('tick');li.setAttribute('role','checkbox');li.setAttribute('aria-checked',saved&&saved[i]?'true':'false');li.tabIndex=0;
  const tg=()=>{li.setAttribute('aria-checked',li.getAttribute('aria-checked')==='true'?'false':'true');cb()};li.onclick=tg;li.onkeydown=e=>{if(e.key===' '||e.key==='Enter'){e.preventDefault();tg()}}});return lis}
function ticks(r){const card=$('#checkcard');if(!card)return;const N=D.names,top=r.ranked[0][0],sec=r.ranked[1][0],cmp=r.status==='uncertain'&&sec!=='no_landslide'&&D.info[sec];
 card.querySelector('h2').insertAdjacentHTML('afterend','<p class="hungr" style="margin:-6px 0 10px">Tap each sign you can see on site.</p>');
 if(cmp)card.insertAdjacentHTML('beforeend',`<h2 style="margin-top:16px">${esc(D.info[sec].title)} (second choice)</h2><ul class="list" id="ck2">${D.info[sec].check.map(t=>`<li>${ICO('check')}<span>${esc(t)}</span></li>`).join('')}</ul>`);
 card.insertAdjacentHTML('beforeend','<div class="tally" id="tally" aria-live="polite"></div>');
 const ul1=card.querySelector('ul.list'),ul2=$('#ck2'),sv=r.signs||{};let l1=[],l2=[];
 const upd=()=>{const on=l=>l.map(li=>li.getAttribute('aria-checked')==='true'),a=on(l1).filter(Boolean).length,b=on(l2).filter(Boolean).length;
  r.signs={[top]:on(l1),...(cmp?{[sec]:on(l2)}:{})};
  $('#tally').innerHTML=!cmp?`<span>Signs seen:</span> <b>${a} / ${l1.length}</b>${a===0?'':' · <span>'+(a>=Math.ceil(l1.length/2)?'The field signs support this type.':'Few signs match: take another photo or ask a geologist.')+'</span>'}`:
  `<div class="cmp"><div class="${a/l1.length>b/l2.length?'lead':''}">${esc(N[top])}<b>${a}/${l1.length}</b></div><div class="${b/l2.length>a/l1.length?'lead':''}">${esc(N[sec])}<b>${b}/${l2.length}</b></div></div><p class="hungr" style="margin:8px 0 0">The type with the larger share of signs seen is the better field match.</p>`};
 l1=tickList(ul1,()=>{upd();cur&&cur.__id&&finish(cur)},sv[top]);if(ul2)l2=tickList(ul2,()=>{upd();cur&&cur.__id&&finish(cur)},sv[sec]);upd()}
function gloss(){const card=findDims();if(!card||card.querySelector('.what'))return;
 card.insertAdjacentHTML('beforeend',`<details class="what"><summary>What do these numbers mean?</summary><dl class="gloss">${Object.entries(G.glossary).slice(4).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl></details>`)}

/* ---------- finish: keep the result on the phone, survey bookkeeping, combine the site ---------- */
const strip=d=>{if(!d)return d;const o={...d};delete o.overlay;return o};
let saving=Promise.resolve();
function finish(r){saving=saving.then(()=>finishNow(r)).catch(()=>{});return saving}
async function finishNow(r){const first=!r.__id;if(first){r.__id=Date.now();curPhoto=await photoURL(lastBlob,720)}
 await saveResult(r.__id,r,curMeta||meta(),curPhoto,false,lastBlob);
 if(!first)return;gstate={flags:[],override:false,retakes:0,live:false};
 const sid=(curMeta||{}).shotId;
 if(mode==='survey'&&sid&&r.status!=='bad_photo'){let sv={};try{sv=JSON.parse(sess.get('survey')||'{}')}catch(_){}
  sv[sid]={id:sid,ranked:r.ranked,dimensions:strip(r.dimensions)};sess.set('survey',JSON.stringify(sv));done.add(sid);
  const nx=G.shots.findIndex(s=>!done.has(s.id));if(nx>=0){shot=nx;addAct(`<span>${esc(tr('Survey: next shot'))}</span> <b>${esc(G.shots[nx].t)}</b>. <span>${esc(G.shots[nx].d)}</span>`,'Take it','next')}
  else addAct('Survey complete: all 4 shots taken. The combined site result is above.','Site report','sitereport');
  saveShots();drawMode();if(Object.keys(sv).length>=2)fuse(sv)}}
function fuse(sv){const x=new XMLHttpRequest();x.open('POST','fuse?'+params({}));x.setRequestHeader('Content-Type','application/json');
 x.onload=()=>{if(x.status!==200)return;let s;try{s=JSON.parse(x.responseText)}catch(_){return}sess.set('site',JSON.stringify(s));siteCard(s)};x.send(JSON.stringify({shots:Object.values(sv)}))}
const num=v=>v>=100?Math.round(v).toLocaleString():v;
function siteCard(s){const N=D.names,t=s.type.ranked,d=s.dimensions,old=$('#sitecard');
 let h=`<section class="card" id="sitecard"><div class="kicker">Site result · ${s.n} shots combined</div><h2>${esc(N[t[0][0]])} · ${Math.round(t[0][1]*100)}%</h2>
 <div class="probs">${t.slice(0,3).map(([c,v],k)=>`<div class="prob ${k?'':'top'}">${D.glyphs[c]}<div class="nm">${esc(N[c])}<div class="tr"><div class="fl" style="width:${(v*100).toFixed(1)}%"></div></div></div><div class="pc">${Math.round(v*100)}%</div></div>`).join('')}</div>`;
 if(d&&d.available)h+=d.measures.map(m=>`<div class="drow"><span>${esc(m.name)}</span><b>${num(m.value)} ${m.unit} <small>(90%: ${num(m.range[0])}–${num(m.range[1])})</small></b></div>`).join('')+
  (d.measures.some(m=>m.between_photo_log_sd>.7)?`<p class="note">${esc(tr('The shots disagree strongly on size (more than 2x). Check that they show the same landslide, or retake the whole-slope shot with the distance typed in.'))}</p>`:'')+
  `<p class="hungr" style="margin:8px 0 0">${esc(tr('Sizes: median of the whole-slope shots, with the spread between photos added to the range.'))} (${s.size_shots.map(k=>esc((G.shots.find(x=>x.id===k)||{}).t||k)).join(', ')})</p>`;
 if(s.grains)h+=`<div class="drow"><span>Grain size D16 / D50 / D84</span><b>${s.grains.D16_m} / ${s.grains.D50_m} / ${s.grains.D84_m} m</b></div>`;
 h+=`<div style="margin-top:12px"><button type="button" class="minibtn pri" data-a="sitereport">Save site report</button></div></section>`;
 const w=document.createElement('div');w.innerHTML=h;const c=w.firstElementChild,first=out.querySelector('.card.result');if(old)old.replaceWith(c);else if(first)first.after(c);else return;c.querySelector('[data-a]').onclick=siteReport}

/* ---------- reports: a standalone HTML page (opens anywhere, prints to PDF) and a short shareable text ---------- */
function reportBar(){const b=document.createElement('div');b.className='btns';b.id='repbar';b.style.marginTop='14px';
 b.innerHTML=`<button type="button" class="btn ghost" data-a="report">Save report</button><button type="button" class="btn ghost" data-a="share">Share</button>`;
 const m=out.querySelector('.meta');m?out.insertBefore(b,m):out.appendChild(b);
 b.onclick=e=>{const a=e.target.closest('[data-a]')?.dataset.a;if(a==='report')saveReport();if(a==='share')share()}}
const RCSS='body{font:15px/1.5 system-ui,sans-serif;max-width:780px;margin:24px auto;padding:0 16px;color:#13201c}h1{font-size:1.35rem}h2{font-size:1.05rem;margin-top:22px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}img{max-width:100%;border-radius:8px;margin:8px 0}.m{color:#5c6b65;font-size:.9rem}';
function metaHTML(mt){return `<p class="m">${new Date(mt.t).toLocaleString()}${mt.gps?` · <a href="https://www.openstreetmap.org/?mlat=${mt.gps.lat}&mlon=${mt.gps.lon}#map=17/${mt.gps.lat}/${mt.gps.lon}">${mt.gps.lat.toFixed(5)}, ${mt.gps.lon.toFixed(5)}</a> (±${Math.round(mt.gps.acc)} m)`:''}${mt.weather?' · '+esc(mt.weather):''}${mt.shot?' · '+esc(mt.shot):''}${mt.set?' · '+esc(mt.set):''}${mt.dist?' · distance '+mt.dist+' m':''}</p>${mt.notes?`<p><b>Notes:</b> ${esc(mt.notes)}</p>`:''}`}
function resultHTML(r,photo){const N=D.names,d=r.dimensions;let h=photo?`<img src="${photo}" alt="photo">`:'';
 h+=`<h2>Type</h2><p><b>${r.status==='bad_photo'?'Photo not usable: retake':esc(N[r.label])}</b> · ${Math.round(r.confidence*100)}% (${esc(r.status)}${r.family?', '+esc(r.family)+' family':''})</p><table>${r.ranked.map(([c,p])=>`<tr><td>${esc(N[c])}</td><td>${Math.round(p*100)}%</td></tr>`).join('')}</table>`;
 if(r.answers&&r.answers.length)h+=`<h2>Field questions</h2><table>${r.answers.map(a=>`<tr><td>${esc(a.q)}</td><td>${a.a}</td></tr>`).join('')}</table>${r.fieldRanked?`<p>With these answers: <b>${esc(N[r.fieldRanked[0][0]])}</b> ${Math.round(r.fieldRanked[0][1]*100)}%</p>`:''}`;
 if(r.signs)h+=`<h2>Signs checked on site</h2><table>${Object.entries(r.signs).map(([c,v])=>v.map((on,i)=>`<tr><td>${esc(D.info[c].check[i])}</td><td>${on?'seen':'not seen'}</td></tr>`).join('')).join('')}</table>`;
 if(d&&d.available)h+=`<h2>Estimated dimensions</h2>${d.overlay?`<img src="${d.overlay}" alt="measured outline">`:''}<table><tr><th>Measure</th><th>Value</th><th>90% range</th></tr>${d.measures.map(m=>`<tr><td>${esc(m.name)}</td><td>${num(m.value)} ${m.unit}</td><td>${num(m.range[0])} – ${num(m.range[1])}</td></tr>`).join('')}
  ${d.size_class?`<tr><td>Size class (Fell 1994)</td><td colspan="2">${esc(d.size_class)}</td></tr>`:''}<tr><td>Slope angle</td><td colspan="2">${d.slope_angle_deg}°</td></tr></table>
  <p class="m">Scale: ${esc(d.scale_source)}. Depth trust: ${esc(d.depth_trust)}. ${esc([...(d.notes||[]),...(d.cut_edges||[]).length?['Runs off the '+d.cut_edges.join(', ')+' edge of the photo.']:[]].join(' '))}</p>`;
 return h}
const foot=m=>`<p class="m">Landslide Type Check field app${m?' ('+esc(m)+')':''}. Decision support only: confirm with a geologist or geotechnical engineer before acting.</p>`;
function download(name,html){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([html],{type:'text/html'}));a.download=name;document.body.appendChild(a);a.click();setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove()},3000);toast('Report saved: '+name)}
const fname=(k,t)=>`landslide_${k}_${new Date(t).toISOString().slice(0,16).replace(/[:T]/g,'-')}.html`;
function saveReport(){if(!cur)return;const mt=curMeta||meta();
 download(fname('observation',mt.t),`<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Landslide observation</title><style>${RCSS}</style></head><body><h1>Landslide observation</h1>${metaHTML(mt)}${resultHTML(cur,curPhoto)}${foot(cur.model)}</body></html>`)}
async function siteReport(){let s=null;try{s=JSON.parse(sess.get('site')||'null')}catch(_){}const hs=store.get('hist',[]),N=D.names,mt=meta();
 const ids=hs.filter(o=>o.site===siteId()).map(o=>o.id),rs=(await Promise.all(ids.map(id=>safe(()=>DB.get('results',id),null)))).filter(Boolean).reverse();
 let h=`<h1>Landslide site survey</h1>${metaHTML(mt)}`;
 if(s){const t=s.type.ranked,d=s.dimensions;h+=`<h2>Site result (${s.n} shots combined)</h2><p><b>${esc(N[t[0][0]])}</b> · ${Math.round(t[0][1]*100)}%</p>`;
  if(d&&d.available)h+=`<table><tr><th>Measure</th><th>Value</th><th>90% range</th></tr>${d.measures.map(m=>`<tr><td>${esc(m.name)}</td><td>${num(m.value)} ${m.unit}</td><td>${num(m.range[0])} – ${num(m.range[1])}</td></tr>`).join('')}</table><p class="m">Median of the whole-slope shots; the spread between photos is added to the range.</p>`;
  if(s.grains)h+=`<p>Grain size D16 / D50 / D84: ${s.grains.D16_m} / ${s.grains.D50_m} / ${s.grains.D84_m} m</p>`}
 for(const x of rs)h+=`<hr><h2>${esc(x.meta.shot||'Photo')}</h2>${metaHTML(x.meta)}${resultHTML(x.r,x.photo)}`;
 download(fname('site',mt.t),`<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Landslide site survey</title><style>${RCSS}</style></head><body>${h}${foot('')}</body></html>`)}
async function share(){if(!cur)return;const r=cur,mt=curMeta||meta(),N=D.names,d=r.dimensions;
 const txt=[`Landslide observation, ${new Date(mt.t).toLocaleString()}`,r.status==='bad_photo'?'Photo not usable':`${N[r.label]} (${Math.round(r.confidence*100)}%, ${r.status})`,
  ...(d&&d.available?d.measures.slice(0,4).map(m=>`${m.name}: ${num(m.value)} ${m.unit} (90%: ${num(m.range[0])}-${num(m.range[1])})`):[]),
  mt.gps?`Location: https://www.openstreetmap.org/?mlat=${mt.gps.lat}&mlon=${mt.gps.lon}`:'',mt.weather?'Weather: '+mt.weather:'',mt.notes?'Notes: '+mt.notes:'','Decision support only; confirm with a geologist.'].filter(Boolean).join('\n');
 try{const f=lastBlob?new File([lastBlob],'landslide.jpg',{type:'image/jpeg'}):null;
  if(f&&navigator.canShare&&navigator.canShare({files:[f]}))await navigator.share({text:txt,files:[f]});else if(navigator.share)await navigator.share({text:txt});
  else{await navigator.clipboard.writeText(txt);toast('Summary copied: paste it into a message or e-mail.')}}catch(e){if(e&&e.name!=='AbortError')toast('Sharing is not available here: use Save report.')}}

/* ---------- offline queue: photos wait on the phone and are sent when the connection is back ---------- */
async function queueIt(){const it={id:Date.now(),blob:lastBlob,q:params({stage:'full',d:lastDist,g:gArg(),f35:lastF35,gx:gx()}).toString(),meta:curMeta||meta(),photo:await photoURL(lastBlob,720),site:siteId()};
 const ok=await safe(async()=>{await DB.put('queue',it);return true},false);
 A.setChip('signal',ok?'Saved for later':'Offline');step(1);
 out.innerHTML=`<section class="card bad" aria-live="polite"><div class="kicker">No connection</div><h2>${ok?'Saved on the phone: it will be analysed automatically when the connection is back':'You are offline and this browser cannot store the photo'}</h2>
 <p class="hungr" style="margin:0">${ok?'Keep the app open or open it again later. You can keep photographing; each photo waits in the queue.':'Reconnect to the laptop or server and tap Try again.'}</p></section>`;
 if(ok&&mode==='survey'&&it.meta.shotId){done.add(it.meta.shotId);const nx=G.shots.findIndex(s=>!done.has(s.id));if(nx>=0)shot=nx;saveShots();drawMode()}
 gstate={flags:[],override:false,retakes:0,live:false};netUI()}
let flushing=false;
async function flush(){if(flushing||!navigator.onLine)return;const q=await safe(()=>DB.all('queue'),[]);if(!q.length){netUI();return}flushing=true;netUI();
 for(const it of q.sort((a,b)=>a.id-b.id)){try{const res=await fetch('predict?'+it.q,{method:'POST',body:it.blob,headers:{'Content-Type':it.blob.type||'application/octet-stream'}});
   if(res.status>=500)break;const r=await res.json().catch(()=>null);
   if(res.ok&&r){r.__id=it.id;await saveResult(it.id,r,it.meta,it.photo,true,it.blob,it.site);
    if(it.meta.mode==='survey'&&it.meta.shotId&&r.status!=='bad_photo'&&it.site===siteId()){let sv={};try{sv=JSON.parse(sess.get('survey')||'{}')}catch(_){}sv[it.meta.shotId]={id:it.meta.shotId,ranked:r.ranked,dimensions:strip(r.dimensions)};sess.set('survey',JSON.stringify(sv));if(Object.keys(sv).length>=2)fuse(sv)}
    toast(tr('A saved photo was analysed')+': '+(r.status==='bad_photo'?tr('retake needed'):tr(D.names[r.label]))+'. '+tr('See Recent observations.'))}
   await DB.del('queue',it.id)}catch(_){break}}
 flushing=false;netUI()}
const nb=$('#netbar');
async function netUI(){const n=(await safe(()=>DB.all('queue'),[])).length;nb.hidden=navigator.onLine&&!n;nb.classList.toggle('sync',navigator.onLine);
 nb.textContent=navigator.onLine?(n?`Sending ${n} saved photo(s)…`:''):(n?`You are offline. ${n} photo(s) waiting to be sent.`:'You are offline. New photos will be saved and sent later.')}
window.addEventListener('online',()=>{netUI();flush()});window.addEventListener('offline',netUI);netUI();setTimeout(flush,1500);setInterval(flush,60000);

/* ---------- history: every result is kept on this phone (last 20) and can be reopened ---------- */
function siteId(){let s=sess.get('siteId');if(!s){s=String(Date.now());sess.set('siteId',s)}return s}
async function saveResult(id,r,mt,photo,queued,blob,site){await safe(()=>DB.put('results',{id,r,meta:mt,photo,blob}));
 const d=r.dimensions,m=d&&d.available&&d.measures[0],old=store.get('hist',[]),prev=old.find(o=>o.id===id);
 const th=prev&&prev.th?prev.th:await urlShrink(photo,96),h=old.filter(o=>o.id!==id);
 h.unshift({id,t:mt?.t||id,label:r.label,status:r.status,conf:r.confidence,shot:mt?.shot||'',size:m?m.name.split(' (')[0]+' '+num(m.value)+' '+m.unit:'',th,queued:!!queued,site:site||(mt&&mt.mode==='survey'?siteId():'')});
 h.sort((a,b)=>b.t-a.t);const keep=h.slice(0,20);store.set('hist',keep);
 const ids=new Set(keep.map(o=>o.id));for(const x of await safe(()=>DB.all('results'),[]))if(!ids.has(x.id))safe(()=>DB.del('results',x.id));drawHist()}
function drawHist(){const h=store.get('hist',[]),el=$('#history');if(!h.length){el.innerHTML='';return}
 el.innerHTML=`<section class="card"><div class="kicker">Recent observations (this phone)</div><ul class="hist">${h.map(o=>`<li ${o.id?`data-id="${o.id}" role="button" tabindex="0"`:''}>${o.th?`<img src="${o.th}" alt="">`:'<span></span>'}<span><b>${o.status==='bad_photo'?'Retake needed':esc(D.names[o.label]||o.label)}</b>${o.status==='uncertain'?' <span>(uncertain)</span>':''}${o.queued?' <span>(sent later)</span>':''}<br>${esc([o.shot,o.size].filter(Boolean).join(' · '))}</span><span>${Math.round(o.conf*100)}%<br><small>${new Date(o.t).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</small></span></li>`).join('')}</ul></section>`;
 el.querySelectorAll('li[data-id]').forEach(li=>{const go=()=>openHist(+li.dataset.id);li.onclick=go;li.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go()}}})}
async function openHist(id){const x=await safe(()=>DB.get('results',id),null);if(!x){toast('This result is no longer stored on the phone.');return}
 cur=x.r;curMeta=x.meta;curPhoto=x.photo||'';lastBlob=x.blob||null;lastDist=x.meta&&x.meta.dist||null;job=null;
 $('#prev').src=curPhoto;$('#fig').style.display=curPhoto?'block':'none';A.render(cur);typeChip(cur);afterType(cur,true);if(cur.dimensions)afterDims(cur,true);
 $('#fig').scrollIntoView({behavior:'smooth',block:'start'})}
drawHist();A._metrics=metrics;A._judge=judge;A._fuse=fuse;A._flush=flush;A._q=()=>safe(()=>DB.all('queue'),[]);A._st=()=>({flushing,done:[...done],shot,mode,curId:cur&&cur.__id,metaShot:curMeta&&curMeta.shotId,site:!!sess.get('site')});   // exposed for tests
/* ---------- shared with the overlapping-set module (src/pano.py) ---------- */
function runSet(r,blob,g,label){lastBlob=blob;lastSrc=g?'cam':'gal';lastTilt=g?{g}:null;const dv=parseFloat(dist.value);lastDist=dv>=1&&dv<=3000?dv:null;
 curMeta=meta();curMeta.set=label||'';cur=r;job=r.job||null;const fig=$('#fig');$('#prev').src=URL.createObjectURL(blob);fig.style.display='block';
 if(r.ms)store.set('etaT',ema('etaT',r.ms,4000));A.render(r);afterType(r);if(job)measure();else finish(r)}
A._g={say,buzz,toast,tr,store,sess,step,safety,openFinder,closeFinder,openCamera,ICO,LIVE,prog,post,busy,cancelled,params,gx,runSet,fail,RX,out,vf,vid,
 ori:()=>ori,mode:()=>mode,setMode:m=>{mode=m;store.set('mode',m);drawMode()},abort:()=>xhr&&xhr.abort(),camIn,galIn,ema};
A.pano&&A.pano.init();
const demo=new URLSearchParams(location.search).get('demo');   // demo pages show the guidance too (nothing saved)
if(demo&&D.samples[demo]&&out.firstElementChild){cur=D.samples[demo];afterType(cur,true);if(cur.dimensions)afterDims(cur,true)}
})();
"""


def data():
    return {"check": photo_check(), "glossary": GLOSSARY, "shots": SHOTS, "safety": SAFETY, "tips": FIELD_TIPS,
            "tour": TOUR, "illus": ILLUS, "questions": QUESTIONS, "quiz": QUIZ, "hi": HI,
            # live steadiness: mean |grey change| between frames 0.45 s apart at 512 px. A heuristic (no
            # labelled video exists to calibrate it); it only delays the green "ready" ring, never the shutter.
            "steadyMax": 8.0,
            # stage starts as a fraction of the expected time of each request (type ~4 s, sizes ~28 s); the
            # expected times are learned on the phone from past requests (moving average)
            "stagesType": [[0, "Checking it is a usable slope photo"], [0.4, "Identifying the landslide type"]],
            "stagesDims": [[0, "Outlining the landslide (CLIPSeg, SAM, outline head)"],
                           [0.4, "Estimating depth and scale"], [0.75, "Measuring sizes and their ranges"]]}
