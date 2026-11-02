"""Overlapping photo sets on the phone (feature 2): visual guide, live camera help, overlap map.

What the junior engineer sees, in order:
  1. Mode "Overlapping set" -> choose "Turn on the spot" (wide view, joined into one picture) or "Walk along"
     (survey strip). A drawn diagram shows how to stand and move, six short rules, and a sheet of good and
     bad examples (gap, nearly the same photo, walked while turning, tilt, zoom, light, both ends included).
  2. Taking the photos. With the live camera (https), the last photo stays on screen as a see-through ghost
     that moves with the view, a dashed target line shows where its edge should sit, and a meter shows the
     overlap now (estimated on the phone from edge maps, no server needed). The shutter adds the photo and
     keeps the camera open. Without the live camera, photos are taken one by one or picked together from
     the gallery (several at once).
  3. The tray shows the photos with an instant overlap estimate between neighbours.
  4. "Check overlap" sends the set to the laptop (src/overlap.py): the overlap map shows the joined picture
     with each photo's outline and every shared area coloured (good, weak, gap, nearly the same), a bar per
     link against the ideal band, and each problem with how to fix it (add a photo between 2 and 3, drop 4,
     stand still, same tilt ...). The photos are put in left-to-right order.
  5. "Join and analyse" gives one answer for the whole set through the normal result page (type, questions,
     sizes measured on the joined picture, report), plus each photo's own answer for comparison.
The video feature (next) will add frames to the same tray.
"""
import json
import math

from overlap import CFG, MAX_IMAGES

_V = 'viewBox="0 0 240 130" class="il pil" aria-hidden="true"'
_E = 'viewBox="0 0 160 92" class="pex" aria-hidden="true"'
_SLOPE = ('<path d="M0 70L34 38L62 48L96 14L128 30L160 22V92H0Z" fill="url(#g-ground)" opacity=".75"/>'
          '<path d="M84 30Q96 26 104 44L110 70" fill="none" stroke="url(#g-signal)" stroke-width="4" stroke-linecap="round"/>')


def _frames(rects, tilt=None):
    out = []
    for k, (x, y, w, h) in enumerate(rects):
        tr = f' transform="rotate({tilt} {x + w / 2} {y + h / 2})"' if tilt and k == len(rects) - 1 else ''
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="none" stroke="currentColor" '
                   f'stroke-width="2"{tr}/><text x="{x + 4}" y="{y + 11}" font-size="9" font-weight="700" '
                   f'fill="currentColor"{tr}>{k + 1}</text>')
    return ''.join(out)


def _shade(x0, x1, y, h, cls):
    return f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{h}" class="{cls}"/>'


TURN_SVG = (
    f'<svg {_V}><path d="M6 34Q120 2 234 34L234 50Q120 18 6 50Z" fill="url(#g-ground)" opacity=".85"/>'
    '<path d="M108 24Q122 20 130 36" fill="none" stroke="url(#g-signal)" stroke-width="4" stroke-linecap="round"/>'
    + ''.join(f'<path d="M120 104L{120 + 98 * math.sin(math.radians(a - 22)):.1f} '
              f'{104 - 98 * math.cos(math.radians(a - 22)):.1f}L'
              f'{120 + 98 * math.sin(math.radians(a + 22)):.1f} '
              f'{104 - 98 * math.cos(math.radians(a + 22)):.1f}Z" class="cone"/>'
              f'<text x="{120 + 70 * math.sin(math.radians(a)) - 3:.1f}" '
              f'y="{104 - 70 * math.cos(math.radians(a)) + 3:.1f}" font-size="10" '
              f'font-weight="700" fill="currentColor">{k + 1}</text>'
              for k, a in enumerate((-45, -15, 15, 45)))
    + '<circle cx="120" cy="104" r="5" fill="currentColor"/><path d="M120 109V120M114 126L120 120L126 126" '
    'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" fill="none"/>'
    '<path d="M100 117Q120 128 140 117" fill="none" stroke="currentColor" stroke-width="1.6" marker-end="url(#parr)"/>'
    '<defs><marker id="parr" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto">'
    '<path d="M0 0L8 4L0 8Z" fill="currentColor"/></marker></defs>'
    '<text x="4" y="126" font-size="9.5" fill="currentColor" class="t-turn1"></text>'
    '<text x="236" y="126" font-size="9.5" fill="currentColor" text-anchor="end" class="t-turn2"></text></svg>')

STRIP_SVG = (
    f'<svg {_V}>' + _SLOPE.replace('M0 70L34 38', 'M0 100L34 58').replace('V92', 'V118')
    .replace('L62 48L96 14L128 30L160 22', 'L62 68L96 30L128 48L160 36L200 52L240 40')
    .replace('M84 30Q96 26 104 44L110 70', 'M86 46Q100 40 110 64L118 104')
    + _shade(70, 98, 16, 96, 'ovg') + _shade(142, 170, 16, 96, 'ovg') + _frames([(12, 16, 86, 96), (70, 16, 100, 96), (142, 16, 86, 96)])
    + '<path d="M70 10H98" stroke="currentColor" stroke-width="1.4"/><path d="M142 10H170" stroke="currentColor" stroke-width="1.4"/>'
    '<text x="84" y="8" font-size="9" text-anchor="middle" fill="currentColor" class="t-third"></text>'
    '<text x="156" y="8" font-size="9" text-anchor="middle" fill="currentColor" class="t-third"></text></svg>')

WALK_SVG = (
    f'<svg {_V}><path d="M6 20H234V40H6Z" fill="url(#g-ground)" opacity=".85"/>'
    '<path d="M100 22Q116 18 124 36" fill="none" stroke="url(#g-signal)" stroke-width="4" stroke-linecap="round"/>'
    + ''.join(f'<path d="M{x} 100L{x - 34} 40L{x + 34} 40Z" class="cone"/><circle cx="{x}" cy="102" r="4.5" fill="currentColor"/>'
              f'<text x="{x - 3}" y="118" font-size="10" font-weight="700" fill="currentColor">{k + 1}</text>'
              for k, x in enumerate((48, 96, 144, 192)))
    + '<path d="M30 108H214" stroke="currentColor" stroke-width="1.6" stroke-dasharray="3 4" marker-end="url(#parr2)"/>'
    '<defs><marker id="parr2" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto">'
    '<path d="M0 0L8 4L0 8Z" fill="currentColor"/></marker></defs>'
    '<path d="M220 40V100" stroke="currentColor" stroke-width="1.2"/><path d="M216 40H224M216 100H224" stroke="currentColor" stroke-width="1.2"/>'
    '<text x="4" y="128" font-size="9.5" fill="currentColor" class="t-walk1"></text>'
    '<text x="236" y="72" font-size="9" fill="currentColor" text-anchor="end" class="t-walk2"></text></svg>')

EXAMPLES = [
    ("good", True, f'<svg {_E}>{_SLOPE}' + _shade(46, 62, 14, 66, 'ovg') + _shade(96, 112, 14, 66, 'ovg')
     + _frames([(6, 14, 56, 66), (46, 14, 66, 66), (96, 14, 58, 66)]) + '</svg>'),
    ("gap", False, f'<svg {_E}>{_SLOPE}' + _shade(64, 92, 14, 66, 'ovr') + _frames([(8, 14, 56, 66), (92, 14, 60, 66)]) + '</svg>'),
    ("dup", False, f'<svg {_E}>{_SLOPE}' + _shade(52, 112, 14, 66, 'ovb') + _frames([(46, 14, 66, 66), (52, 16, 66, 66)]) + '</svg>'),
    ("moved", False, '<svg viewBox="0 0 160 92" class="pex" aria-hidden="true">'
     '<rect x="4" y="10" width="72" height="70" rx="3" fill="none" stroke="currentColor" stroke-width="2"/>'
     '<rect x="84" y="10" width="72" height="70" rx="3" fill="none" stroke="currentColor" stroke-width="2"/>'
     '<path d="M4 60L30 30L50 44L76 34V80H4Z" fill="url(#g-ground)" opacity=".6"/>'
     '<path d="M84 60L110 30L130 44L156 34V80H84Z" fill="url(#g-ground)" opacity=".6"/>'
     '<path d="M30 80V56M30 58Q20 50 30 40Q40 50 30 58" stroke="currentColor" stroke-width="2.5" fill="currentColor"/>'
     '<path d="M96 80V56M96 58Q86 50 96 40Q106 50 96 58" stroke="currentColor" stroke-width="2.5" fill="currentColor"/>'
     '<path d="M110 30V22M30 30V22" stroke="#d64545" stroke-width="1.5" stroke-dasharray="2 2"/></svg>'),
    ("tilt", False, f'<svg {_E}>{_SLOPE}' + _frames([(10, 16, 66, 62), (68, 16, 66, 62)], tilt=12) + '</svg>'),
    ("zoom", False, f'<svg {_E}>{_SLOPE}' + _frames([(8, 12, 72, 70), (78, 28, 40, 38)]) + '</svg>'),
    ("light", False, f'<svg {_E}>{_SLOPE}<rect x="70" y="14" width="66" height="66" fill="#000" opacity=".38"/>'
     + _frames([(10, 14, 66, 66), (70, 14, 66, 66)]) + '</svg>'),
    ("ends", True, f'<svg {_E}>{_SLOPE}<path d="M60 84H136" stroke="url(#g-signal)" stroke-width="2"/>'
     '<path d="M30 84V76M150 84V76" stroke="currentColor" stroke-width="2"/>'
     + _shade(68, 84, 10, 70, 'ovg') + _frames([(22, 10, 62, 70), (68, 10, 86, 70)]) + '</svg>'),
]

# English / Hindi text for this module (dynamic sentences are built here, so both languages read naturally)
TEXT = {
    "mode_turn": ["Turn on the spot", "एक जगह घूमकर"],
    "mode_walk": ["Walk along", "साथ-साथ चलकर"],
    "intro_turn": ["Too big for one photo? Stand in one safe spot and turn the phone. The photos are joined into one wide picture, so the whole landslide can be typed and measured.",
                   "एक फ़ोटो में नहीं आता? एक सुरक्षित जगह खड़े होकर फ़ोन घुमाएँ। फ़ोटो जोड़कर एक चौड़ी तस्वीर बनती है, जिससे पूरे भूस्खलन का प्रकार और माप मिलता है।"],
    "intro_walk": ["For a long slope along a road: walk parallel to it and take a photo every few steps. Each photo should share most of its view with the next.",
                   "सड़क के साथ लंबी ढलान के लिए: उसके समानांतर चलें और हर कुछ कदम पर फ़ोटो लें। हर फ़ोटो का अधिकांश भाग अगली फ़ोटो में भी दिखना चाहिए।"],
    "t_turn1": ["stand still", "एक जगह रुकें"], "t_turn2": ["turn the phone", "फ़ोन घुमाएँ"],
    "t_third": ["~1/3 shared", "~1/3 साझा"],
    "t_walk1": ["walk along, every 3-5 steps", "साथ चलें, हर 3-5 कदम"], "t_walk2": ["same distance", "समान दूरी"],
    "rules_turn": [["Stand in one safe spot. Turn your body and phone; do not walk between photos.",
                    "Start a little beyond one end of the landslide and finish a little beyond the other end.",
                    "Each new photo should repeat about one third of the last one (the ghost on screen helps).",
                    "Same height and tilt every time; keep the phone level. Same zoom (1x), no zooming in.",
                    "All portrait or all landscape. Take the set quickly so the light stays the same.",
                    "Avoid cars and people moving in the shared part. 3 to 6 photos are usually enough."],
                   ["एक सुरक्षित जगह खड़े रहें। शरीर और फ़ोन घुमाएँ; फ़ोटो के बीच चलें नहीं।",
                    "भूस्खलन के एक छोर से थोड़ा आगे से शुरू करें और दूसरे छोर से थोड़ा आगे तक जाएँ।",
                    "हर नई फ़ोटो में पिछली फ़ोटो का लगभग एक-तिहाई भाग दोबारा आए (स्क्रीन पर धुंधली छवि मदद करती है)।",
                    "हर बार वही ऊँचाई और झुकाव; फ़ोन सीधा रखें। वही ज़ूम (1x), ज़ूम न करें।",
                    "सभी खड़ी या सभी आड़ी फ़ोटो। जल्दी लें ताकि रोशनी न बदले।",
                    "साझा भाग में चलती गाड़ियाँ और लोग न हों। आमतौर पर 3 से 6 फ़ोटो काफ़ी हैं।"]],
    "rules_walk": [["Walk along the slope at the same distance from it (a road edge is a good line).",
                    "Face the slope square-on for every photo; do not turn to follow it.",
                    "Take a photo every 3-5 steps so each one shares 60-80 % with the next.",
                    "Same zoom and orientation all the way; keep the phone level.",
                    "Only walk where it is safe: not under the scarp, not in a channel.",
                    "Up to 8 photos; for longer slopes make two sets."],
                   ["ढलान के साथ उससे समान दूरी पर चलें (सड़क का किनारा अच्छी रेखा है)।",
                    "हर फ़ोटो में ढलान की ओर सीधे देखें; उसके पीछे मुड़ें नहीं।",
                    "हर 3-5 कदम पर फ़ोटो लें ताकि हर फ़ोटो अगली से 60-80 % साझा करे।",
                    "पूरे रास्ते वही ज़ूम और दिशा; फ़ोन सीधा रखें।",
                    "केवल सुरक्षित जगह चलें: निशान के नीचे नहीं, नाली में नहीं।",
                    "अधिकतम 8 फ़ोटो; लंबी ढलान के लिए दो सेट बनाएँ।"]],
    "examples": ["Good and bad examples", "अच्छे और गलत उदाहरण"],
    "ex_title": ["Overlapping photos: what works", "जुड़ने वाली फ़ोटो: क्या सही है"],
    "ex_good": ["Good: each photo shares about a third with the next", "सही: हर फ़ोटो अगली के साथ लगभग एक-तिहाई साझा करती है"],
    "ex_gap": ["Gap: nothing shared, the photos cannot be joined", "खाली जगह: कुछ साझा नहीं, फ़ोटो जुड़ नहीं सकतीं"],
    "ex_dup": ["Nearly the same photo: wasted, turn further", "लगभग एक जैसी फ़ोटो: बेकार, और घुमाएँ"],
    "ex_moved": ["Walked while turning: the near tree moved against the ridge (ghosts)", "घूमते हुए चले: पास का पेड़ पहाड़ी के सामने खिसक गया (दोहरी छवि)"],
    "ex_tilt": ["Tilted or aimed higher: keep the same level", "झुका या ऊपर की ओर: एक जैसा स्तर रखें"],
    "ex_zoom": ["Zoomed in: keep the same zoom", "ज़ूम किया: वही ज़ूम रखें"],
    "ex_light": ["Light changed (cloud): take the set quickly", "रोशनी बदली (बादल): जल्दी फ़ोटो लें"],
    "ex_ends": ["Both ends of the landslide plus stable ground beyond", "भूस्खलन के दोनों छोर और उनके आगे स्थिर ज़मीन"],
    "close": ["Close", "बंद करें"],
    "add_hint": ["Add 2 to 8 photos: Take photo (the camera guides each one) or From gallery (choose several at once).",
                 "2 से 8 फ़ोटो जोड़ें: फ़ोटो लें (कैमरा हर फ़ोटो में मदद करेगा) या गैलरी से (एक साथ कई चुनें)।"],
    "n_photos": ["{n} photos", "{n} फ़ोटो"],
    "check": ["Check overlap", "ओवरलैप जाँचें"], "clear": ["Clear", "हटाएँ"],
    "remove": ["Remove photo {k}", "फ़ोटो {k} हटाएँ"],
    "est_q": ["no clear overlap", "साफ़ ओवरलैप नहीं"], "est_orient": ["portrait/landscape mixed", "खड़ी/आड़ी मिली"],
    "est_note": ["Numbers between photos are a quick estimate on the phone; Check overlap measures them exactly.",
                 "फ़ोटो के बीच के अंक फ़ोन पर तुरंत अनुमान हैं; ओवरलैप जाँचें सटीक मापता है।"],
    "max": ["At most {n} photos in one set.", "एक सेट में अधिकतम {n} फ़ोटो।"],
    "added": ["Photo {k} added.", "फ़ोटो {k} जोड़ी गई।"],
    "next_turn": ["Turn {d} so about a third of the last photo stays in view, then take the next.",
                  "{d} घुमाएँ ताकि पिछली फ़ोटो का लगभग एक-तिहाई दिखता रहे, फिर अगली लें।"],
    "next_walk": ["Walk 3-5 steps along the slope, face it, and take the next.",
                  "ढलान के साथ 3-5 कदम चलें, उसकी ओर देखें, और अगली लें।"],
    "right": ["right", "दाएँ"], "left": ["left", "बाएँ"],
    "label_turn": ["Overlapping set · photo {k}", "ओवरलैप सेट · फ़ोटो {k}"],
    "hint_first_turn": ["Start just beyond one end of the landslide. Take the first photo.",
                        "भूस्खलन के एक छोर से थोड़ा आगे से शुरू करें। पहली फ़ोटो लें।"],
    "hint_first_walk": ["Start at one end, facing the slope. Take the first photo.",
                        "एक छोर से शुरू करें, ढलान की ओर देखते हुए। पहली फ़ोटो लें।"],
    "lost_turn": ["Photo {k} is out of view. Turn back {d} a little.", "फ़ोटो {k} दिख नहीं रही। थोड़ा {d} वापस घुमाएँ।"],
    "lost_walk": ["Photo {k} is out of view. Step back a little.", "फ़ोटो {k} दिख नहीं रही। थोड़ा पीछे जाएँ।"],
    "more_turn": ["Turn {d} a bit more (overlap {p} %).", "थोड़ा और {d} घुमाएँ (ओवरलैप {p} %)।"],
    "more_walk": ["Walk a few more steps (overlap {p} %).", "कुछ कदम और चलें (ओवरलैप {p} %)।"],
    "less_turn": ["Too far: turn back {d} a little (overlap {p} %).", "बहुत दूर: थोड़ा {d} वापस घुमाएँ (ओवरलैप {p} %)।"],
    "less_walk": ["Too far: step back (overlap {p} %).", "बहुत दूर: पीछे आएँ (ओवरलैप {p} %)।"],
    "good_ov": ["Good overlap ({p} %). Take the photo.", "अच्छा ओवरलैप ({p} %)। फ़ोटो लें।"],
    "tilt_same": ["Aim as high as photo 1 ({a}°, now {b}°).", "फ़ोटो 1 जितना ऊपर रखें ({a}°, अभी {b}°)।"],
    "pill_ov": ["Overlap {p} %", "ओवरलैप {p} %"], "pill_ovq": ["Overlap ?", "ओवरलैप ?"], "pill_n": ["{n} taken", "{n} ली गईं"],
    "pill_tilt": ["Same tilt", "समान झुकाव"],
    "target": ["edge of the last photo here", "पिछली फ़ोटो का किनारा यहाँ"],
    "done": ["Done ({n})", "पूरा ({n})"], "dir": ["Direction", "दिशा"],
    "checking": ["Checking how the photos overlap", "फ़ोटो का ओवरलैप जाँच रहे हैं"],
    "st_send": ["Sending the photos", "फ़ोटो भेज रहे हैं"], "st_match": ["Finding shared points between photos", "फ़ोटो के बीच साझा बिंदु खोज रहे हैं"],
    "st_join": ["Joining them into one picture", "उन्हें एक तस्वीर में जोड़ रहे हैं"],
    "need_net": ["The overlap check needs the connection to the laptop. Your photos stay here; try again when connected.",
                 "ओवरलैप जाँच के लिए लैपटॉप से कनेक्शन चाहिए। आपकी फ़ोटो यहीं रहेंगी; जुड़ने पर फिर कोशिश करें।"],
    "kick": ["Overlap check · {n} photos · {k}", "ओवरलैप जाँच · {n} फ़ोटो · {k}"],
    "v_ready": ["Ready to join", "जोड़ने के लिए तैयार"], "v_warn": ["Can be used, but check these", "उपयोग हो सकता है, पर इन्हें देखें"],
    "v_fix": ["Fix these first", "पहले इन्हें ठीक करें"],
    "sum_join": ["{a} of {n} photos join into one picture", "{n} में से {a} फ़ोटो एक तस्वीर में जुड़ती हैं"],
    "sum_walk": ["{a} of {n} photos overlap (analysed together, not joined)", "{n} में से {a} फ़ोटो ओवरलैप करती हैं (साथ में विश्लेषण, जोड़ी नहीं गईं)"],
    "sum_span": ["covers {x}° wide × {y}° high", "{x}° चौड़ा × {y}° ऊँचा भाग"],
    "legend_good": ["good overlap", "अच्छा ओवरलैप"], "legend_weak": ["weak", "कमज़ोर"], "legend_gap": ["gap", "खाली"],
    "legend_dup": ["nearly the same", "लगभग एक जैसी"],
    "map_note": ["Tap a number to highlight that photo.", "किसी अंक पर टैप करें, वह फ़ोटो उभरेगी।"],
    "links_h": ["Overlap between neighbours", "पड़ोसी फ़ोटो के बीच ओवरलैप"],
    "ideal": ["ideal {a}-{b} %", "आदर्श {a}-{b} %"],
    "s_good": ["good", "अच्छा"], "s_weak": ["weak", "कमज़ोर"], "s_gap": ["gap", "खाली"], "s_dup": ["nearly the same", "लगभग एक जैसी"],
    "moved_tag": ["camera moved", "कैमरा खिसका"],
    "fix_h": ["What to fix", "क्या ठीक करें"], "all_good": ["Nothing to fix. The photos overlap well.", "कुछ ठीक नहीं करना। फ़ोटो अच्छी तरह ओवरलैप करती हैं।"],
    "unmatched": ["Photo {a} shares nothing with the others (another place, or too far apart). Remove it, or add photos between it and the rest.",
                  "फ़ोटो {a} बाकी किसी से कुछ साझा नहीं करती (दूसरी जगह, या बहुत दूर)। इसे हटाएँ, या इसके और बाकी के बीच फ़ोटो जोड़ें।"],
    "gap": ["No overlap between photos {a} and {b}. Take one more photo between them, turning only half as far.",
            "फ़ोटो {a} और {b} के बीच कोई ओवरलैप नहीं। उनके बीच एक और फ़ोटो लें, आधा ही घुमाकर।"],
    "weak": ["Photos {a} and {b} share only {v} %. Joining may fail there; add a photo between them.",
             "फ़ोटो {a} और {b} केवल {v} % साझा करती हैं। वहाँ जोड़ना विफल हो सकता है; उनके बीच एक फ़ोटो जोड़ें।"],
    "dup": ["Photos {a} and {b} are nearly the same ({v} %). You can remove photo {b}.",
            "फ़ोटो {a} और {b} लगभग एक जैसी हैं ({v} %)। आप फ़ोटो {b} हटा सकते हैं।"],
    "moved": ["You moved between photos {a} and {b} (near and far parts shift differently). Stand in one spot and only turn.",
              "फ़ोटो {a} और {b} के बीच आप खिसके (पास और दूर के भाग अलग-अलग खिसकते हैं)। एक जगह खड़े रहकर केवल घुमाएँ।"],
    "roll": ["Photo {b} is tilted {v}° compared with photo {a}. Hold the phone level.",
             "फ़ोटो {b} फ़ोटो {a} की तुलना में {v}° झुकी है। फ़ोन सीधा रखें।"],
    "zoom": ["Photo {b} is zoomed differently from photo {a} ({v}×). Keep the same zoom (1x).",
             "फ़ोटो {b} का ज़ूम फ़ोटो {a} से अलग है ({v}×)। वही ज़ूम (1x) रखें।"],
    "light": ["The light changed between photos {a} and {b} ({v}× brighter or darker). Take the set quickly; tap and hold on the slope to lock exposure.",
              "फ़ोटो {a} और {b} के बीच रोशनी बदली ({v}× तेज़ या धीमी)। जल्दी फ़ोटो लें; एक्सपोज़र लॉक करने के लिए ढलान पर दबाकर रखें।"],
    "wide": ["The set covers {v}°; one flat picture can show at most 120°, so the outer parts were cut. Split it into two sets.",
             "सेट {v}° कवर करता है; एक सपाट तस्वीर अधिकतम 120° दिखा सकती है, इसलिए बाहरी भाग कटे। इसे दो सेट में बाँटें।"],
    "spacing": ["Spacing is uneven (overlap from {lo} % to {hi} %). Keep the same number of steps between photos.",
                "दूरी असमान है (ओवरलैप {lo} % से {hi} %)। फ़ोटो के बीच समान कदम रखें।"],
    "turned": ["These were taken from one spot (turning), so they were joined into one wide picture.",
               "ये एक ही जगह से (घुमाकर) ली गईं, इसलिए एक चौड़ी तस्वीर में जोड़ी गईं।"],
    "not_joined": ["These photos show parallax (you walked and the slope has depth), so they are analysed together instead of joined.",
                   "इन फ़ोटो में पैरेलैक्स है (आप चले और ढलान में गहराई है), इसलिए इन्हें जोड़ने के बजाय साथ में विश्लेषित किया गया।"],
    "order": ["The photos were put in order from {first} to {last}.", "फ़ोटो को {first} से {last} क्रम में रखा गया।"],
    "first_l": ["left", "बाएँ"], "last_r": ["right", "दाएँ"], "first_t": ["top", "ऊपर"], "last_b": ["bottom", "नीचे"],
    "join": ["Join and analyse", "जोड़ें और विश्लेषण करें"], "together": ["Analyse together", "साथ में विश्लेषण करें"],
    "drop": ["Remove the suggested photos", "सुझाई गई फ़ोटो हटाएँ"], "addmore": ["Add a photo", "फ़ोटो जोड़ें"],
    "restart": ["Start the set again", "सेट फिर से शुरू करें"],
    "joining": ["Analysing the photo set", "फ़ोटो सेट का विश्लेषण"],
    "st_type": ["Identifying the landslide type in each photo and the joined picture", "हर फ़ोटो और जुड़ी तस्वीर में भूस्खलन का प्रकार पहचान रहे हैं"],
    "set_label_j": ["Overlapping set: {n} photos joined", "ओवरलैप सेट: {n} फ़ोटो जोड़ी गईं"],
    "set_label_w": ["Overlapping set: {n} photos analysed together", "ओवरलैप सेट: {n} फ़ोटो साथ में विश्लेषित"],
    "per_h": ["Each photo on its own", "हर फ़ोटो अलग से"],
    "per_note": ["The answer above combines the joined picture with every single photo. Agreement here makes it more trustworthy.",
                 "ऊपर का उत्तर जुड़ी तस्वीर और हर अकेली फ़ोटो को मिलाता है। यहाँ सहमति उसे अधिक भरोसेमंद बनाती है।"],
    "per_joined": ["Joined picture", "जुड़ी तस्वीर"], "per_photo": ["Photo {k}", "फ़ोटो {k}"],
    "agree": ["{a} of {n} agree with the combined answer", "{n} में से {a} संयुक्त उत्तर से सहमत"],
    "resend": ["The laptop no longer has these photos; sending them again.", "लैपटॉप पर ये फ़ोटो अब नहीं हैं; फिर से भेज रहे हैं।"],
}


CSS = """
#pano[hidden],.pgide[hidden],.tips[hidden]{display:none}
.pgide{margin:0 0 12px}.pgide .pk{margin-bottom:10px}
.pil{max-height:170px}.pil .cone{fill:var(--teal-400);fill-opacity:.16;stroke:var(--teal-400);stroke-width:1.2}
.ovg{fill:var(--teal-400);fill-opacity:.38}.ovr{fill:var(--ruby);fill-opacity:.28}.ovb{fill:#5b8def;fill-opacity:.3}
.prules{margin:6px 0 10px;padding-left:20px;font-size:.88rem;color:var(--ink-2)}.prules li{margin:3px 0}
.tray{display:flex;align-items:center;gap:0;overflow-x:auto;padding:4px 2px 8px;margin:10px 0 4px;scroll-snap-type:x proximity}
.tray:empty{display:none}
.pit{position:relative;flex:0 0 auto;width:84px;height:64px;border-radius:9px;overflow:hidden;border:1.5px solid var(--line);scroll-snap-align:start;background:var(--surface-2)}
.pit img{width:100%;height:100%;object-fit:cover;display:block}
.pit .pn{position:absolute;left:4px;top:4px;min-width:20px;height:20px;border-radius:10px;background:rgba(0,0,0,.6);color:#fff;font:700 .72rem/20px inherit;text-align:center}
.pit .px{position:absolute;right:3px;top:3px;width:24px;height:24px;border-radius:50%;border:0;background:rgba(0,0,0,.6);color:#fff;font:700 .9rem/1 inherit;cursor:pointer}
.pit.cam::after{content:"";position:absolute;right:5px;bottom:5px;width:7px;height:7px;border-radius:50%;background:#5fd1a8}
.plink{flex:0 0 auto;min-width:46px;margin:0 2px;text-align:center;font-size:.7rem;font-weight:700;color:var(--mute)}
.plink i{display:block;height:4px;border-radius:2px;margin:3px 4px 0;background:var(--line)}
.plink.good{color:var(--teal-600)}.plink.good i{background:var(--teal-400)}
.plink.weak,.plink.dup{color:#b07a00}.plink.weak i{background:var(--amber)}.plink.dup i{background:#5b8def}
.plink.gap{color:var(--ruby)}.plink.gap i{background:var(--ruby)}
.pstat{font-size:.82rem;color:var(--mute);margin:2px 0 10px}
.pex{width:100%;height:auto;color:var(--ink-2)}
.exgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0}
.excard{border:1.5px solid var(--line);border-radius:12px;padding:6px 8px 8px;font-size:.78rem;line-height:1.3;color:var(--ink-2)}
.excard b{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;color:#fff;font-size:.7rem;margin-right:4px;vertical-align:-3px}
.excard.ok{border-color:var(--teal-400)}.excard.ok b{background:var(--teal-600)}.excard.no b{background:var(--ruby)}
.omap{position:relative;border-radius:12px;overflow:hidden;margin:8px 0 6px;background:#000}
.omap img{display:block;width:100%;height:auto}.omap svg{position:absolute;inset:0;width:100%;height:100%}
.omap .q{fill:none;stroke:#fff;stroke-width:2.2;vector-effect:non-scaling-stroke;opacity:.9}
.omap .q.sel{stroke:#ffd166;stroke-width:3.5}.omap .q.dim{opacity:.25}
.omap .sh{fill-opacity:.42;stroke:none}.omap .sh.good{fill:#2bb38a}.omap .sh.weak{fill:#f2a900}.omap .sh.gap{fill:#e04848}.omap .sh.dup{fill:#5b8def}
.omap .nb circle{fill:rgba(0,0,0,.72);stroke:#fff;stroke-width:1.5;vector-effect:non-scaling-stroke}.omap .nb text{fill:#fff;font-weight:700;text-anchor:middle;dominant-baseline:central}
.omap .nb{cursor:pointer}
.olegend{display:flex;flex-wrap:wrap;gap:10px;font-size:.74rem;color:var(--mute);margin:0 0 8px}.olegend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.ostrip{display:flex;align-items:center;overflow-x:auto;gap:0;padding:4px 0 8px}
.olinks{list-style:none;margin:6px 0 12px;padding:0;display:grid;gap:8px}
.olinks li{display:grid;grid-template-columns:62px 1fr 64px;align-items:center;gap:8px;font-size:.84rem}
.obar{position:relative;height:10px;border-radius:5px;background:var(--surface-2);border:1px solid var(--line);overflow:hidden}
.obar .z{position:absolute;top:0;bottom:0;background:var(--tint-ground);border-left:1px dashed var(--teal-400);border-right:1px dashed var(--teal-400)}
.obar .v{position:absolute;top:2px;bottom:2px;left:0;border-radius:3px}
.obar .v.good{background:var(--teal-400)}.obar .v.weak{background:var(--amber)}.obar .v.gap{background:var(--ruby)}.obar .v.dup{background:#5b8def}
.olinks .st{font-size:.74rem;font-weight:700;text-align:right}.olinks .st.good{color:var(--teal-600)}.olinks .st.weak{color:#b07a00}.olinks .st.gap{color:var(--ruby)}.olinks .st.dup{color:#3d6fd6}
.oiss{list-style:none;margin:6px 0 12px;padding:0;display:grid;gap:8px}
.oiss li{display:grid;grid-template-columns:22px 1fr;gap:8px;font-size:.88rem;padding:8px 10px;border-radius:10px;background:var(--surface-2)}
.oiss li.bad{box-shadow:inset 3px 0 0 var(--ruby)}.oiss li.warn{box-shadow:inset 3px 0 0 var(--amber)}.oiss li.info{box-shadow:inset 3px 0 0 #5b8def}
.oiss li.hl{outline:2px solid #ffd166}
.vbadge{display:inline-block;padding:3px 10px;border-radius:99px;font-size:.78rem;font-weight:700;margin-bottom:4px}
.vbadge.ready{background:var(--tint-ground);color:var(--teal-800)}.vbadge.warn{background:var(--tint-signal);color:#8a5a00}.vbadge.fix{background:#fde3e3;color:#a12a2a}
.perset{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:8px;margin:8px 0}
.perset div{border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:.74rem}
.perset img{width:100%;height:58px;object-fit:cover;display:block}.perset p{margin:4px 6px 6px}.perset .agree{color:var(--teal-600)}.perset .dis{color:#b07a00}
.vfg[data-frame=pano] .band{display:none}
.pghost{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.pghost img{position:absolute;opacity:.42;transition:transform .35s ease-out,opacity .3s;filter:saturate(.6);outline:2px dashed rgba(255,209,102,.9)}
.pghost img.lost{opacity:.18}
.ptarget{position:absolute;top:8%;bottom:22%;width:0;border-left:2px dashed #ffd166;pointer-events:none}
.ptarget span{position:absolute;top:-2px;left:6px;white-space:nowrap;font-size:.7rem;font-weight:700;color:#ffd166;background:rgba(0,0,0,.5);padding:2px 6px;border-radius:6px}
.ptarget.r span{left:auto;right:6px}
.pmeter{position:absolute;left:12%;right:12%;bottom:calc(150px + env(safe-area-inset-bottom));height:12px;border-radius:6px;background:rgba(255,255,255,.18);overflow:hidden;pointer-events:none}
.pmeter .z{position:absolute;top:0;bottom:0;background:rgba(95,209,168,.35);border-left:2px solid #5fd1a8;border-right:2px solid #5fd1a8}
.pmeter .v{position:absolute;top:2px;bottom:2px;width:4px;margin-left:-2px;border-radius:2px;background:#fff;transition:left .3s}
.pstrip{position:absolute;left:0;right:0;bottom:calc(104px + env(safe-area-inset-bottom));display:flex;gap:6px;justify-content:center;pointer-events:none}
.pstrip img{width:44px;height:33px;object-fit:cover;border-radius:5px;border:1.5px solid #fff;opacity:.9}
.vf.pano .vfr{display:none}
.vfb .pdone,.vfb .pdir{grid-row:1;border:0;background:rgba(0,0,0,.45);color:#fff;border-radius:99px;padding:9px 12px;font:700 .8rem inherit;cursor:pointer}
.vfb .pdone{grid-column:3;justify-self:end;background:#2bb38a}.vfb .pdir{grid-column:1;justify-self:start;margin-left:96px}
.pflash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none;transition:opacity .25s}.pflash.on{opacity:.55;transition:none}
"""

JS = r"""
(()=>{const A=window.App,$=s=>document.querySelector(s);let g=null,D=null,P=null,esc=null,want=false,ready=false;
const S={kind:'turn',items:[],rep:null,est:[],dir:0};let nid=1,lang='en';
const L=(k,o={})=>{const t=(P&&P.text[k])||[k,k];let s=t[lang==='hi'?1:0];if(Array.isArray(s))return s;for(const [a,b] of Object.entries(o))s=s.split('{'+a+'}').join(b);return s};
const pct=v=>Math.round(v*100);

/* ---------- quick overlap estimate on the phone: edge maps + normalised cross-correlation over shifts ---------- */
const SW=80;
function sig(src){const w0=src.videoWidth||src.width,h0=src.videoHeight||src.height;if(!w0||!h0)return null;const W=SW,H=Math.max(12,Math.round(SW*h0/w0));
 const c=document.createElement('canvas');c.width=W;c.height=H;const x=c.getContext('2d',{willReadFrequently:true});x.drawImage(src,0,0,W,H);
 const d=x.getImageData(0,0,W,H).data,l=new Float32Array(W*H),m=new Float32Array(W*H);for(let i=0;i<W*H;i++)l[i]=.299*d[4*i]+.587*d[4*i+1]+.114*d[4*i+2];
 for(let y=1;y<H-1;y++)for(let k=1;k<W-1;k++){const i=y*W+k,a=l[i+1]-l[i-1],b=l[i+W]-l[i-W];m[i]=Math.sqrt(a*a+b*b)}return {W,H,m}}
function ncc(a,b,W,H,dx,dy,minA,st){const x0=Math.max(0,-dx),x1=Math.min(W,W-dx),y0=Math.max(0,-dy),y1=Math.min(H,H-dy);
 if((x1-x0)*(y1-y0)<minA)return -2;let sa=0,sb=0,saa=0,sbb=0,sab=0,n=0;
 for(let y=y0;y<y1;y+=st){const ra=(y+dy)*W+dx,rb=y*W;for(let x=x0;x<x1;x+=st){const p=a[ra+x],q=b[rb+x];sa+=p;sb+=q;saa+=p*p;sbb+=q*q;sab+=p*q;n++}}
 const va=saa-sa*sa/n,vb=sbb-sb*sb/n;return va>1e-6&&vb>1e-6?(sab-sa*sb/n)/Math.sqrt(va*vb):-2}
// b(x, y) ~ a(x + dx, y + dy): dx > 0 when the camera turned (or walked) right between a and b
function match(a,b){if(!a||!b)return null;if(a.W!==b.W||Math.abs(a.H-b.H)>2)return {orient:true};const W=a.W,H=Math.min(a.H,b.H),minA=.1*W*H,ry=Math.round(H*.3),rx=Math.round(W*.92);
 let s0=-2,bx=0,by=0;for(let dy=-ry;dy<=ry;dy+=2)for(let dx=-rx;dx<=rx;dx+=2){let s=ncc(a.m,b.m,W,H,dx,dy,minA,2);if(s<=-2)continue;
  const ov=(W-Math.abs(dx))*(H-Math.abs(dy))/(W*H);s*=.88+.12*ov;if(s>s0){s0=s;bx=dx;by=dy}}
 let s1=-2,fx=bx,fy=by;for(let dy=by-2;dy<=by+2;dy++)for(let dx=bx-2;dx<=bx+2;dx++){const s=ncc(a.m,b.m,W,H,dx,dy,minA,1);if(s>s1){s1=s;fx=dx;fy=dy}}
 return {dx:fx/W,dy:fy/H,ov:(W-Math.abs(fx))*(H-Math.abs(fy))/(W*H),score:s1}}
function band(ov){const B=P.bands[S.kind];return ov<B.gap?'gap':ov<B.weak?'weak':ov>B.dup?'dup':'good'}

/* ---------- EXIF focal length (35 mm equivalent), read before the photo is shrunk ---------- */
async function f35Of(f){try{const b=new DataView(await f.slice(0,262144).arrayBuffer());if(b.getUint16(0)!==0xFFD8)return null;let o=2;
 while(o+4<b.byteLength){const mk=b.getUint16(o),len=b.getUint16(o+2);
  if(mk===0xFFE1&&b.getUint32(o+4)===0x45786966){const t=o+10,le=b.getUint16(t)===0x4949,u16=p=>b.getUint16(p,le),u32=p=>b.getUint32(p,le);
   const ifd=(p,tag)=>{const n=u16(p);for(let i=0;i<n;i++){const e=p+2+i*12;if(u16(e)===tag)return e}return -1};
   const e0=ifd(t+u32(t+4),0x8769);if(e0<0)return null;const e1=ifd(t+u32(e0+8),0xA405);if(e1<0)return null;const v=u16(e1+8);return v>0?v:null}
  if((mk&0xFF00)!==0xFF00)break;o+=2+len}}catch(_){}return null}
async function bitmapOf(f){try{return await createImageBitmap(f,{imageOrientation:'from-image'})}catch(_){return await new Promise((res,rej)=>{const i=new Image();i.onload=()=>res(i);i.onerror=rej;i.src=URL.createObjectURL(f)})}}
function thumbOf(src,max){const w0=src.videoWidth||src.width,h0=src.videoHeight||src.height,s=Math.min(1,max/Math.max(w0,h0)),c=document.createElement('canvas');
 c.width=Math.round(w0*s);c.height=Math.round(h0*s);c.getContext('2d').drawImage(src,0,0,c.width,c.height);return c.toDataURL('image/jpeg',.7)}
const toB64=b=>new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result);f.readAsDataURL(b)});
async function urlBlob(u){const s=atob(u.split(',')[1]),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return new Blob([a],{type:'image/jpeg'})}

/* ---------- the set ---------- */
async function add(file,t,src,live){if(S.items.length>=P.max){g.toast(L('max',{n:P.max}));return null}
 const it={id:nid++,src,g:t?t.g:null,elev:t?t.elev:null,f35:null,blob:null,thumb:'',sig:null,ready:null};S.items.push(it);S.rep=null;
 if(live){it.sig=sig(live);it.thumb=thumbOf(live,240)}
 it.ready=(async()=>{it.f35=await f35Of(file);try{const bmp=await bitmapOf(file);if(!it.sig)it.sig=sig(bmp);if(!it.thumb)it.thumb=thumbOf(bmp,240);bmp.close&&bmp.close()}catch(_){}
  it.blob=await A.shrink(file)})();
 await it.ready;estimate();render();return it}
function estimate(){S.est=[];for(let k=0;k+1<S.items.length;k++)S.est.push(match(S.items[k].sig,S.items[k+1].sig));
 const e=S.est.find(x=>x&&!x.orient&&x.score>.3);if(e&&!S.dir)S.dir=Math.sign(e.dx)||1}
function remove(id){S.items=S.items.filter(i=>i.id!==id);S.rep=null;estimate();render();const c=$('#ovcard');if(c)c.remove()}
function clearSet(){S.items=[];S.rep=null;S.est=[];S.dir=0;render();$('#ovcard')?.remove()}
async function onPicks(files,cam){if(g.mode()!=='pano')return false;g.step(1);const room=Math.max(0,P.max-S.items.length);
 if(files.length>room)g.toast(L('max',{n:P.max}));
 for(const f of files.slice(0,room))await add(f,cam?g.ori():null,cam?'cam':'gal');
 if(cam&&S.items.length<P.max){const k=S.items.length;g.toast(L('added',{k})+' '+(S.kind==='turn'?L('next_turn',{d:L(S.dir<0?'left':'right')}):L('next_walk')))}
 $('#pano .tray')?.scrollIntoView({behavior:'smooth',block:'nearest'});return true}

/* ---------- panel: guide, rules, examples, tray ---------- */
function build(){const card=$('#capH').closest('.card');
 card.querySelector('#shots').insertAdjacentHTML('afterend',`<div class="pgide" hidden><div class="seg pk" role="group" aria-label="${esc(L('mode_turn'))} / ${esc(L('mode_walk'))}">
  <button type="button" data-k="turn">${esc(L('mode_turn'))}</button><button type="button" data-k="walk">${esc(L('mode_walk'))}</button></div>
  <p class="pintro hungr" style="margin:0 0 6px"></p><div class="pdiag"></div><ol class="prules"></ol>
  <button type="button" class="minibtn" data-a="examples">${esc(L('examples'))}</button></div>`);
 card.querySelector('.btns').insertAdjacentHTML('afterend',`<div id="pano" hidden><div class="tray" aria-live="polite"></div><p class="pstat"></p>
  <div class="row" style="margin:0 0 6px"><button type="button" class="btn primary" data-a="check" disabled>${esc(L('check'))}</button><button type="button" class="minibtn" data-a="clear">${esc(L('clear'))}</button></div></div>`);
 const gd=card.querySelector('.pgide');
 gd.addEventListener('click',e=>{const k=e.target.closest('[data-k]')?.dataset.k;if(k){S.kind=k;g.store.set('pkind',k);S.rep=null;$('#ovcard')?.remove();drawGuide();render()}
  if(e.target.dataset.a==='examples')examples()});
 $('#pano').addEventListener('click',e=>{const a=e.target.closest('[data-a]')?.dataset.a,x=e.target.closest('.px');
  if(x){remove(+x.dataset.id);return}if(a==='check')check();if(a==='clear')clearSet()})}
function drawGuide(){const gd=$('.pgide');gd.querySelectorAll('[data-k]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.k===S.kind));
 gd.querySelector('.pintro').textContent=L('intro_'+S.kind);
 gd.querySelector('.pdiag').innerHTML=S.kind==='turn'?P.svg.turn+P.svg.strip:P.svg.walk;
 gd.querySelectorAll('.pdiag text[class^=t-]').forEach(t=>{t.textContent=L(t.getAttribute('class').replace('-','_'))});
 gd.querySelector('.prules').innerHTML=L('rules_'+S.kind).map(r=>`<li>${esc(r)}</li>`).join('')}
function examples(){const m=$('#mTour');m.firstElementChild.innerHTML=`<div class="row" style="margin:0"><h2 id="tourT" style="margin:0">${esc(L('ex_title'))}</h2><button type="button" class="minibtn" data-a="x" data-f>${esc(L('close'))}</button></div>
 <div class="exgrid">${P.examples.map(([k,ok,svg])=>`<div class="excard ${ok?'ok':'no'}">${svg}<b>${ok?'&#10003;':'&#10005;'}</b>${esc(L('ex_'+k))}</div>`).join('')}</div>`;
 m.hidden=false;m.querySelector('[data-f]').focus();m.onclick=e=>{if(e.target===m||e.target.dataset.a==='x')m.hidden=true}}
function render(){const tr=$('#pano .tray');if(!tr)return;
 tr.innerHTML=S.items.map((it,k)=>{let link='';if(k<S.items.length-1){const e=S.est[k];
   link=!e?'<div class="plink"><i></i></div>':e.orient?`<div class="plink gap" title="${esc(L('est_orient'))}">&ne;<i></i></div>`:e.score<.3?`<div class="plink" title="${esc(L('est_q'))}">?<i></i></div>`:`<div class="plink ${band(e.ov)}">~${pct(e.ov)}%<i></i></div>`}
  return `<div class="pit ${it.src==='cam'?'cam':''}"><img src="${it.thumb}" alt=""><span class="pn">${k+1}</span><button type="button" class="px" data-id="${it.id}" aria-label="${esc(L('remove',{k:k+1}))}">&times;</button></div>${link}`}).join('');
 const n=S.items.length;$('#pano .pstat').textContent=n?L('n_photos',{n})+' · '+L('est_note'):L('add_hint');
 $('#pano [data-a=check]').disabled=n<2}
function apply(){const on=want;$('.pgide').hidden=!on;$('#pano').hidden=!on;g.galIn.multiple=on;
 const tips=$('#capH').closest('.card').querySelector('.tips');if(tips)tips.hidden=on;if(on){drawGuide();render()}}

/* ---------- live camera: ghost of the last photo, target line, overlap meter ---------- */
let ghost=null,tgt=null,meter=null,strip=null,flash=null,last=null;
function finder(on){const vf=g.vf;vf.classList.toggle('pano',on);vf.querySelectorAll('.pghost,.ptarget,.pmeter,.pstrip,.pflash,.pdone,.pdir').forEach(e=>e.remove());last=null;if(!on){render();return}
 vf.querySelector('video').insertAdjacentHTML('afterend','<div class="pghost"><img alt=""></div>');
 vf.insertAdjacentHTML('beforeend',`<div class="ptarget"><span>${esc(L('target'))}</span></div><div class="pmeter"><div class="z"></div><div class="v"></div></div><div class="pstrip"></div><div class="pflash"></div>`);
 vf.querySelector('.vfb').insertAdjacentHTML('beforeend',`<button type="button" class="pdir" aria-label="${esc(L('dir'))}"></button><button type="button" class="pdone"></button>`);
 ghost=vf.querySelector('.pghost img');tgt=vf.querySelector('.ptarget');meter=vf.querySelector('.pmeter');strip=vf.querySelector('.pstrip');flash=vf.querySelector('.pflash');
 vf.querySelector('.pdone').onclick=()=>{g.closeFinder();$('#pano .tray')?.scrollIntoView({behavior:'smooth',block:'nearest'})};
 vf.querySelector('.pdir').onclick=()=>{S.dir=(S.dir||1)*-1;layout()};
 const B=P.bands[S.kind];const z=meter.querySelector('.z');z.style.left=B.ideal[0]*100+'%';z.style.width=(B.ideal[1]-B.ideal[0])*100+'%';layout()}
function box(){const v=g.vid,bw=v.clientWidth,bh=v.clientHeight,vw=v.videoWidth||bw,vh=v.videoHeight||bh,s=Math.max(bw/vw,bh/vh);return {bw,bh,dw:vw*s,dh:vh*s}}
function layout(est){if(!ghost)return;const n=S.items.length,prev=S.items[n-1],vf=g.vf,dir=S.dir||1;
 vf.querySelector('.pdone').textContent=L('done',{n});vf.querySelector('.pdone').hidden=!n;vf.querySelector('.pdir').textContent=dir<0?'←':'→';vf.querySelector('.pdir').hidden=!n;
 strip.innerHTML=S.items.slice(-6).map(i=>`<img src="${i.thumb}" alt="">`).join('');
 tgt.hidden=meter.hidden=!n;if(!n){ghost.hidden=true;return}
 const b=box(),B=P.bands[S.kind],mid=(B.ideal[0]+B.ideal[1])/2;ghost.hidden=false;if(ghost.dataset.id!=String(prev.id)){ghost.src=prev.thumb;ghost.dataset.id=prev.id}
 ghost.style.width=b.dw+'px';ghost.style.height=b.dh+'px';ghost.style.left=(b.bw-b.dw)/2+'px';ghost.style.top=(b.bh-b.dh)/2+'px';
 // where the last photo's trailing edge should sit when the overlap is ideal
 const ex=dir>0?mid:1-mid;tgt.style.left=((b.bw-b.dw)/2+ex*b.dw)+'px';tgt.classList.toggle('r',dir<0);
 const e=est&&!est.orient&&est.score>.3?est:null,dx=e?e.dx:dir*(1-mid),dy=e?e.dy:0;
 ghost.style.transform=`translate(${-dx*b.dw}px,${-dy*b.dh}px)`;ghost.classList.toggle('lost',!e);
 meter.querySelector('.v').style.left=(e?Math.min(1,Math.max(0,e.ov)):0)*100+'%';meter.querySelector('.v').style.opacity=e?1:.3}
function tick(vid,t){const n=S.items.length,first=S.items[0],turn=S.kind==='turn',dir=S.dir||1,dw=L(dir<0?'left':'right'),back=L(dir<0?'right':'left');
 const out={bad:[],pills:[`<span class="ok">${esc(L('pill_n',{n}))}</span>`],good:null};
 if(!n){out.good=L(turn?'hint_first_turn':'hint_first_walk');layout();return out}
 const e=match(S.items[n-1].sig,sig(vid));last=e;layout(e);
 if(!e||e.orient||e.score<.3){out.pills.unshift(`<span class="warn">${esc(L('pill_ovq'))}</span>`);out.bad.push(['Overlap',turn?L('lost_turn',{k:n,d:back}):L('lost_walk',{k:n})])}
 else{const B=P.bands[S.kind],p=pct(e.ov),ok=e.ov>=B.ideal[0]&&e.ov<=B.ideal[1]+.08;out.pills.unshift(`<span class="${ok?'ok':'warn'}">${esc(L('pill_ov',{p}))}</span>`);
  if(e.ov>B.ideal[1]+.08)out.bad.push(['Overlap',turn?L('more_turn',{d:dw,p}):L('more_walk',{p})]);
  else if(e.ov<B.ideal[0])out.bad.push(['Overlap',turn?L('less_turn',{d:back,p}):L('less_walk',{p})]);
  else out.good=L('good_ov',{p})}
 if(first&&first.elev!=null&&t&&Math.abs(t.elev-first.elev)>6){out.pills.push(`<span class="warn">${esc(L('pill_tilt'))}</span>`);out.bad.push(['Same tilt',L('tilt_same',{a:Math.round(first.elev),b:Math.round(t.elev)})])}
 return out}
async function shot(blob,t,vid){flash.classList.add('on');setTimeout(()=>flash.classList.remove('on'),60);g.buzz([20,40,20]);
 const e=last;const it=await add(blob,t,'cam',vid);if(!it)return;const n=S.items.length;
 if(e&&!e.orient&&e.score>.3&&n===2&&!S.dir)S.dir=Math.sign(e.dx)||1;layout();
 g.say(L('added',{k:n})+' '+(n<P.max?(S.kind==='turn'?L('next_turn',{d:L((S.dir||1)<0?'left':'right')}):L('next_walk')):''));
 g.vf.querySelector('.vfs').textContent=L('label_turn',{k:n+1});if(n>=P.max)g.closeFinder()}

/* ---------- overlap check (server) and the overlap map ---------- */
function check(){if(S.items.length<2)return;if(!navigator.onLine){g.toast(L('need_net'));return}
 const out=g.out;$('#ovcard')?.remove();out.innerHTML='';g.step(2);g.busy(L('checking'));
 const pg=g.prog($('#pg'),[[0,L('st_match')],[.6,L('st_join')]],g.store.get('etaO',3000)+S.items.length*400,true);
 (async()=>{await Promise.all(S.items.map(i=>i.ready));const body={kind:S.kind,images:await Promise.all(S.items.map(i=>toB64(i.blob))),f35:S.items.map(i=>i.f35)};
  g.post('overlap?'+g.params({kind:S.kind}),new Blob([JSON.stringify(body)],{type:'application/json'}),pg,{
   ok:r=>{if(r.ms)g.store.set('etaO',g.ema('etaO',r.ms,3000));reorder(r);S.rep=r;render();report(r)},
   err:(m,s)=>s===-2?g.cancelled():pfail(m,check)})})()}
function pfail(m,again){A.fail(m);const c=g.out.firstElementChild;g.say(m);g.buzz([80,60,80]);
 c.insertAdjacentHTML('beforeend',`<div class="btns" style="margin-top:12px"><button type="button" class="btn primary" data-a="again">${esc(L('check'))}</button></div>`);
 c.querySelector('[data-a=again]').textContent=again===check?L('check'):L('join');c.querySelector('[data-a=again]').onclick=()=>again()}
function reorder(r){const items=r.order.map(i=>S.items[i]);const map={};r.order.forEach((i,k)=>{map[i]=k});r._map=map;
 const fix=x=>map[x];r.issues.forEach(x=>{['a','b'].forEach(k=>{if(x[k]!=null)x[k]=fix(x[k])});if(x.group)x.group=x.group.map(fix)});
 r.links.forEach(l=>{l.a=fix(l.a);l.b=fix(l.b);if(l.a>l.b){[l.a,l.b]=[l.b,l.a];l.share&&l.share.reverse()}});r.links.sort((p,q)=>p.a-q.a);
 r.main=r.main.map(fix);r.ref=fix(r.ref);
 if(r.preview){const q={},sh={};for(const [i,v] of Object.entries(r.preview.quads))q[fix(+i)]=v;for(const [k,v] of Object.entries(r.preview.shared)){const [a,b]=k.split('-').map(Number);sh[[fix(a),fix(b)].sort((x,y)=>x-y).join('-')]=v}r.preview.quads=q;r.preview.shared=sh}
 r.order=r.order.map((_,k)=>k);S.items=items;estimate()}
function issueText(x){const n=v=>v+1;const o={a:x.a!=null?n(x.a):'',b:x.b!=null?n(x.b):'',v:x.val==null?'':Array.isArray(x.val)?'':(x.code==='weak'||x.code==='dup'?pct(x.val):x.val)};
 if(x.code==='spacing'){o.lo=pct(x.val[0]);o.hi=pct(x.val[1])}
 if(x.code==='order'){const vert=S.rep&&S.rep.preview&&S.rep.preview.h>S.rep.preview.w*1.2;o.first=L(vert?'first_t':'first_l');o.last=L(vert?'last_b':'last_r')}
 return L(x.code,o)}
const SEVI={bad:'alert',warn:'alert',info:'check'};
function report(r){const out=g.out,n=S.items.length,ICO=g.ICO,B=r.bands,vt={ready:L('v_ready'),warn:L('v_warn'),fix:L('v_fix')}[r.verdict];
 const kname=L(r.joined||r.kind==='turn'?'mode_turn':'mode_walk');
 let h=`<section class="card" id="ovcard"><div class="kicker">${esc(L('kick',{n,k:kname}))}</div><span class="vbadge ${r.verdict}">${esc(vt)}</span>
 <h2 style="margin:4px 0 2px">${esc(r.joined?L('sum_join',{a:r.main.length,n}):L('sum_walk',{a:r.main.length,n}))}</h2>
 ${r.span_deg?`<p class="hungr" style="margin:0 0 6px">${esc(L('sum_span',{x:Math.round(r.span_deg[0]),y:Math.round(r.span_deg[1])}))}</p>`:''}`;
 if(r.preview){const p=r.preview,R=Math.max(p.w,p.h)/28;
  h+=`<div class="omap"><img src="${p.src}" alt=""><svg viewBox="0 0 ${p.w} ${p.h}" preserveAspectRatio="none">`;
  for(const l of r.links){const poly=p.shared[l.a+'-'+l.b];if(poly&&poly.length)h+=`<polygon class="sh ${l.status}" points="${poly.map(q=>q.join(',')).join(' ')}"/>`}
  for(const [i,q] of Object.entries(p.quads))h+=`<polygon class="q" data-i="${i}" points="${q.map(v=>v.join(',')).join(' ')}"/>`;
  for(const [i,q] of Object.entries(p.quads)){const cx=q.reduce((s,v)=>s+v[0],0)/4,cy=q.reduce((s,v)=>s+v[1],0)/4;
   h+=`<g class="nb" data-i="${i}" role="button" tabindex="0" aria-label="${esc(L('per_photo',{k:+i+1}))}"><circle cx="${cx}" cy="${cy}" r="${R}"/><text x="${cx}" y="${cy}" font-size="${R*1.15}">${+i+1}</text></g>`}
  h+=`</svg></div><div class="olegend"><span><i style="background:#2bb38a"></i>${esc(L('legend_good'))}</span><span><i style="background:#f2a900"></i>${esc(L('legend_weak'))}</span><span><i style="background:#e04848"></i>${esc(L('legend_gap'))}</span><span><i style="background:#5b8def"></i>${esc(L('legend_dup'))}</span></div><p class="hungr" style="margin:0 0 8px">${esc(L('map_note'))}</p>`}
 else{h+=`<div class="ostrip">${S.items.map((it,k)=>{const l=r.links.find(x=>x.a===k&&x.b===k+1);const lk=k<n-1?(l?`<div class="plink ${l.status}">${pct(l.overlap)}%<i></i></div>`:'<div class="plink gap">&mdash;<i></i></div>'):'';
  return `<div class="pit" style="${r.main.includes(k)?'':'opacity:.4'}"><img src="${it.thumb}" alt=""><span class="pn">${k+1}</span></div>${lk}`}).join('')}</div>`}
 if(r.links.length){h+=`<h3 style="margin:10px 0 2px">${esc(L('links_h'))}</h3><p class="hungr" style="margin:0">${esc(L('ideal',{a:pct(B.ideal[0]),b:pct(B.ideal[1])}))}</p><ul class="olinks">`;
  for(const l of r.links)h+=`<li data-a="${l.a}" data-b="${l.b}"><span><b>${l.a+1} &harr; ${l.b+1}</b></span><span class="obar"><span class="z" style="left:${B.ideal[0]*100}%;width:${(B.ideal[1]-B.ideal[0])*100}%"></span><span class="v ${l.status}" style="width:${Math.min(100,pct(l.overlap))}%"></span></span>
   <span class="st ${l.status}">${pct(l.overlap)}% ${esc(L('s_'+l.status))}${l.motion==='moved'&&r.kind==='turn'?'<br>'+esc(L('moved_tag')):''}</span></li>`;h+='</ul>'}
 const iss=r.issues;
 h+=`<h3 style="margin:10px 0 4px">${esc(L('fix_h'))}</h3><ul class="oiss">${iss.length?iss.map(x=>`<li class="${x.sev}" data-a="${x.a??''}" data-b="${x.b??''}"><span>${ICO(SEVI[x.sev])}</span><span>${esc(issueText(x))}</span></li>`).join(''):`<li class="info"><span>${ICO('check')}</span><span>${esc(L('all_good'))}</span></li>`}</ul>`;
 const drop=r.issues.filter(x=>x.code==='dup'||x.code==='unmatched').length;
 h+=`<div class="btns">${r.joinable?`<button type="button" class="btn primary" data-a="join">${esc(L(r.joined?'join':'together'))}</button>`:''}
  ${drop?`<button type="button" class="btn ghost" data-a="drop">${esc(L('drop'))}</button>`:''}<button type="button" class="btn ghost" data-a="more">${esc(L('addmore'))}</button></div>
  <div class="row" style="margin-top:8px"><button type="button" class="minibtn" data-a="restart">${esc(L('restart'))}</button><span></span></div></section>`;
 out.innerHTML=h;g.step(r.verdict==='fix'?1:2);const c=$('#ovcard');c.scrollIntoView({behavior:'smooth',block:'start'});
 g.say(vt+'. '+(iss.length?issueText(iss[0]):L('all_good')));g.buzz(r.verdict==='ready'?40:[40,80,40]);
 const sel=i=>{c.querySelectorAll('.q').forEach(q=>{q.classList.toggle('sel',q.dataset.i==i);q.classList.toggle('dim',i!=null&&q.dataset.i!=i)});
  c.querySelectorAll('.oiss li').forEach(li=>li.classList.toggle('hl',i!=null&&(li.dataset.a==i||li.dataset.b==i)))};
 c.addEventListener('click',e=>{const nb=e.target.closest('.nb');if(nb){const i=nb.dataset.i;sel(c.querySelector('.q.sel')?.dataset.i===i?null:i);return}
  const a=e.target.closest('[data-a]')?.dataset.a;
  if(a==='join')join();if(a==='restart'){clearSet();g.out.innerHTML='';g.step(1)}
  if(a==='more'){if(g.LIVE)g.openFinder();else g.galIn.click()}
  if(a==='drop'){const ids=new Set();r.issues.forEach(x=>{if(x.code==='dup')ids.add(S.items[x.b]?.id);if(x.code==='unmatched')ids.add(S.items[x.a]?.id)});S.items=S.items.filter(i=>!ids.has(i.id));S.rep=null;estimate();render();check()}})}

/* ---------- join and analyse: one answer for the set through the normal result page ---------- */
function join(retry){const r0=S.rep;if(!r0)return;g.busy(L('joining'));
 const pg=g.prog($('#pg'),[[0,L('st_join')],[.3,L('st_type')]],g.store.get('etaT',4000)+2000,false);
 g.post('pano?'+g.params({set:r0.set,gx:g.gx()}),null,pg,{
  ok:async r=>{const items=S.items.slice(),ref=items[r0._map[r.ref]]||items[0];
   const blob=r.image?await urlBlob(r.image):ref.blob,label=L(r.set.joined?'set_label_j':'set_label_w',{n:r.set.n});
   g.runSet(r,blob,ref.g,label);perPhoto(r,items,r0)},
  err:(m,s)=>{if(s===410&&!retry){g.toast(L('resend'));resend();return}s===-2?g.cancelled():pfail(m,join)}})}
async function resend(){const body={kind:S.kind,images:await Promise.all(S.items.map(i=>toB64(i.blob))),f35:S.items.map(i=>i.f35)};
 g.post('overlap?'+g.params({kind:S.kind}),new Blob([JSON.stringify(body)],{type:'application/json'}),null,{ok:r=>{reorder(r);S.rep=r;join(true)},err:m=>pfail(m,check)})}
function perPhoto(r,items,r0){const N=A.D.names,top=r.label,rows=[];
 if(r.set.joined_type)rows.push({img:r.image,t:L('per_joined'),lab:r.set.joined_type.label,c:r.set.joined_type.confidence});
 r.set.photos.forEach(p=>{const k=r0._map[p.i];rows.push({img:items[k]?.thumb,t:L('per_photo',{k:k+1}),lab:p.label,c:p.confidence})});
 const agree=rows.filter(x=>x.lab===top).length;
 const html=`<section class="card" id="percard"><div class="kicker">${esc(L('per_h'))}</div><p class="hungr" style="margin:2px 0 4px">${esc(L('per_note'))}</p>
  <p style="margin:0"><b>${esc(L('agree',{a:agree,n:rows.length}))}</b></p><div class="perset">${rows.map(x=>`<div><img src="${x.img||''}" alt=""><p><b>${esc(x.t)}</b><br><span class="${x.lab===top?'agree':'dis'}">${esc(N[x.lab]||x.lab)} ${pct(x.c)}%</span></p></div>`).join('')}</div></section>`;
 const anchor=g.out.querySelector('.card.result')||g.out.firstElementChild;if(anchor)anchor.insertAdjacentHTML('afterend',html)}

A.f35Of=f35Of;   // also used by single photos (src/guide.py): the shrunk upload has no EXIF
A.pano={show(on){want=on;if(ready)apply()},init(){g=A._g;D=A.D;P=D.pano;esc=A.esc;lang=g.store.get('lang','en');S.kind=g.store.get('pkind','turn');
  build();ready=true;apply();A.onPicks=onPicks},
 label(){return L('label_turn',{k:S.items.length+1})},hint(){return S.items.length?(S.kind==='turn'?L('next_turn',{d:L((S.dir||1)<0?'left':'right')}):L('next_walk')):L(S.kind==='turn'?'hint_first_turn':'hint_first_walk')},
 finder,tick,shot,gx(){return `${S.kind}:${S.items.length}:${S.rep?S.rep.verdict:'-'}:${S.items.filter(i=>i.src==='cam').length}`},
 _S:S,_match:match,_sig:sig,_add:add,_check:check,_join:join};
})();
"""


def data():
    return {"max": MAX_IMAGES, "bands": CFG["bands"], "text": TEXT,
            "svg": {"turn": TURN_SVG, "strip": STRIP_SVG, "walk": WALK_SVG}, "examples": EXAMPLES}


if __name__ == "__main__":
    print(json.dumps(data())[:300])
