"""Data for the guidance layer (src/guide.py): field questions, practice quiz, Hindi strings.

Field questions (asked when the model is unsure). Each question has L[c] = P(answer "yes" | the landslide is
of class c). The app multiplies the model's probabilities by these likelihoods (Bayes' rule) and asks next the
question with the largest expected information gain.
The likelihoods are EXPERT RULES OF THUMB taken from the type descriptions of Hungr, Leroueil & Picarelli
(2014) and Varnes-style field guides, not values fitted to data (the dataset has no field answers). They are
kept within 0.05-0.92, so one answer can shift but not overrule a confident model; the app says this on screen.
"""

C6 = ("fall_topple", "rotational_slide", "translational_slide", "debris_flow", "earth_mud_flow", "no_landslide")


def _L(fall, rot, trans, debris, earth, none):
    return dict(zip(C6, (fall, rot, trans, debris, earth, none)))


QUESTIONS = [
    {"id": "fresh", "t": "Is there any fresh scar, crack or moved material on the slope?",
     "hint": "Bare soil or rock that is lighter than its surroundings, broken vegetation, debris on the road.",
     "L": _L(.92, .92, .92, .92, .92, .12)},
    {"id": "curved", "t": "Is the top scar curved like a spoon, with ground or trees tilted back towards the slope?",
     "hint": "Seen from below, the main scarp looks like an arc; blocks below it lean backwards.",
     "L": _L(.08, .85, .2, .08, .35, .05)},
    {"id": "planar", "t": "Is a flat, smooth sliding surface exposed (a rock bedding or joint plane, or soil on rock)?",
     "hint": "A large tilted plane, often parallel to the slope, from which a sheet of material slid off.",
     "L": _L(.3, .15, .85, .1, .1, .05)},
    {"id": "blocks", "t": "Did separate rocks or blocks drop from a steep rock face (cliff) above?",
     "hint": "Angular blocks at the foot of a cliff, fresh light-coloured spots on the face, impact marks.",
     "L": _L(.9, .1, .3, .35, .05, .1)},
    {"id": "channel", "t": "Did the material come down a gully or stream channel and spread as a fan at its end?",
     "hint": "A narrow path down the slope, then a wide deposit where the gully meets the road or valley.",
     "L": _L(.12, .08, .1, .9, .35, .05)},
    {"id": "levees", "t": "Are there boulder ridges (levees) along the channel or mud lines on trees and walls?",
     "hint": "Signs of a fast, wet surge: ridges of coarse debris on both sides, mud splashed high on trees.",
     "L": _L(.05, .05, .08, .82, .25, .03)},
    {"id": "fine", "t": "Is the moved material mostly fine, wet soil forming a lobe or tongue, with few large boulders?",
     "hint": "Soft, muddy, bulging tongue; hummocky surface; springs or ponded water.",
     "L": _L(.05, .35, .2, .3, .88, .05)},
    {"id": "steep", "t": "Is the source a near-vertical rock face (steeper than about 60 degrees)?",
     "hint": "Exposed bedrock wall rather than a soil slope.",
     "L": _L(.88, .1, .35, .15, .05, .3)},
]

# ---------------------------------------------------------------- practice quiz illustrations
_F = '<rect x="3" y="3" width="94" height="124" rx="10" fill="none" stroke="currentColor" stroke-width="3"/>'
_SKY = '<rect x="6" y="6" width="88" height="{h}" rx="7" fill="#9fd3f0"/>'
_SLOPE = ('<path d="M6 {y0}L52 {y1}L94 {y1}V124H6Z" fill="#7a9a5a"/><path d="M40 {sy}Q58 {sy2} 62 {ty}L70 124" fill="#b58a5a" '
          'stroke="#ff745e" stroke-width="2"/><path d="M38 {sy}Q50 {sc} 64 {sy}" fill="none" stroke="#e04282" stroke-width="2.5"/>')


def _quiz_svg(kind):
    body = ""
    if kind in ("good", "tilt", "dark"):
        body = _SKY.format(h=30) + _SLOPE.format(y0=110, y1=30, sy=46, sy2=60, ty=96, sc=38)
        if kind == "tilt":
            body = f'<g transform="rotate(16 50 65)">{body}</g>'
        if kind == "dark":
            body += '<rect x="6" y="6" width="88" height="118" rx="7" fill="#0b0f0d" opacity=".78"/>'
    elif kind == "sky":
        body = _SKY.format(h=100) + '<path d="M6 124L40 104L94 110V124Z" fill="#7a9a5a"/><path d="M50 106L58 124" stroke="#b58a5a" stroke-width="4"/>'
    elif kind == "close":
        import random
        rnd = random.Random(3)
        body = '<rect x="6" y="6" width="88" height="118" rx="7" fill="#b58a5a"/>' + "".join(
            f'<circle cx="{rnd.randint(10, 90)}" cy="{rnd.randint(10, 120)}" r="{rnd.randint(2, 7)}" fill="#8a6a46"/>' for _ in range(40))
    elif kind == "aerial":
        body = ('<rect x="6" y="6" width="88" height="118" rx="7" fill="#6f9152"/><path d="M6 80C30 70 60 96 94 84" stroke="#ccc" stroke-width="6" fill="none"/>'
                '<path d="M44 20C60 22 64 40 58 60C54 74 40 76 36 62C32 46 30 22 44 20Z" fill="#b58a5a"/>'
                '<circle cx="24" cy="30" r="8" fill="#4f7039"/><circle cx="76" cy="110" r="9" fill="#4f7039"/>')
    return f'<svg viewBox="0 0 100 130" aria-hidden="true">{body}{_F}</svg>'


QUIZ = [
    {"q": "Which photo will give a reliable landslide type?", "a": _quiz_svg("close"), "b": _quiz_svg("good"), "correct": "b",
     "why": "The whole failure, scarp to toe, must be in view. A close-up of soil hides the shape that defines the type."},
    {"q": "Which framing is better?", "a": _quiz_svg("good"), "b": _quiz_svg("sky"), "correct": "a",
     "why": "In B the slope is a thin strip under the sky. Tilt down so the slope fills most of the frame."},
    {"q": "Which view does this app work with?", "a": _quiz_svg("aerial"), "b": _quiz_svg("good"), "correct": "b",
     "why": "The app is trained on ground photos taken from below. Drone, satellite or map views are rejected."},
    {"q": "Which photo gives better sizes and slope angle?", "a": _quiz_svg("tilt"), "b": _quiz_svg("good"), "correct": "b",
     "why": "Hold the phone level: a tilted photo distorts the slope angle and the size estimate."},
    {"q": "Which photo should you send?", "a": _quiz_svg("good"), "b": _quiz_svg("dark"), "correct": "a",
     "why": "In poor light cracks and scarps disappear. Photograph in daylight with the sun behind or beside you."},
]

# ---------------------------------------------------------------- Hindi (machine-drafted; needs a native-speaker review)
HI = {
    # page and result cards
    "Landslide Type Check": "भूस्खलन प्रकार जाँच", "Field classifier · Hungr 2014": "फ़ील्ड वर्गीकरण · हंगर 2014",
    "New observation": "नया अवलोकन", "Photograph the slope from below": "ढलान की फ़ोटो नीचे से लें",
    "Stand at the foot of the slope or across the road, facing up at it.": "ढलान के नीचे या सड़क के उस पार खड़े होकर ऊपर की ओर देखें।",
    "Fit the whole slope: the top (scarp or source) and the bottom (deposit).": "पूरी ढलान फ़्रेम में लें: ऊपर (दरार/स्रोत) और नीचे (मलबा)।",
    "Hold the phone level, in daylight. One slope per photo.": "फ़ोन सीधा रखें, दिन की रोशनी में। एक फ़ोटो में एक ढलान।",
    "Take photo": "फ़ोटो लें", "From gallery": "गैलरी से",
    "Distance from you to the slope, metres (optional, makes sizes more accurate)": "आपसे ढलान तक की दूरी, मीटर (वैकल्पिक, आकार अधिक सटीक होते हैं)",
    "e.g. 40": "जैसे 40",
    "Decision support only — confirm with a geologist or geotechnical engineer before acting.": "केवल निर्णय-सहायता — कार्रवाई से पहले भूविज्ञानी या भू-तकनीकी इंजीनियर से पुष्टि करें।",
    "Classification after Hungr, Leroueil & Picarelli (2014).": "वर्गीकरण: हंगर, लेरुइल और पिकारेली (2014)।",
    "Landslide type": "भूस्खलन का प्रकार", "Uncertain — most likely": "अनिश्चित — सबसे संभावित", "Landslide family": "भूस्खलन समूह",
    "Confidence": "विश्वास", "Confidence (family)": "विश्वास (समूह)", "All types": "सभी प्रकार", "Check on site": "स्थल पर जाँचें", "Safety": "सुरक्षा",
    "Photo check": "फ़ोटो जाँच", "Retake the photo": "फ़ोटो दोबारा लें", "Retake photo": "फ़ोटो दोबारा लें",
    "Estimated dimensions": "अनुमानित आयाम", "Analysing photo": "फ़ोटो का विश्लेषण", "Something went wrong": "कुछ गलत हुआ",
    "Check the connection to the laptop running the app, then try again.": "ऐप चला रहे लैपटॉप से कनेक्शन जाँचें, फिर दोबारा कोशिश करें।",
    "Uncertain": "अनिश्चित", "Analysing…": "विश्लेषण…", "Error": "त्रुटि", "Checking photo…": "फ़ोटो जाँच रहे हैं…",
    "Check the photo": "फ़ोटो जाँचें", "Measuring…": "माप रहे हैं…", "Identifying…": "पहचान रहे हैं…", "Cancelled": "रद्द",
    "Saved for later": "बाद के लिए सहेजा", "Offline": "ऑफ़लाइन", "Rock fall / topple": "चट्टान गिरना / पलटना",
    "Rotational slide (slump)": "घूर्णी स्खलन (स्लंप)", "Translational (planar) slide": "समतल स्खलन",
    "Debris flow": "मलबा प्रवाह", "Earthflow / mudflow": "मिट्टी / कीचड़ प्रवाह", "No landslide visible": "कोई भूस्खलन नहीं दिखा",
    "high": "उच्च", "medium": "मध्यम", "low": "कम",
    # on-site checks and safety (app.py INFO)
    "Fresh, light-coloured scars on the rock face where blocks came from": "चट्टान की सतह पर ताज़े, हल्के रंग के निशान जहाँ से पत्थर गिरे",
    "Open joints or overhangs above the scar that could release more blocks": "निशान के ऊपर खुली दरारें या लटकी चट्टानें जो और पत्थर गिरा सकती हैं",
    "Size of the largest block on the road or at the toe": "सड़क पर या तलहटी में सबसे बड़े पत्थर का आकार",
    "Impact marks on the road, damaged trees or barriers": "सड़क पर टक्कर के निशान, टूटे पेड़ या अवरोधक",
    "Curved (spoon-shaped) main scarp at the top": "ऊपर मुड़ी हुई (चम्मच जैसी) मुख्य दरार",
    "Ground or trees tilted back towards the slope": "ज़मीन या पेड़ ढलान की ओर पीछे झुके हुए",
    "Tension cracks above the scarp, bulging at the toe": "दरार के ऊपर तनाव-दरारें, तलहटी पर उभार",
    "Road sagging or dropped with a curved crack": "मुड़ी दरार के साथ सड़क धँसी या नीचे गिरी",
    "Flat sliding surface exposed (bedding, joint plane or soil-rock contact)": "समतल फिसलन सतह खुली (परत, जोड़ या मिट्टी-चट्टान संपर्क)",
    "Dip of that surface compared with the slope (out-of-slope dip is unstable)": "ढलान की तुलना में उस सतह का झुकाव (बाहर की ओर झुकाव अस्थिर है)",
    "Shallow, elongated scar; displaced sheet of soil or slabs below": "उथला, लंबा निशान; नीचे खिसकी मिट्टी या पट्टियाँ",
    "Seepage or wet zone along the sliding surface": "फिसलन सतह के साथ रिसाव या गीला क्षेत्र",
    "Gully or stream channel feeding the deposit": "मलबे तक आने वाली नाली या जलधारा",
    "Levees of boulders along the channel, mud lines on trees": "नाली के किनारे पत्थरों की मेड़ें, पेड़ों पर कीचड़ की रेखाएँ",
    "Fan of mixed boulders and mud at the gully outlet / road crossing": "नाली के मुहाने / सड़क पर पत्थर और कीचड़ का पंखा",
    "Blocked culvert or bridge": "बंद पुलिया या पुल",
    "Lobe or tongue of fine, wet soil; few large boulders": "महीन, गीली मिट्टी की जीभ जैसी आकृति; बड़े पत्थर कम",
    "Hummocky surface, transverse cracks, wet springs": "ऊबड़-खाबड़ सतह, आड़ी दरारें, गीले झरने",
    "Source scar upslope, often in clayey or weathered soil": "ऊपर स्रोत का निशान, अक्सर चिकनी या अपक्षयित मिट्टी में",
    "Movement indicators: bent fences, tilted poles, pushed drains": "खिसकने के संकेत: मुड़ी बाड़, झुके खंभे, धकेली नालियाँ",
    "Cracks at the top of the cut or slope": "कटान या ढलान के ऊपर दरारें",
    "Bulging, seepage or wet patches on the face": "सतह पर उभार, रिसाव या गीले धब्बे",
    "Leaning trees, poles or retaining walls": "झुके पेड़, खंभे या रिटेनिंग दीवारें",
    "Small fresh rock fragments on the road (early rock fall)": "सड़क पर छोटे ताज़े पत्थर (शुरुआती चट्टान गिरना)",
    # journey, modes, shots, distance, notes
    "Safe spot": "सुरक्षित जगह", "Frame": "फ़्रेम", "Check": "जाँच", "Result": "परिणाम",
    "Single photo": "एक फ़ोटो", "Site survey (4 shots)": "स्थल सर्वे (4 शॉट)", "Start a new site": "नया स्थल शुरू करें",
    "Clears the shot list, notes and location for the next landslide.": "अगले भूस्खलन के लिए शॉट सूची, नोट्स और स्थान साफ़ करता है।",
    "Whole slope": "पूरी ढलान", "Source / scarp": "स्रोत / दरार", "Deposit close-up": "मलबे का क्लोज़-अप", "Something for scale": "पैमाने के लिए कोई वस्तु",
    "From below, 20-80 m away. Get the top (scarp) and the bottom (deposit) in one frame.": "नीचे से, 20-80 मीटर दूर से। ऊपर (दरार) और नीचे (मलबा) एक फ़्रेम में लें।",
    "Zoom in (do not climb) so the upper break, cracks or bare face fill the frame.": "ज़ूम करें (चढ़ें नहीं) ताकि ऊपरी टूटन, दरारें या खुली सतह फ़्रेम भर दें।",
    "5-15 m from the toe, boulders and soil filling the frame. This gives the grain size.": "तलहटी से 5-15 मीटर, पत्थर और मिट्टी फ़्रेम भरें। इससे कण-आकार मिलता है।",
    "The whole slope again with a person, car or pole near the toe. This fixes the metres.": "तलहटी के पास किसी व्यक्ति, गाड़ी या खंभे के साथ पूरी ढलान फिर से। इससे मीटर तय होते हैं।",
    "Don't know the distance?": "दूरी नहीं पता?", "Pace it to the toe (only where safe):": "तलहटी तक कदम गिनें (केवल जहाँ सुरक्षित हो):",
    "steps": "कदम", "Use": "उपयोग करें", "Use a distance between 1 and 3000 m (or leave it empty).": "1 से 3000 मीटर के बीच दूरी लिखें (या खाली छोड़ें)।",
    "Notes & location (for the report)": "नोट्स और स्थान (रिपोर्ट के लिए)", "Site name, road / km, what you saw": "स्थल का नाम, सड़क / किमी, आपने क्या देखा",
    "Weather…": "मौसम…", "Dry": "सूखा", "Raining now": "अभी बारिश", "Rain in the last 24 h": "पिछले 24 घंटे में बारिश",
    "Add my location": "मेरा स्थान जोड़ें", "Finding your location…": "आपका स्थान ढूँढ रहे हैं…",
    # tour, safety, help
    "What this app does": "यह ऐप क्या करता है", "Stand below, see it all": "नीचे खड़े हों, पूरा देखें", "Frame top to bottom": "ऊपर से नीचे तक फ़्रेम करें",
    "Light, level, steady": "रोशनी, सीधा, स्थिर", "Read the answer": "उत्तर पढ़ें", "Skip": "छोड़ें", "Back": "पीछे", "Next": "आगे", "Start": "शुरू करें",
    "Photograph a landslide from the ground. The app names its type, shows the on-site signs to confirm, and estimates its size. It supports your judgement; it does not replace a geologist.":
        "ज़मीन से भूस्खलन की फ़ोटो लें। ऐप उसका प्रकार बताता है, पुष्टि के लिए स्थल-संकेत दिखाता है और आकार का अनुमान लगाता है। यह आपके निर्णय में मदद करता है; भूविज्ञानी का स्थान नहीं लेता।",
    "Stand at the foot of the slope or across the road, 20-80 m away, never on the slope or the deposit. Face up at it.":
        "ढलान के नीचे या सड़क के उस पार 20-80 मीटर दूर खड़े हों, कभी ढलान या मलबे पर नहीं। ऊपर की ओर देखें।",
    "Fit the whole failure: the scarp at the top and the toe at the bottom. Not just a close-up of soil, not mostly sky.":
        "पूरा भूस्खलन फ़्रेम में लें: ऊपर दरार और नीचे तलहटी। सिर्फ़ मिट्टी का क्लोज़-अप नहीं, ज़्यादातर आसमान नहीं।",
    "Daylight with the sun behind or beside you. Hold the phone level with two hands. The live guide turns green when the shot is good.":
        "दिन की रोशनी, सूरज आपके पीछे या बगल में। फ़ोन दोनों हाथों से सीधा पकड़ें। शॉट अच्छा होने पर लाइव गाइड हरा हो जाता है।",
    "Confidence says how sure the type is. Sizes come with a 90% range. Below each result, 'Next steps' tells you what extra photo or detail would make it better.":
        "विश्वास बताता है कि प्रकार कितना निश्चित है। आकार 90% सीमा के साथ आते हैं। हर परिणाम के नीचे 'अगले कदम' बताते हैं कि कौन सी अतिरिक्त फ़ोटो या जानकारी इसे बेहतर बनाएगी।",
    "Before you photograph": "फ़ोटो लेने से पहले", "Are you in a safe spot?": "क्या आप सुरक्षित जगह पर हैं?",
    "I am below or beside the slope, not on it or on the deposit.": "मैं ढलान के नीचे या बगल में हूँ, उस पर या मलबे पर नहीं।",
    "Nothing hangs over me and I am outside the path of falling rocks.": "मेरे ऊपर कुछ लटका नहीं है और मैं गिरते पत्थरों के रास्ते से बाहर हूँ।",
    "No fresh cracking, rumbling, running muddy water or moving debris right now.": "अभी कोई नई दरार, गड़गड़ाहट, बहता कीचड़-पानी या खिसकता मलबा नहीं है।",
    "On a road: I am out of traffic, or someone is watching for me.": "सड़क पर: मैं यातायात से बाहर हूँ, या कोई मेरी निगरानी कर रहा है।",
    "If any of these is not true, move back first. If the slope is moving, leave and report it; no photo is worth it.":
        "इनमें से कोई भी सच नहीं है तो पहले पीछे हटें। ढलान खिसक रही हो तो वहाँ से हटें और सूचना दें; कोई फ़ोटो इसके लायक नहीं।",
    "Not now": "अभी नहीं", "I am safe, continue": "मैं सुरक्षित हूँ, आगे बढ़ें",
    "Help & guide": "सहायता और गाइड", "Close": "बंद करें", "Taking a good photo": "अच्छी फ़ोटो लेना", "Reading the result": "परिणाम पढ़ना",
    "Limits": "सीमाएँ", "Settings": "सेटिंग्स", "Voice guidance (camera and results)": "आवाज़ मार्गदर्शन (कैमरा और परिणाम)",
    "Sun mode (high contrast, bigger text)": "धूप मोड (उच्च कंट्रास्ट, बड़ा टेक्स्ट)", "Replay the guide": "गाइड फिर से देखें",
    "Practice quiz": "अभ्यास प्रश्नोत्तरी", "Clear recent photos": "हाल की फ़ोटो हटाएँ", "Cleared": "हटा दी गईं",
    # photo check and live camera
    "Instant photo check": "तुरंत फ़ोटो जाँच", "One thing could be better": "एक चीज़ बेहतर हो सकती है", "OK": "ठीक",
    "Retake": "दोबारा लें", "Analyse anyway": "फिर भी विश्लेषण करें",
    "Light": "रोशनी", "Shadows": "छाया", "Highlights": "चमक", "Haze / contrast": "धुंध / कंट्रास्ट", "Sharpness": "स्पष्टता", "Framing": "फ़्रेमिंग",
    "Resolution": "रिज़ॉल्यूशन", "Level": "सीधापन", "Aim": "दिशा", "Steady": "स्थिर",
    "Too dark. Face the slope with the sun behind or beside you, or wait for better light.": "बहुत अंधेरा। सूरज पीछे या बगल में रखकर ढलान की ओर देखें, या बेहतर रोशनी का इंतज़ार करें।",
    "Too bright. Point less at the sky; tap the slope on screen to set exposure.": "बहुत ज़्यादा रोशनी। आसमान की ओर कम रखें; एक्सपोज़र के लिए स्क्रीन पर ढलान को छुएँ।",
    "Large black areas: deep shadow hides the surface. Step sideways so the slope is evenly lit.": "बड़े काले हिस्से: गहरी छाया सतह छिपा रही है। बगल में हटें ताकि ढलान पर एक समान रोशनी हो।",
    "Large white areas: sky or glare is blown out. Tilt down so less sky is in the frame.": "बड़े सफ़ेद हिस्से: आसमान या चमक बहुत तेज़ है। फ़ोन नीचे झुकाएँ ताकि आसमान कम आए।",
    "Flat, low-contrast image (fog, rain or a dirty lens). Wipe the lens; in fog, move closer.": "फीकी तस्वीर (कोहरा, बारिश या गंदा लेंस)। लेंस पोंछें; कोहरे में पास जाएँ।",
    "Mostly sky. Tilt down so the slope fills most of the frame.": "ज़्यादातर आसमान। नीचे झुकाएँ ताकि ढलान फ़्रेम का अधिकांश भाग भरे।",
    "Blurred. Use two hands, tap the slope to focus, and hold still while the photo is taken.": "धुंधली। दोनों हाथों का उपयोग करें, फ़ोकस के लिए ढलान को छुएँ और फ़ोटो लेते समय स्थिर रहें।",
    "Camera points down. Stand below the slope and look up at it.": "कैमरा नीचे की ओर है। ढलान के नीचे खड़े होकर ऊपर देखें।",
    "Camera points steeply up. Step back so the whole slope fits.": "कैमरा बहुत ऊपर की ओर है। पीछे हटें ताकि पूरी ढलान आ जाए।",
    "Hold still for a moment.": "एक पल स्थिर रहें।", "Good. Take the photo.": "अच्छा। फ़ोटो लें।", "Ready": "तैयार",
    "Top of the slope (scarp)": "ढलान का ऊपरी भाग (दरार)", "Bottom (toe / deposit)": "निचला भाग (तलहटी / मलबा)",
    "Phone camera": "फ़ोन कैमरा", "Frame the whole slope": "पूरी ढलान फ़्रेम करें",
    # progress
    "Uploading…": "अपलोड हो रहा है…", "Cancel": "रद्द करें", "Skip sizes": "आकार छोड़ें", "Measure sizes": "आकार मापें", "Sizes skipped.": "आकार छोड़े गए।",
    "Checking it is a usable slope photo": "जाँच रहे हैं कि फ़ोटो उपयोगी है", "Identifying the landslide type": "भूस्खलन का प्रकार पहचान रहे हैं",
    "Outlining the landslide (CLIPSeg, SAM, outline head)": "भूस्खलन की रूपरेखा बना रहे हैं", "Estimating depth and scale": "गहराई और पैमाने का अनुमान",
    "Measuring sizes and their ranges": "आकार और उनकी सीमा माप रहे हैं", "Sending the photo again": "फ़ोटो फिर से भेज रहे हैं",
    "The type is ready above. Measuring the size now; you can already tick the signs and answer the questions.":
        "प्रकार ऊपर तैयार है। अब आकार माप रहे हैं; आप संकेतों पर टिक कर सकते हैं और प्रश्नों के उत्तर दे सकते हैं।",
    # field tips
    "Cracks above the scarp mean the slope may still be moving: keep clear of the crown.": "दरार के ऊपर दरारें बताती हैं कि ढलान अभी खिसक सकती है: शिखर से दूर रहें।",
    "Photos from two or three spots give a more reliable type than one photo.": "दो-तीन जगहों से ली गई फ़ोटो एक फ़ोटो से अधिक भरोसेमंद प्रकार देती हैं।",
    "Typing the distance to the slope makes the sizes much more accurate.": "ढलान तक की दूरी लिखने से आकार बहुत अधिक सटीक होते हैं।",
    "A person or car in the photo lets the app check its own scale.": "फ़ोटो में कोई व्यक्ति या गाड़ी हो तो ऐप अपना पैमाना जाँच सकता है।",
    "Morning or late-afternoon side light shows cracks and steps best.": "सुबह या देर दोपहर की तिरछी रोशनी में दरारें सबसे अच्छी दिखती हैं।",
    "Wipe the lens: dust and rain drops make photos soft and hazy.": "लेंस पोंछें: धूल और बारिश की बूँदें फ़ोटो को धुंधला करती हैं।",
    "Note the date, location and weather with your observation.": "अपने अवलोकन के साथ तारीख, स्थान और मौसम लिखें।",
    # next steps, questions, ticks
    "Next steps": "अगले कदम", "Another photo": "एक और फ़ोटो", "Add distance": "दूरी जोड़ें", "Change": "बदलें", "Scale shot": "पैमाना शॉट",
    "Close-up": "क्लोज़-अप", "Take it": "लें", "Site report": "स्थल रिपोर्ट", "Re-measure this photo": "इस फ़ोटो को फिर से मापें",
    "Retake from below with the whole slope in view.": "पूरी ढलान दिखाते हुए नीचे से दोबारा लें।",
    "Not sure. Answer the questions below, or take a second photo from another spot.": "निश्चित नहीं। नीचे के प्रश्नों के उत्तर दें, या दूसरी जगह से एक और फ़ोटो लें।",
    "Add the distance to the slope and re-measure: sizes get much tighter ranges.": "ढलान तक की दूरी जोड़ें और फिर से मापें: आकार की सीमा बहुत सटीक होगी।",
    "Sizes are rough here. A person, car or pole near the toe lets the app check its scale.": "यहाँ आकार मोटे अनुमान हैं। तलहटी के पास व्यक्ति, गाड़ी या खंभा हो तो ऐप पैमाना जाँच सकता है।",
    "Grain size not readable. A close-up of the deposit (5–15 m) gives it.": "कण-आकार नहीं पढ़ा जा सका। मलबे का क्लोज़-अप (5–15 मीटर) इसे देगा।",
    "The landslide runs off the edge of the photo": "भूस्खलन फ़ोटो के किनारे से बाहर जा रहा है", "sizes are minimums": "आकार न्यूनतम हैं",
    "top": "ऊपर", "bottom": "नीचे", "left": "बाएँ", "right": "दाएँ",
    "tilt the phone up or step back so the scarp is inside the frame": "फ़ोन ऊपर झुकाएँ या पीछे हटें ताकि दरार फ़्रेम में आए",
    "tilt down or step back so the toe and deposit are inside": "नीचे झुकाएँ या पीछे हटें ताकि तलहटी और मलबा फ़्रेम में आएँ",
    "turn left or step back": "बाएँ मुड़ें या पीछे हटें", "turn right or step back": "दाएँ मुड़ें या पीछे हटें",
    "Survey: next shot": "सर्वे: अगला शॉट", "Survey complete: all 4 shots taken. The combined site result is above.": "सर्वे पूरा: सभी 4 शॉट लिए गए। संयुक्त स्थल परिणाम ऊपर है।",
    "Help decide · answer what you can see": "निर्णय में मदद · जो दिखे उसका उत्तर दें", "Yes": "हाँ", "No": "नहीं", "Not sure": "पता नहीं",
    "Undo last answer": "पिछला उत्तर वापस लें", "Your answers point to": "आपके उत्तर इस ओर इशारा करते हैं:",
    "No further question would change the answer much": "कोई और प्रश्न उत्तर को ज़्यादा नहीं बदलेगा",
    "Model result combined with your answers.": "मॉडल परिणाम आपके उत्तरों के साथ मिलाया गया।", "Starts from the model result.": "मॉडल परिणाम से शुरू।",
    "The answer weights are expert rules of thumb from the Hungr 2014 type descriptions, not trained on data; a geologist has the final word.":
        "उत्तरों के भार हंगर 2014 के विवरणों पर आधारित विशेषज्ञ अनुमान हैं, डेटा से सीखे नहीं गए; अंतिम निर्णय भूविज्ञानी का है।",
    "Tap each sign you can see on site.": "स्थल पर दिखने वाले हर संकेत को छुएँ।", "Signs seen:": "दिखे संकेत:",
    "The field signs support this type.": "स्थल-संकेत इस प्रकार का समर्थन करते हैं।",
    "Few signs match: take another photo or ask a geologist.": "कम संकेत मिलते हैं: एक और फ़ोटो लें या भूविज्ञानी से पूछें।",
    "The type with the larger share of signs seen is the better field match.": "जिस प्रकार के अधिक संकेत दिखे, वही स्थल पर बेहतर मेल है।",
    "What do these numbers mean?": "इन संख्याओं का क्या अर्थ है?", "Save report": "रिपोर्ट सहेजें", "Share": "साझा करें", "Save site report": "स्थल रिपोर्ट सहेजें",
    "Sizes: median of the whole-slope shots, with the spread between photos added to the range.": "आकार: पूरी-ढलान शॉट्स का माध्यिका, फ़ोटो के बीच का अंतर सीमा में जोड़ा गया।",
    "Please retake the photo.": "कृपया फ़ोटो दोबारा लें।", "Uncertain. Most likely": "अनिश्चित। सबसे संभावित", "metres": "मीटर",
    "Retake needed": "दोबारा लेना ज़रूरी", "(uncertain)": "(अनिश्चित)", "(sent later)": "(बाद में भेजा)",
    "A saved photo was analysed": "एक सहेजी गई फ़ोटो का विश्लेषण हुआ", "retake needed": "दोबारा लेना ज़रूरी", "See Recent observations.": "हाल के अवलोकन देखें।",
    "No connection": "कनेक्शन नहीं", "Try again": "फिर कोशिश करें", "Send later": "बाद में भेजें", "New photo": "नई फ़ोटो",
    "Saved on the phone: it will be analysed automatically when the connection is back": "फ़ोन पर सहेजा गया: कनेक्शन लौटने पर अपने आप विश्लेषण होगा",
    "Keep the app open or open it again later. You can keep photographing; each photo waits in the queue.": "ऐप खुला रखें या बाद में फिर खोलें। आप फ़ोटो लेते रहें; हर फ़ोटो कतार में इंतज़ार करेगी।",
    "Could not reach the server. Check the Wi-Fi or hotspot connection to the laptop running the app.": "सर्वर तक नहीं पहुँच सके। ऐप चला रहे लैपटॉप से वाई-फ़ाई या हॉटस्पॉट कनेक्शन जाँचें।",
    "New site started.": "नया स्थल शुरू हुआ।",
    "The shots disagree strongly on size (more than 2x). Check that they show the same landslide, or retake the whole-slope shot with the distance typed in.":
        "शॉट्स आकार पर बहुत असहमत हैं (2 गुना से अधिक)। जाँचें कि वे एक ही भूस्खलन दिखाते हैं, या दूरी लिखकर पूरी-ढलान शॉट दोबारा लें।",
    # glossary
    "Scarp / crown": "दरार / शिखर", "Toe / deposit": "तलहटी / मलबा", "Uncertain / family": "अनिश्चित / समूह", "90% range": "90% सीमा",
    "Depth trust": "गहराई भरोसा", "Grain size D16/D50/D84": "कण-आकार D16/D50/D84", "Travel angle (H/L)": "यात्रा कोण (H/L)",
    "Size class (Fell 1994)": "आकार वर्ग (फ़ेल 1994)", "Slope angle": "ढलान कोण",
    # practice quiz
    "Practice": "अभ्यास", "Skip practice": "अभ्यास छोड़ें", "Done": "हो गया", "Correct.": "सही।", "Not quite.": "पूरी तरह सही नहीं।",
    "Which photo will give a reliable landslide type?": "कौन सी फ़ोटो भरोसेमंद भूस्खलन प्रकार देगी?", "Which framing is better?": "कौन सी फ़्रेमिंग बेहतर है?",
    "Which view does this app work with?": "यह ऐप किस दृश्य के साथ काम करता है?", "Which photo gives better sizes and slope angle?": "कौन सी फ़ोटो बेहतर आकार और ढलान कोण देती है?",
    "Which photo should you send?": "आपको कौन सी फ़ोटो भेजनी चाहिए?", "Well done. You are ready for the field.": "बहुत अच्छा। आप फ़ील्ड के लिए तैयार हैं।",
    # field questions
    "Is there any fresh scar, crack or moved material on the slope?": "क्या ढलान पर कोई ताज़ा निशान, दरार या खिसकी सामग्री है?",
    "Is the top scar curved like a spoon, with ground or trees tilted back towards the slope?": "क्या ऊपरी दरार चम्मच की तरह मुड़ी है, और ज़मीन या पेड़ ढलान की ओर पीछे झुके हैं?",
    "Is a flat, smooth sliding surface exposed (a rock bedding or joint plane, or soil on rock)?": "क्या कोई समतल, चिकनी फिसलन सतह खुली है (चट्टान की परत, जोड़, या चट्टान पर मिट्टी)?",
    "Did separate rocks or blocks drop from a steep rock face (cliff) above?": "क्या ऊपर की खड़ी चट्टान से अलग-अलग पत्थर या खंड गिरे?",
    "Did the material come down a gully or stream channel and spread as a fan at its end?": "क्या सामग्री नाली या जलधारा से नीचे आई और अंत में पंखे की तरह फैली?",
    "Are there boulder ridges (levees) along the channel or mud lines on trees and walls?": "क्या नाली के किनारे पत्थरों की मेड़ें या पेड़ों और दीवारों पर कीचड़ की रेखाएँ हैं?",
    "Is the moved material mostly fine, wet soil forming a lobe or tongue, with few large boulders?": "क्या खिसकी सामग्री ज़्यादातर महीन, गीली मिट्टी है जो जीभ जैसी आकृति बनाती है, बड़े पत्थर कम?",
    "Is the source a near-vertical rock face (steeper than about 60 degrees)?": "क्या स्रोत लगभग खड़ी चट्टान है (लगभग 60 डिग्री से अधिक)?",
    # safety notes, measure names, size classes
    'More blocks can fall without warning, especially during/after rain or blasting. Stay out of the run-out zone below the face; observe from the side.': 'बिना चेतावनी के और पत्थर गिर सकते हैं, खासकर बारिश या ब्लास्टिंग के दौरान/बाद। चट्टान के नीचे गिरने वाले क्षेत्र से बाहर रहें; बगल से देखें।',
    'Slumps often keep moving and can retrogress upslope. Keep off the cracked crown and the bulging toe; mark cracks and check them again for widening.': 'स्लंप अक्सर खिसकते रहते हैं और ऊपर की ओर बढ़ सकते हैं। दरार वाले शिखर और उभरी तलहटी से दूर रहें; दरारों पर निशान लगाएँ और चौड़ी होने की जाँच फिर करें।',
    'Remaining material on the same plane can slide again, often after rain. Do not stand on the displaced mass or directly below the scar.': 'उसी सतह पर बचा हुआ पदार्थ फिर से खिसक सकता है, अक्सर बारिश के बाद। खिसके हुए भाग पर या निशान के ठीक नीचे खड़े न हों।',
    'Debris flows recur in the same channel and can arrive as fast surges during rain. Never stand in or close to the channel; watch for rising water or noise upstream.': 'मलबा प्रवाह उसी नाली में दोबारा आते हैं और बारिश में तेज़ लहरों की तरह पहुँच सकते हैं। कभी भी नाली में या उसके पास खड़े न हों; ऊपर से बढ़ते पानी या आवाज़ पर ध्यान दें।',
    'Wet earthflows can surge and are soft underfoot. Do not walk on the flow; watch for new cracks and water ponding on the source area.': 'गीले मिट्टी-प्रवाह अचानक बढ़ सकते हैं और पैरों के नीचे नरम होते हैं। प्रवाह पर न चलें; स्रोत क्षेत्र पर नई दरारों और जमा पानी पर नज़र रखें।',
    'No failure is visible in this photo. If you saw cracks, seepage or fresh debris on site, report it anyway: this tool only sees what is in the photo.': 'इस फ़ोटो में कोई विफलता नहीं दिखती। यदि आपने स्थल पर दरारें, रिसाव या ताज़ा मलबा देखा, तो फिर भी सूचना दें: यह उपकरण केवल फ़ोटो में दिखने वाली चीज़ें देखता है।',
    'Flow-type landslide': 'प्रवाह-प्रकार भूस्खलन',
    'Slide (rotational or translational)': 'स्खलन (घूर्णी या समतल)',
    'Deposit width (across slope)': 'मलबे की चौड़ाई (ढलान के आर-पार)',
    'Deposit length (down slope)': 'मलबे की लंबाई (ढलान के साथ)',
    'Height of the rock face + deposit': 'चट्टान सतह + मलबे की ऊँचाई',
    'Deposit area': 'मलबे का क्षेत्रफल',
    'Width (across slope)': 'चौड़ाई (ढलान के आर-पार)',
    'Length (down slope)': 'लंबाई (ढलान के साथ)',
    'Height (crown to toe)': 'ऊँचाई (शिखर से तलहटी)',
    'Area of the scar': 'निशान का क्षेत्रफल',
    'Track length (down slope)': 'मार्ग की लंबाई (ढलान के साथ)',
    'Median width': 'माध्यिका चौड़ाई',
    'Max width (fan / lobe)': 'अधिकतम चौड़ाई (पंखा / जीभ)',
    'Vertical drop along the track': 'मार्ग के साथ ऊर्ध्वाधर गिरावट',
    'Inundated area': 'प्रभावित क्षेत्र',
    'Largest block (b-axis)': 'सबसे बड़ा खंड (b-अक्ष)',
    'Blocks counted': 'गिने गए खंड',
    'Grain size': 'कण-आकार',
    'Grain size D16 / D50 / D84': 'कण-आकार D16 / D50 / D84',
    'Length / width': 'लंबाई / चौड़ाई',
    'Perimeter': 'परिधि',
    'Camera to slope': 'कैमरा से ढलान',
    'extremely small': 'अत्यंत छोटा',
    'very small': 'बहुत छोटा',
    'small': 'छोटा',
    'large': 'बड़ा',
    'very large': 'बहुत बड़ा',
    'extremely large': 'अत्यंत बड़ा',
    'not resolvable in this photo (take a closer photo of the deposit)': 'इस फ़ोटो में पढ़ा नहीं जा सकता (मलबे की नज़दीकी फ़ोटो लें)',
    'Answer the questions or tick the signs below to tell': 'अंतर बताने के लिए प्रश्नों के उत्तर दें या नीचे संकेतों पर टिक करें:',
    'from': 'या',
    'and': 'और',
    '(second choice)': '(दूसरा विकल्प)',
}
