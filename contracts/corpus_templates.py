"""Hand-written phrase bank for the synthetic complaint corpus (Step 0.5). Data only, no logic.

Languages: hi (Devanagari), en, pa (Gurmukhi), mixed (Hinglish, Roman script).
`{loc}` is filled with a location phrase, `{n}` with a number of days.
Every cue phrase is listed here so tests can check cue labels against the text.
"""

# Ward names as citizens write them. English names come from data/world/wards.geojson.
WARD_NAMES = {
    "hi": {"Karol Bagh": "करोल बाग", "Paharganj": "पहाड़गंज", "Chandni Chowk": "चांदनी चौक",
           "Daryaganj": "दरियागंज", "Rajinder Nagar": "राजिंदर नगर", "Gol Market": "गोल मार्केट",
           "Connaught Place": "कनॉट प्लेस", "Mandi House": "मंडी हाउस",
           "Chanakyapuri": "चाणक्यपुरी", "Race Course": "रेस कोर्स", "India Gate": "इंडिया गेट",
           "Lodhi Colony": "लोधी कॉलोनी"},
    "pa": {"Karol Bagh": "ਕਰੋਲ ਬਾਗ", "Paharganj": "ਪਹਾੜਗੰਜ", "Chandni Chowk": "ਚਾਂਦਨੀ ਚੌਕ",
           "Daryaganj": "ਦਰਿਆਗੰਜ", "Rajinder Nagar": "ਰਾਜਿੰਦਰ ਨਗਰ", "Gol Market": "ਗੋਲ ਮਾਰਕੀਟ",
           "Connaught Place": "ਕਨਾਟ ਪਲੇਸ", "Mandi House": "ਮੰਡੀ ਹਾਊਸ",
           "Chanakyapuri": "ਚਾਣਕਿਆਪੁਰੀ", "Race Course": "ਰੇਸ ਕੋਰਸ", "India Gate": "ਇੰਡੀਆ ਗੇਟ",
           "Lodhi Colony": "ਲੋਧੀ ਕਲੋਨੀ"},
}

LANDMARKS = {
    "en": {"school": "the govt school", "hospital": "the hospital", "market": "the market",
           "bus_stop": "the bus stop"},
    "hi": {"school": "सरकारी स्कूल", "hospital": "अस्पताल", "market": "बाज़ार",
           "bus_stop": "बस स्टॉप"},
    "pa": {"school": "ਸਰਕਾਰੀ ਸਕੂਲ", "hospital": "ਹਸਪਤਾਲ", "market": "ਬਾਜ਼ਾਰ",
           "bus_stop": "ਬੱਸ ਸਟਾਪ"},
    "mixed": {"school": "govt school", "hospital": "hospital", "market": "market",
              "bus_stop": "bus stop"},
}

# Location phrase near a landmark: {ward} and {lm} are filled in.
POI_LOCATIONS = {
    "en": ["near {lm} in {ward}", "opposite {lm}, {ward}", "next to {lm} in {ward}"],
    "hi": ["{ward} में {lm} के पास", "{ward}, {lm} के सामने"],
    "pa": ["{ward} ਵਿੱਚ {lm} ਦੇ ਕੋਲ", "{ward}, {lm} ਦੇ ਸਾਹਮਣੇ"],
    "mixed": ["{ward} mein {lm} ke paas", "{lm} ke saamne, {ward}", "{ward} {lm} ke bagal mein"],
}

# Location phrase with no landmark: keyed by the real road type at the incident point.
ROAD_LOCATIONS = {
    "en": {"main": "on the main road in {ward}", "residential": "on the colony road in {ward}",
           "lane": "in a lane in {ward}"},
    "hi": {"main": "{ward} में मेन रोड पर", "residential": "{ward} में कॉलोनी की सड़क पर",
           "lane": "{ward} की गली में"},
    "pa": {"main": "{ward} ਵਿੱਚ ਮੇਨ ਰੋਡ ਉੱਤੇ", "residential": "{ward} ਵਿੱਚ ਕਲੋਨੀ ਦੀ ਸੜਕ ਉੱਤੇ",
           "lane": "{ward} ਦੀ ਗਲੀ ਵਿੱਚ"},
    "mixed": {"main": "{ward} main road pe", "residential": "{ward} colony road pe",
              "lane": "{ward} ki gali mein"},
}

# Core sentence per issue type. "other" = a real civic problem outside the six categories.
CORE = {
    "en": {
        "pothole": ["There is a big pothole {loc}.",
                    "Deep potholes {loc}, two-wheelers keep skidding.",
                    "The road {loc} is completely broken, full of potholes.",
                    "The road has caved in {loc}, there is a large crater."],
        "streetlight": ["The streetlight {loc} is not working.",
                        "Street lights {loc} have been off for many nights, it is pitch dark.",
                        "A street lamp {loc} keeps flickering and going off.",
                        "No lights {loc}, the pole light is dead."],
        "garbage": ["Garbage is piled up {loc} and nobody is collecting it.",
                    "Huge heap of waste {loc}, it stinks badly.",
                    "The dustbin {loc} is overflowing, trash all over the road.",
                    "People are dumping garbage {loc}, the MCD van has not come."],
        "water_leak": ["A water pipeline has burst {loc}, clean water is being wasted.",
                       "Water is leaking from the pipe {loc}.",
                       "Jal Board pipe leaking {loc}, water flowing on the road.",
                       "Broken water main {loc}, our taps have no pressure."],
        "drain_blockage": ["The drain {loc} is completely blocked.",
                           "Nala choked {loc}, dirty water is overflowing onto the road.",
                           "Sewer is overflowing {loc}, the drain is clogged with plastic.",
                           "Blocked drain {loc}, the water is not going anywhere."],
        "damaged_sign": ["The road sign {loc} is broken and lying on the ground.",
                         "Traffic signboard {loc} has fallen down.",
                         "The stop sign {loc} is bent and not visible.",
                         "Direction board {loc} is damaged, nobody can read it."],
        "other": ["A tree has fallen {loc} and is blocking the footpath.",
                  "Stray cattle sit in the middle of the road {loc}.",
                  "Illegal encroachment on the footpath {loc}, people cannot walk.",
                  "Stray dogs are attacking people {loc}."],
    },
    "hi": {
        "pothole": ["{loc} सड़क पर बड़ा गड्ढा है।",
                    "{loc} सड़क पूरी टूट गई है, जगह-जगह गहरे गड्ढे हैं।",
                    "{loc} गड्ढों की वजह से दोपहिया वाहन फिसल रहे हैं।",
                    "सड़क में बहुत बड़ा गड्ढा हो गया है, {loc}।"],
        "streetlight": ["{loc} स्ट्रीट लाइट खराब है।",
                        "{loc} कई रातों से बत्ती नहीं जल रही, पूरा अंधेरा रहता है।",
                        "{loc} खंभे की लाइट बार-बार बंद हो जाती है।",
                        "स्ट्रीट लाइट बंद पड़ी है, {loc}।"],
        "garbage": ["{loc} कूड़े का ढेर लगा है, कोई उठाने नहीं आता।",
                    "{loc} बहुत कचरा पड़ा है, बदबू आ रही है।",
                    "{loc} कूड़ेदान भरा हुआ है, कचरा सड़क पर फैल गया है।",
                    "कई दिन से कूड़ा नहीं उठा, {loc}।"],
        "water_leak": ["{loc} पानी की पाइपलाइन फट गई है, साफ पानी बर्बाद हो रहा है।",
                       "{loc} पाइप से पानी लीक हो रहा है।",
                       "{loc} जल बोर्ड की पाइप टूटी है, सड़क पर पानी बह रहा है।",
                       "पाइप लीकेज की वजह से नल में पानी नहीं आ रहा, {loc}।"],
        "drain_blockage": ["{loc} नाली पूरी तरह बंद है।",
                           "{loc} नाला जाम है, गंदा पानी सड़क पर आ रहा है।",
                           "{loc} सीवर ओवरफ्लो हो रहा है, नाली में प्लास्टिक फँसा है।",
                           "नाली बंद होने से पानी निकल नहीं रहा, {loc}।"],
        "damaged_sign": ["{loc} सड़क का साइन बोर्ड टूटकर गिरा पड़ा है।",
                         "{loc} ट्रैफिक का बोर्ड गिर गया है।",
                         "{loc} स्टॉप का बोर्ड मुड़ गया है, दिखता नहीं।",
                         "दिशा बताने वाला बोर्ड खराब है, पढ़ा नहीं जाता, {loc}।"],
        "other": ["{loc} पेड़ गिर गया है, फुटपाथ बंद है।",
                  "{loc} सड़क के बीच आवारा पशु बैठे रहते हैं।",
                  "{loc} फुटपाथ पर अवैध कब्ज़ा है, लोग चल नहीं पाते।",
                  "आवारा कुत्ते लोगों पर हमला कर रहे हैं, {loc}।"],
    },
    "pa": {
        "pothole": ["{loc} ਸੜਕ ਉੱਤੇ ਵੱਡਾ ਟੋਆ ਹੈ।",
                    "{loc} ਸੜਕ ਪੂਰੀ ਟੁੱਟੀ ਹੋਈ ਹੈ, ਡੂੰਘੇ ਟੋਏ ਹਨ।",
                    "{loc} ਟੋਇਆਂ ਕਰਕੇ ਸਕੂਟਰ ਤਿਲਕ ਰਹੇ ਹਨ।",
                    "ਸੜਕ ਵਿੱਚ ਬਹੁਤ ਵੱਡਾ ਟੋਆ ਪੈ ਗਿਆ ਹੈ, {loc}।"],
        "streetlight": ["{loc} ਸਟ੍ਰੀਟ ਲਾਈਟ ਖਰਾਬ ਹੈ।",
                        "{loc} ਕਈ ਰਾਤਾਂ ਤੋਂ ਬੱਤੀ ਨਹੀਂ ਜਗ ਰਹੀ, ਪੂਰਾ ਹਨੇਰਾ ਹੈ।",
                        "{loc} ਖੰਭੇ ਦੀ ਲਾਈਟ ਵਾਰ-ਵਾਰ ਬੰਦ ਹੋ ਜਾਂਦੀ ਹੈ।",
                        "ਸਟ੍ਰੀਟ ਲਾਈਟ ਬੰਦ ਪਈ ਹੈ, {loc}।"],
        "garbage": ["{loc} ਕੂੜੇ ਦਾ ਢੇਰ ਲੱਗਿਆ ਹੈ, ਕੋਈ ਚੁੱਕਣ ਨਹੀਂ ਆਉਂਦਾ।",
                    "{loc} ਬਹੁਤ ਕੂੜਾ ਪਿਆ ਹੈ, ਬਦਬੂ ਆ ਰਹੀ ਹੈ।",
                    "{loc} ਕੂੜੇਦਾਨ ਭਰਿਆ ਹੋਇਆ ਹੈ, ਕੂੜਾ ਸੜਕ ਉੱਤੇ ਖਿੱਲਰਿਆ ਹੈ।",
                    "ਕਈ ਦਿਨਾਂ ਤੋਂ ਕੂੜਾ ਨਹੀਂ ਚੁੱਕਿਆ ਗਿਆ, {loc}।"],
        "water_leak": ["{loc} ਪਾਣੀ ਦੀ ਪਾਈਪਲਾਈਨ ਫਟ ਗਈ ਹੈ, ਸਾਫ਼ ਪਾਣੀ ਬਰਬਾਦ ਹੋ ਰਿਹਾ ਹੈ।",
                       "{loc} ਪਾਈਪ ਵਿੱਚੋਂ ਪਾਣੀ ਲੀਕ ਹੋ ਰਿਹਾ ਹੈ।",
                       "{loc} ਜਲ ਬੋਰਡ ਦੀ ਪਾਈਪ ਟੁੱਟੀ ਹੈ, ਸੜਕ ਉੱਤੇ ਪਾਣੀ ਵਗ ਰਿਹਾ ਹੈ।",
                       "ਪਾਈਪ ਲੀਕ ਹੋਣ ਕਰਕੇ ਟੂਟੀ ਵਿੱਚ ਪਾਣੀ ਨਹੀਂ ਆ ਰਿਹਾ, {loc}।"],
        "drain_blockage": ["{loc} ਨਾਲੀ ਪੂਰੀ ਬੰਦ ਹੈ।",
                           "{loc} ਨਾਲਾ ਜਾਮ ਹੈ, ਗੰਦਾ ਪਾਣੀ ਸੜਕ ਉੱਤੇ ਆ ਰਿਹਾ ਹੈ।",
                           "{loc} ਸੀਵਰ ਓਵਰਫਲੋ ਹੋ ਰਿਹਾ ਹੈ, ਨਾਲੀ ਵਿੱਚ ਪਲਾਸਟਿਕ ਫਸਿਆ ਹੈ।",
                           "ਨਾਲੀ ਬੰਦ ਹੋਣ ਕਰਕੇ ਪਾਣੀ ਨਹੀਂ ਨਿਕਲ ਰਿਹਾ, {loc}।"],
        "damaged_sign": ["{loc} ਸੜਕ ਦਾ ਸਾਈਨ ਬੋਰਡ ਟੁੱਟ ਕੇ ਡਿੱਗਿਆ ਪਿਆ ਹੈ।",
                         "{loc} ਟ੍ਰੈਫਿਕ ਦਾ ਬੋਰਡ ਡਿੱਗ ਗਿਆ ਹੈ।",
                         "{loc} ਸਟਾਪ ਦਾ ਬੋਰਡ ਮੁੜ ਗਿਆ ਹੈ, ਦਿਸਦਾ ਨਹੀਂ।",
                         "ਰਸਤਾ ਦੱਸਣ ਵਾਲਾ ਬੋਰਡ ਖਰਾਬ ਹੈ, ਪੜ੍ਹਿਆ ਨਹੀਂ ਜਾਂਦਾ, {loc}।"],
        "other": ["{loc} ਦਰਖ਼ਤ ਡਿੱਗ ਗਿਆ ਹੈ, ਫੁੱਟਪਾਥ ਬੰਦ ਹੈ।",
                  "{loc} ਸੜਕ ਦੇ ਵਿਚਕਾਰ ਅਵਾਰਾ ਪਸ਼ੂ ਬੈਠੇ ਰਹਿੰਦੇ ਹਨ।",
                  "{loc} ਫੁੱਟਪਾਥ ਉੱਤੇ ਨਜਾਇਜ਼ ਕਬਜ਼ਾ ਹੈ, ਲੋਕ ਤੁਰ ਨਹੀਂ ਸਕਦੇ।",
                  "ਅਵਾਰਾ ਕੁੱਤੇ ਲੋਕਾਂ ਨੂੰ ਵੱਢ ਰਹੇ ਹਨ, {loc}।"],
    },
    "mixed": {
        "pothole": ["{loc} road pe bahut bada gaddha hai",
                    "{loc} road poori tooti hui hai, deep potholes hain",
                    "{loc} gaddhon ki wajah se bikes slip ho rahi hain",
                    "road mein huge pothole ho gaya hai, {loc}"],
        "streetlight": ["{loc} streetlight kharab hai",
                        "{loc} kai raaton se light nahi jal rahi, full andhera hai",
                        "{loc} pole ki light baar baar off ho jaati hai",
                        "street light band padi hai yaar, {loc}"],
        "garbage": ["{loc} kooda pada hai, koi uthane nahi aata",
                    "{loc} bahut garbage jama hai, smell aa rahi hai",
                    "{loc} dustbin overflow ho raha hai, kachra road pe fail gaya",
                    "kai din se garbage pickup nahi hua, {loc}"],
        "water_leak": ["{loc} paani ki pipeline burst ho gayi hai, saaf paani waste ho raha hai",
                       "{loc} pipe se paani leak ho raha hai",
                       "{loc} Jal Board ka pipe toota hai, road pe paani beh raha hai",
                       "pipe leakage ki wajah se tap mein paani nahi aa raha, {loc}"],
        "drain_blockage": ["{loc} naali poori block hai",
                           "{loc} nala choke hai, ganda paani road pe aa raha hai",
                           "{loc} sewer overflow ho raha hai, drain mein plastic fasa hai",
                           "drain block hone se paani nikal nahi raha, {loc}"],
        "damaged_sign": ["{loc} road ka sign board toot ke gira pada hai",
                         "{loc} traffic board gir gaya hai",
                         "{loc} stop sign mud gaya hai, dikhta nahi",
                         "direction board kharab hai, padh nahi sakte, {loc}"],
        "other": ["{loc} ped gir gaya hai, footpath block hai",
                  "{loc} road ke beech mein stray cattle baithe rehte hain",
                  "{loc} footpath pe illegal encroachment hai, log chal nahi paate",
                  "stray dogs logon pe attack kar rahe hain, {loc}"],
    },
}

# Cue sentences. A cue flag is True exactly when one of its sentences is in the text
# (user decision, Step 0.5: cues are what the text says, not what the situation implies).
CUES = {
    "children": {
        "en": ["Children walk past here to school every day.",
               "Small kids play here, it is dangerous for them."],
        "hi": ["यहाँ से रोज़ बच्चे स्कूल जाते हैं।", "छोटे बच्चे यहाँ खेलते हैं, उनके लिए खतरा है।"],
        "pa": ["ਇੱਥੋਂ ਰੋਜ਼ ਬੱਚੇ ਸਕੂਲ ਜਾਂਦੇ ਹਨ।", "ਛੋਟੇ ਬੱਚੇ ਇੱਥੇ ਖੇਡਦੇ ਹਨ, ਉਹਨਾਂ ਲਈ ਖ਼ਤਰਾ ਹੈ।"],
        "mixed": ["yahan se roz bacche school jaate hain",
                  "chhote bacche yahan khelte hain, unke liye risky hai"],
    },
    "accident": {
        "en": ["A scooter rider fell here yesterday.", "There was an accident here last week."],
        "hi": ["कल यहाँ एक स्कूटर वाला गिर गया।", "पिछले हफ्ते यहाँ एक्सीडेंट हुआ था।"],
        "pa": ["ਕੱਲ੍ਹ ਇੱਥੇ ਇੱਕ ਸਕੂਟਰ ਵਾਲਾ ਡਿੱਗ ਪਿਆ।", "ਪਿਛਲੇ ਹਫ਼ਤੇ ਇੱਥੇ ਐਕਸੀਡੈਂਟ ਹੋਇਆ ਸੀ।"],
        "mixed": ["kal yahan ek scooty wala gir gaya", "last week yahan accident hua tha"],
    },
    "flooding": {
        "en": ["After the rain the whole road is waterlogged.",
               "Water is standing knee-deep on the road."],
        "hi": ["बारिश के बाद पूरी सड़क पर पानी भर गया है।", "सड़क पर घुटनों तक पानी भरा है।"],
        "pa": ["ਮੀਂਹ ਤੋਂ ਬਾਅਦ ਪੂਰੀ ਸੜਕ ਉੱਤੇ ਪਾਣੀ ਭਰ ਗਿਆ ਹੈ।", "ਸੜਕ ਉੱਤੇ ਗੋਡਿਆਂ ਤੱਕ ਪਾਣੀ ਖੜ੍ਹਾ ਹੈ।"],
        "mixed": ["baarish ke baad poori road pe paani bhar gaya hai",
                  "road pe ghutno tak paani hai"],
    },
}

OPENERS = {
    "en": ["", "", "Sir, ", "Hello, ", "Please help. ", "Complaint: "],
    "hi": ["", "", "सर, ", "नमस्ते, ", "कृपया ध्यान दें, "],
    "pa": ["", "", "ਸਰ, ", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਜੀ, ", "ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ, "],
    "mixed": ["", "", "sir ", "hello ", "plz ", "bhai "],
}

DURATIONS = {
    "en": "It has been like this for {n} days.",
    "hi": "{n} दिन से ऐसा ही है।",
    "pa": "{n} ਦਿਨਾਂ ਤੋਂ ਇਹੀ ਹਾਲ ਹੈ।",
    "mixed": "{n} din se aisa hi hai",
}

CLOSERS = {
    "en": ["Please fix it soon.", "Kindly take action."],
    "hi": ["जल्दी ठीक करवाइए।", "कृपया कार्रवाई करें।"],
    "pa": ["ਜਲਦੀ ਠੀਕ ਕਰਵਾਓ ਜੀ।", "ਕਿਰਪਾ ਕਰਕੇ ਕਾਰਵਾਈ ਕਰੋ।"],
    "mixed": ["jaldi theek karwao please", "kuch karo yaar"],
}

# Hinglish spelling variants applied at random, as people actually type them.
HINGLISH_VARIANTS = {"hai": "h", "nahi": "nhi", "please": "pls", "paani": "pani",
                     "bahut": "bohot", "yahan": "yaha"}

# Spam / irrelevant messages: no location, no issue.
SPAM = {
    "en": ["Hi, is this the right number for train tickets?",
           "Earn Rs 50,000 per week from home, call now!!!",
           "good morning to all", "asdf test test 123"],
    "hi": ["सबको सुप्रभात 🙏", "घर बैठे पैसे कमाएँ, अभी कॉल करें",
           "यह ऐप कैसे चलाते हैं?", "क्या यहाँ से बिजली का बिल भर सकते हैं?"],
    "pa": ["ਸਭ ਨੂੰ ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "ਘਰ ਬੈਠੇ ਪੈਸੇ ਕਮਾਓ, ਹੁਣੇ ਕਾਲ ਕਰੋ", "ਇਹ ਐਪ ਕਿਵੇਂ ਚਲਾਉਂਦੇ ਹਾਂ?"],
    "mixed": ["bhai cricket ka score kya hai", "free recharge chahiye toh link pe click karo",
              "test msg ignore", "app bahut slow hai yaar"],
}
