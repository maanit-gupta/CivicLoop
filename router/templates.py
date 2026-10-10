"""Hand-written phrase bank for ambiguous routing cases (Step 2.1). Data only, no logic.

Each scenario is one physical situation that two departments could own. The SHARED sentence
describes the situation and says nothing about which department; one CUE sentence decides it.
`issue_types` are the two labels intake could plausibly give such a complaint: the observed
issue_type is drawn from them independently of the true department, so issue_type alone
cannot route these cases. `{loc}` is filled with a location phrase from the corpus.

Labelling rules (user decision, Step 2.1):
- water_on_road:    clean water from a pipe OR sewage from a manhole -> Water Supply (it owns
                    sewers, like the Delhi Jal Board); rainwater not draining -> Stormwater Drains
- fallen_pole:      live wires, sparks, current -> Electrical; dead pole only blocking the road
                    -> Roads (clearing the carriageway)
- garbage_in_drain: garbage inside the drain channel -> Stormwater Drains; garbage on the bank
                    or the road beside it -> Sanitation. Water flow does not matter.
"""

SCENARIOS = {
    "water_on_road": {
        "issue_types": ["water_leak", "drain_blockage"],
        "shared": {
            "en": ["Water is flowing across the road {loc}.",
                   "There is a stream of water on the road {loc}, it does not stop."],
            "hi": ["{loc} सड़क पर पानी बह रहा है।",
                   "{loc} सड़क पर लगातार पानी की धार बह रही है।"],
            "pa": ["{loc} ਸੜਕ ਉੱਤੇ ਪਾਣੀ ਵਗ ਰਿਹਾ ਹੈ।",
                   "{loc} ਸੜਕ ਉੱਤੇ ਲਗਾਤਾਰ ਪਾਣੀ ਦੀ ਧਾਰ ਵਗ ਰਹੀ ਹੈ।"],
            "mixed": ["{loc} road pe paani beh raha hai",
                      "{loc} road pe lagatar paani ki dhaar beh rahi hai"],
        },
        "cues": {
            "Water Supply": {
                "en": ["It is clean water, a supply pipe seems to have burst underneath.",
                       "It is sewage overflowing from the sewer manhole, it stinks."],
                "hi": ["पानी साफ है, लगता है नीचे सप्लाई की पाइप फट गई है।",
                       "सीवर के मैनहोल से गंदा पानी उफन रहा है, बदबू आ रही है।"],
                "pa": ["ਪਾਣੀ ਸਾਫ਼ ਹੈ, ਲੱਗਦਾ ਹੈ ਹੇਠਾਂ ਸਪਲਾਈ ਦੀ ਪਾਈਪ ਫਟ ਗਈ ਹੈ।",
                       "ਸੀਵਰ ਦੇ ਮੈਨਹੋਲ ਵਿੱਚੋਂ ਗੰਦਾ ਪਾਣੀ ਉੱਛਲ ਰਿਹਾ ਹੈ, ਬਦਬੂ ਆ ਰਹੀ ਹੈ।"],
                "mixed": ["paani saaf hai, lagta hai neeche supply pipe phat gayi hai",
                          "sewer ke manhole se ganda paani ubal raha hai, smell aa rahi hai"],
            },
            "Stormwater Drains": {
                "en": ["It is rainwater, the roadside drain is not taking it away.",
                       "The rain drain grating is blocked, so the rainwater has nowhere to go."],
                "hi": ["यह बारिश का पानी है, सड़क किनारे की नाली इसे निकाल नहीं रही।",
                       "बरसाती नाले की जाली बंद है, बारिश का पानी कहीं नहीं जा रहा।"],
                "pa": ["ਇਹ ਮੀਂਹ ਦਾ ਪਾਣੀ ਹੈ, ਸੜਕ ਕਿਨਾਰੇ ਦੀ ਨਾਲੀ ਇਸਨੂੰ ਕੱਢ ਨਹੀਂ ਰਹੀ।",
                       "ਬਰਸਾਤੀ ਨਾਲੇ ਦੀ ਜਾਲੀ ਬੰਦ ਹੈ, ਮੀਂਹ ਦਾ ਪਾਣੀ ਕਿਤੇ ਨਹੀਂ ਜਾ ਰਿਹਾ।"],
                "mixed": ["ye baarish ka paani hai, road ke side wali naali isko nikaal nahi rahi",
                          "barsaati nale ki jaali band hai, baarish ka paani kahin nahi ja raha"],
            },
        },
    },
    "fallen_pole": {
        "issue_types": ["streetlight", "other"],
        "shared": {
            "en": ["An electric pole has fallen {loc}.",
                   "A streetlight pole has collapsed {loc}."],
            "hi": ["{loc} बिजली का खंभा गिर गया है।",
                   "{loc} स्ट्रीट लाइट का खंभा टूटकर गिर गया है।"],
            "pa": ["{loc} ਬਿਜਲੀ ਦਾ ਖੰਭਾ ਡਿੱਗ ਗਿਆ ਹੈ।",
                   "{loc} ਸਟ੍ਰੀਟ ਲਾਈਟ ਦਾ ਖੰਭਾ ਟੁੱਟ ਕੇ ਡਿੱਗ ਗਿਆ ਹੈ।"],
            "mixed": ["{loc} bijli ka pole gir gaya hai",
                      "{loc} streetlight ka khamba toot ke gir gaya hai"],
        },
        "cues": {
            "Electrical": {
                "en": ["Wires are hanging from it and sparking.",
                       "A live wire is lying on the ground, there is current in it."],
                "hi": ["उससे तार लटक रहे हैं और चिंगारी निकल रही है।",
                       "ज़मीन पर नंगा तार पड़ा है, उसमें करंट है।"],
                "pa": ["ਉਸ ਤੋਂ ਤਾਰਾਂ ਲਟਕ ਰਹੀਆਂ ਹਨ ਅਤੇ ਚੰਗਿਆੜੇ ਨਿਕਲ ਰਹੇ ਹਨ।",
                       "ਜ਼ਮੀਨ ਉੱਤੇ ਨੰਗੀ ਤਾਰ ਪਈ ਹੈ, ਉਸ ਵਿੱਚ ਕਰੰਟ ਹੈ।"],
                "mixed": ["usse wires latak rahe hain aur sparking ho rahi hai",
                          "zameen pe nanga wire pada hai, usme current hai"],
            },
            "Roads": {
                "en": ["The power was already cut, but the pole is lying across the road and "
                       "blocking traffic.",
                       "Broken concrete from the pole is lying on the road, vehicles cannot pass."],
                "hi": ["बिजली पहले से कटी है, पर खंभा सड़क के बीच पड़ा है और ट्रैफिक रुका है।",
                       "खंभे का टूटा सीमेंट सड़क पर पड़ा है, गाड़ियाँ नहीं निकल पा रहीं।"],
                "pa": ["ਬਿਜਲੀ ਪਹਿਲਾਂ ਹੀ ਕੱਟੀ ਹੋਈ ਹੈ, ਪਰ ਖੰਭਾ ਸੜਕ ਦੇ ਵਿਚਕਾਰ ਪਿਆ ਹੈ ਅਤੇ ਟ੍ਰੈਫਿਕ ਰੁਕਿਆ ਹੈ।",
                       "ਖੰਭੇ ਦਾ ਟੁੱਟਿਆ ਸੀਮਿੰਟ ਸੜਕ ਉੱਤੇ ਪਿਆ ਹੈ, ਗੱਡੀਆਂ ਨਹੀਂ ਲੰਘ ਸਕਦੀਆਂ।"],
                "mixed": ["light pehle se kati hai, par pole road ke beech pada hai aur traffic "
                          "ruka hai",
                          "pole ka toota cement road pe pada hai, gaadiyan nikal nahi paa rahi"],
            },
        },
    },
    "garbage_in_drain": {
        "issue_types": ["garbage", "drain_blockage"],
        "shared": {
            # "at the nala", not "into": the shared sentence must not say inside or beside.
            "en": ["People keep dumping garbage at the nala {loc}.",
                   "There is a lot of trash at the drain {loc}."],
            "hi": ["{loc} लोग नाले पर कूड़ा फेंकते रहते हैं।",
                   "{loc} नाले पर बहुत कचरा पड़ा है।"],
            "pa": ["{loc} ਲੋਕ ਨਾਲੇ ਉੱਤੇ ਕੂੜਾ ਸੁੱਟਦੇ ਰਹਿੰਦੇ ਹਨ।",
                   "{loc} ਨਾਲੀ ਉੱਤੇ ਬਹੁਤ ਕੂੜਾ ਪਿਆ ਹੈ।"],
            "mixed": ["{loc} log nale pe kooda phenkte rehte hain",
                      "{loc} naali pe bahut kachra pada hai"],
        },
        "cues": {
            "Sanitation": {
                "en": ["The garbage is not inside the drain, it is heaped on the road beside it.",
                       "A heap of waste is lying on the bank of the nala, please get it picked "
                       "up."],
                "hi": ["कूड़ा नाली के अंदर नहीं, बगल की सड़क पर ढेर लगा है।",
                       "नाले के किनारे पर कूड़े का ढेर पड़ा है, उसे उठवाइए।"],
                "pa": ["ਕੂੜਾ ਨਾਲੀ ਦੇ ਅੰਦਰ ਨਹੀਂ, ਨਾਲ ਦੀ ਸੜਕ ਉੱਤੇ ਢੇਰ ਲੱਗਿਆ ਹੈ।",
                       "ਨਾਲੇ ਦੇ ਕਿਨਾਰੇ ਉੱਤੇ ਕੂੜੇ ਦਾ ਢੇਰ ਪਿਆ ਹੈ, ਉਸਨੂੰ ਚੁਕਵਾਓ।"],
                "mixed": ["kachra naali ke andar nahi, bagal wali road pe dher laga hai",
                          "nale ke kinare pe kachre ka dher pada hai, uthwa do"],
            },
            "Stormwater Drains": {
                "en": ["The garbage is stuck inside the drain itself, filling the channel.",
                       "Plastic bags and waste are floating inside the nala."],
                "hi": ["कूड़ा नाली के अंदर ही फँसा है, चैनल में भरा है।",
                       "नाले के अंदर पॉलिथीन और कचरा तैर रहा है।"],
                "pa": ["ਕੂੜਾ ਨਾਲੀ ਦੇ ਅੰਦਰ ਹੀ ਫਸਿਆ ਹੈ, ਚੈਨਲ ਵਿੱਚ ਭਰਿਆ ਹੈ।",
                       "ਨਾਲੇ ਦੇ ਅੰਦਰ ਪੋਲੀਥੀਨ ਅਤੇ ਕੂੜਾ ਤੈਰ ਰਿਹਾ ਹੈ।"],
                "mixed": ["kachra naali ke andar hi fasa hai, channel mein bhara hai",
                          "nale ke andar polythene aur kachra tair raha hai"],
            },
        },
    },
}
