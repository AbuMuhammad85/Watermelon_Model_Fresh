import os
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DISEASE_KNOWLEDGE_BASE = {
    "Disclaimer": {
        "English": "DISCLAIMER: This diagnostic tool provides general agricultural guidance for educational purposes only. It is not professional agronomic advice or certified scientific recommendations. Always consult a local agricultural extension officer or certified agronomist before applying chemical treatments, pesticides, or executing major field alterations.",
        "Hausa": "GARGAƊI: Wannan manhaja tana ba da shawarwarin gona ne don koyo kawai. Ba shawarar kwararrun masanan gona ba ce. A koda yaushe a tuntubi jami'an gona na yankinku (extension officers) kafin amfani da magungunan feshin bishiyoyi ko daukar manyan matakai a gona."
    },
    "Anthracnose": {
        "name": {
            "English": "Anthracnose",
            "Hausa": "Ciwon Digo-digo (Anthracnose)"
        },
        "symptoms": {
            "English": "Dark, water-soaked circular spots on leaves that turn brown or black, dry up and develop shatter holes. Elongated sunken lesions on stems and fruit.",
            "Hausa": "Digage masu ruwa-ruwa a jikin ganye da ke komawa launin ruwan kasa ko baki, bushewa sannan su yage su bar rami. Tabo mai zurfi a jikin kara da yayan kankana."
        },
        "causes": {
            "English": "Fungal pathogen (Colletotrichum orbiculare) spread by warm, wet weather, splashing rain, infected seed, or diseased crop debris in soil.",
            "Hausa": "Naman gwari (fungus) da ke yaduwa lokacin zafi da ruwa, feshin ruwan sama, gurbataccen iri, ko ragowar tsohuwar shuka a kasa."
        },
        "prevention": {
            "English": "Use certified disease-free seeds. Practice 3-year crop rotation (avoid planting other cucurbits/melons in the same spot). Clear and burn crop debris after harvest. Space plants adequately to promote airflow and foliage drying.",
            "Hausa": "Yin amfani da ingantaccen iri mara ciwo. Jujjuya amfanin gona na tsawon shekaru 3 (guji shuka kankana ko kabewa a wuri daya). Share ragowar shuka bayan girbi. Ba da sarari tsakanin shuka don iska ta shiga."
        },
        "management": {
            "English": "Prune and destroy infected lower leaves. Avoid overhead watering to keep leaves dry. If chemical treatment is necessary, apply registered copper-based protective fungicides under expert advice, strictly adhering to label warnings.",
            "Hausa": "Yanke ganyen kasa da suka kamu sannan a kona su. Guji shayarwa ta sama don barin ganye su bushe. Idan ya zama dole, a fesa maganin kashe naman gwari na copper bisa jagorancin masani da bin ka'idoji."
        },
        "farmer_guidance": {
            "English": "Inspect fields weekly, especially the lower canopy after rainfall. Do not work in the fields when the plants are wet, as this rapidly spreads fungal spores.",
            "Hausa": "Bincika gona a kalla sau daya a mako, musamman ganyen kasa bayan ruwan sama. Guji aiki a gona lokacin da ganyen kankana ke da jikakken ruwa."
        }
    },
    "Downy_Mildew": {
        "name": {
            "English": "Downy Mildew",
            "Hausa": "Ciwon Downy Mildew (Farin Kura)"
        },
        "symptoms": {
            "English": "Angular, pale green to bright yellow spots on the upper leaf surface, strictly bounded by leaf veins. Under humid conditions, a purplish-gray downy mold appears on the lower leaf surface.",
            "Hausa": "Digage masu kusurwa da launin dorawa-kore a saman ganye, da ke tsayawa a jijiyoyin ganye. Lokacin danshi, farin kura ko toka mai launin toka-toka tana fitowa a bayan ganye."
        },
        "causes": {
            "English": "Water mold pathogen (Pseudoperonospora cubensis) spread by wind-borne spores and high relative humidity or prolonged leaf wetness.",
            "Hausa": "Kwayar cutar ruwa (water mold) da ke yaduwa ta hanyar iska da danshi mai yawa ko kuma jikewar ganye na tsawon lokaci."
        },
        "prevention": {
            "English": "Choose resistant watermelon cultivars. Avoid overhead sprinkler irrigation; apply drip or furrow irrigation instead. Space rows wide enough for full sun exposure and ventilation. Rotate crops annually.",
            "Hausa": "Zabi irin kankana mai jure ciwo. Guji ban-ruwa ta sama (sprinkler); yi amfani da ban-ruwa na karkashin shuka (drip). Bada babban sarari tsakanin layuka don rana da iska su shiga."
        },
        "management": {
            "English": "Quickly remove initial infected leaves to slow down the spread. Apply protective biological or chemical fungicides early in the season if the area has a history of Downy Mildew. Follow localized extension advisory.",
            "Hausa": "Cire ganyen farko da suka kamu da wuri don rage saurin yaduwa. Fesa maganin kashe naman gwari na rigakafi tun farkon kakar shuka idan gonar tana da tarihin wannan cutar."
        },
        "farmer_guidance": {
            "English": "This pathogen does not survive winter/dry seasons without living hosts, but spores travel long distances on wind currents. Monitor local outbreak reports closely.",
            "Hausa": "Wannan cutar ba ta rayuwa lokacin rani ba tare da korayen shuke-shuke ba, amma iska tana kawo kwayoyin cutar daga nesa. Kula da sanarwar bullar cutar a yankinku."
        }
    },
    "Mosaic_Virus": {
        "name": {
            "English": "Mosaic Virus",
            "Hausa": "Ciwon Mosaic Virus"
        },
        "symptoms": {
            "English": "Mottled light and dark green patterns (mosaic), blistering, puckering, and severe distortion of leaves. Stunted vine growth and small, misshapen fruit with bumpy skins.",
            "Hausa": "Lankwashewa, tabo na kore da fari ko rawaya (mosaic), da gurguncewar ganyen kankana. Dakatar da girman shuka da lalacewar siffar kankana da kuraje a jiki."
        },
        "causes": {
            "English": "Plant viruses (such as Cucumber Mosaic Virus or Watermelon Mosaic Virus) transmitted mechanically via tools, hands, or biological vectors like aphids.",
            "Hausa": "Kwayar cutar virus (kamar Cucumber Mosaic Virus) da kwari (musamman kudan ganye/aphids) ke yadawa, ko ta hanyar kayan aiki da hannun manoma."
        },
        "prevention": {
            "English": "Strictly control insect vectors (aphids) using reflective mulches or organic insecticidal soaps. Eradicate weed hosts from field borders. Wash hands and sterilize tools (using a 10% bleach solution or soap) between plants. Plant virus-resistant varieties.",
            "Hausa": "Kula da kashe kwari (kudan ganye) ta amfani da sabulun kashe kwari. Cire ciyawa a gefen gona. Wanke hannaye da goge kayan aiki da sinadarin tsaftacewa kafin taba wata shukar."
        },
        "management": {
            "English": "There is no chemical cure for plant viral diseases. Immediately uproot and burn/bury infected plants. Do not compost infected debris. Wash hands thoroughly after handling infected vines.",
            "Hausa": "Babu maganin warkewa ga cutar virus ta shuka. Tumbuke shukar da ta kamu da wuri sannan a kona ta ko a binne ta. Kada a shigar da ita cikin takin zamani (compost)."
        },
        "farmer_guidance": {
            "English": "Early roguing (uprooting) of infected plants is the single most critical step to save the rest of your watermelon crop. Never touch healthy vines immediately after handling a diseased plant.",
            "Hausa": "Tumbuke shukar da ta kamu da wuri shine mafi mahimmancin mataki don ceton sauran gonarka. Kada ka taba lafiyayyar shuka bayan ka gama taba wacce ta kamu da ciwon."
        }
    },
    "Healthy": {
        "name": {
            "English": "Healthy",
            "Hausa": "Lafiyayyen Ganye"
        },
        "symptoms": {
            "English": "Broad, vibrant dark green leaves with uniform coloration, no yellowing, spots, or leaf curling. Robust vine extensions and uniform blossom development.",
            "Hausa": "Ganyaye masu fadi da launin kore mara canji, babu digo-digo, babu rawaya ko lankwashewa. Kyawun bishiya da fitowar furanni masu kyau."
        },
        "causes": {
            "English": "Optimal soil health, proper nutrient balance, correct watering schedules, and successful pest/disease exclusion.",
            "Hausa": "Kyawun kasar gona, daidaitaccen taki, isasshen ban-ruwa, da nasarar kare gona daga kwari da cututtuka."
        },
        "prevention": {
            "English": "Maintain current farm hygiene and cultural practices. Continue weekly scouting of leaf undersides and keep irrigation consistent to avoid stress.",
            "Hausa": "Ci gaba da kiyaye tsaftar gona da kula da shuka yadda ya kamata. Ci gaba da duba bayan ganye duk mako don maganin kwari da wuri."
        },
        "management": {
            "English": "No treatment required. Maintain standard irrigation and weed control. Apply compost tea or organic mulches to sustain growth.",
            "Hausa": "Babu bukatar magani. Ci gaba da ban-ruwa da ciyawa. Ana iya saka takin gargajiya don ci gaba da karfafa shuka."
        },
        "farmer_guidance": {
            "English": "A healthy leaf indicates excellent management. Keep documentation of planting dates, fertilizer application times, and weather patterns to duplicate success next season.",
            "Hausa": "Lafiyayyen ganye yana nuna kyakkyawan kula. Rubuta ranar shuka, takawa, da yanayin ruwa don samun irin wannan nasarar a kakar shuka ta gaba."
        }
    }
}

def get_disease_info(condition):
    """
    Safely retrieves knowledge base dictionary for a given condition.
    """
    # Normalize condition string
    normalized = condition.replace(" ", "_")
    if normalized in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[normalized]
    return None
