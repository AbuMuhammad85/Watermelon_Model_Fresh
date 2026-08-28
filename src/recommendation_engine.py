import os
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RecommendationEngine:
    def __init__(self):
        # Dictionary storing recommendations by condition and severity
        self.recommendations = {
            "Anthracnose": {
                "Low": {
                    "English": "Prune and destroy the few affected lower leaves. Apply mulching to prevent fungal spores from splashing up from the soil. Avoid working in the field while the plants are wet.",
                    "Hausa": "Yanke ganyayyaki kalilan na kasa da abin ya shafa sannan a kona su. A shimfida busasshiyar ciyawa (mulching) don hana feshin kwayar naman gwari daga kasa. Guji aiki a gona lokacin da ganye ke da jikakken ruwa."
                },
                "Moderate": {
                    "English": "Remove diseased leaves. Improve spacing between vines to increase air circulation. Apply protective organic or copper-based sprays to limit spread. Switch from overhead irrigation to ground/drip irrigation.",
                    "Hausa": "Cire ganyen da suka kamu da ciwon. Kara sarari tsakanin shuke-shuke don samun shigar iska. Fesa maganin rigakafi na copper don dakatar da yaduwa. Canza ban-ruwa na sama zuwa na karkashin kasa."
                },
                "High": {
                    "English": "Severe fungal spread. Immediately pull up and destroy the heavily damaged vines to protect the rest of the crop. Do not compost this debris. Spot-treat neighboring vines with appropriate fungicides and contact local extension officers.",
                    "Hausa": "Naman gwari ya yadu sosai. Tumbuke shukar da ta lalace da gaggawa sannan a lalata ta don kare sauran amfanin gona. Kada a shigar da ita cikin takin zamani. Fesa maganin naman gwari a shuke-shuken makwabta sannan a tuntubi jami'an gona."
                }
            },
            "Downy_Mildew": {
                "Low": {
                    "English": "Ensure watering is done only in the morning to allow leaves to dry quickly. Prune lower shaded leaves to enhance air flow under the canopy.",
                    "Hausa": "Tabbatar ana ban-ruwa da safe kawai don ganye su bushe da wuri. Yanke tsofaffin ganyen kasa don bari iska ta shiga karkashin ganye."
                },
                "Moderate": {
                    "English": "Prune and destroy leaves showing heavy yellow spotting. Apply preventative copper or botanical fungicides. Avoid overhead irrigation.",
                    "Hausa": "Yanke ganyen da ke da digage masu ruwan dorawa da yawa sannan a kona su. Fesa maganin naman gwari na copper ko na gargajiya. Guji ban-ruwa ta sama."
                },
                "High": {
                    "English": "Foliage is severely damaged. Destroy heavily infected plants immediately. Apply systemic or protective fungicides to all neighboring watermelon plants to control airborne spore spread. Consult agricultural authorities.",
                    "Hausa": "Ganyaye sun lalace sosai. Halakar da shukar da ta kamu da wuri. Fesa maganin naman gwari na rigakafi a sauran shuke-shuken kankana na kusa don hana yaduwar kwayoyin cutar ta iska. Tuntubi masana gona."
                }
            },
            "Mosaic_Virus": {
                "Low": {
                    "English": "Isolate the plant and monitor closely. Check for aphid vectors (look for tiny insects on leaf undersides). Disinfect hands and tools with soap/bleach solution after contact.",
                    "Hausa": "Ware wannan shukar kuma a kula da ita sosai. Duba ko akwai kwari (aphids) a bayan ganye. Wanke hannaye da kayan aiki da sabulu bayan taba shukar."
                },
                "Moderate": {
                    "English": "Uproot the infected plant immediately and burn or bury it deep. Control aphids on nearby plants using neem oil or soap sprays to prevent virus transmission. Clean all gardening tools thoroughly.",
                    "Hausa": "Tumbuke shukar da ta kamu da wuri sannan a kona ko a binne ta da zurfi. Kashe kwari (aphids) a shuke-shuken kusa ta amfani da man rini (neem oil) don hana yada virus. Wanke kayan aiki sosai."
                },
                "High": {
                    "English": "Viral infection has spread widely. Uproot all plants showing mosaic/crinkle symptoms and burn them away from the field. Implement strict vector control (insecticide) across the entire watermelon plot. Avoid touching healthy plants.",
                    "Hausa": "Virus ya yadu sosai. Tumbuke dukkan shuke-shuken da ke nuna alamun mosaic da gaggawa sannan a kona su nesa da gona. Fesa maganin kashe kwari a daukacin gonar kankana don hana yaduwa. Guji taba lafiyayyar shuka."
                }
            },
            "Healthy": {
                "Low": {
                    "English": "Plant is healthy. Continue standard irrigation and weed control. Keep monitoring leaves weekly.",
                    "Hausa": "Shuka tana da lafiya. Ci gaba da ban-ruwa da ciyawa. Ci gaba da duba ganyaye duk mako."
                },
                "Moderate": {
                    "English": "Plant is healthy. Apply compost or organic mulch to retain moisture and sustain vigorous growth.",
                    "Hausa": "Shuka tana da lafiya. A shimfida takin gargajiya ko busasshen ganye (mulch) don rike danshi da inganta girma."
                },
                "High": {
                    "English": "Plant is highly healthy and strong. Maintain your current practices. Keep records of your fertilizer and watering schedules to replicate success.",
                    "Hausa": "Shuka tana da koshin lafiya da karfi sosai. Ci gaba da kula da ita kamar yadda kake yi. Rike bayanan takawa da ban-ruwa don maimaita wannan nasarar."
                }
            }
        }
        
        # General non-diagnosis recommendations
        self.retake_recommendations = {
            "low_confidence": {
                "English": "Low diagnostic confidence. Please retake the photo. Ensure the leaf is centered, lies relatively flat, has bright indirect lighting, and there is minimal soil or background weeds visible.",
                "Hausa": "Karancin tabbaci kan binciken. Da fatan sake daukar hoton. Tabbatar ganyen yana tsakiya, a shimfide, akwai haske mai kyau, kuma babu kasa ko ciyayi da yawa a bayansa."
            },
            "quality_failed": {
                "English": "Image quality check failed. The photo is either too blurry, too dark/bright, or lacks sufficient green leaf area. For best results, retake the photo in good lighting, focus clearly on a single leaf, and hold the camera closer.",
                "Hausa": "Hoton bai cika ka'idojin inganci ba. Hoton ya yi dishi-dishi, ya yi duhu/haske da yawa, ko kuma babu isasshen koran ganye. Don sakamako mai kyau, sake dauka a cikin haske, kuma a mayar da hankali kan ganye guda daya."
            }
        }

    def get_recommendation(self, diagnosis, confidence_status, severity="Low", lang="English"):
        """
        Retrieves localized recommendation based on pipeline status, diagnosis, and severity.
        
        Args:
            diagnosis (str): Predicted class name ("Anthracnose", "Downy_Mildew", "Mosaic_Virus", "Healthy")
            confidence_status (str): "HIGH_CONFIDENCE" or "LOW_CONFIDENCE"
            severity (str): "Low", "Moderate", or "High"
            lang (str): "English" or "Hausa"
            
        Returns:
            dict: Structured recommendation message and safety warning.
        """
        # Ensure correct formatting
        lang = "English" if lang not in ["English", "Hausa"] else lang
        severity = "Low" if severity not in ["Low", "Moderate", "High"] else severity
        
        # Standardize diagnosis name
        normalized_diagnosis = diagnosis.replace(" ", "_")
        
        # Check confidence first
        if confidence_status == "LOW_CONFIDENCE":
            rec_text = self.retake_recommendations["low_confidence"][lang]
            return {
                "recommendation": rec_text,
                "is_actionable": False,
                "type": "retake_request",
                "severity_label": "N/A"
            }
            
        # Get diagnosis-specific recommendation
        if normalized_diagnosis in self.recommendations:
            rec_text = self.recommendations[normalized_diagnosis][severity][lang]
            is_actionable = normalized_diagnosis != "Healthy"
            rec_type = "treatment" if is_actionable else "maintenance"
            return {
                "recommendation": rec_text,
                "is_actionable": is_actionable,
                "type": rec_type,
                "severity_label": severity
            }
        else:
            # Fallback
            rec_text = "No recommendation available." if lang == "English" else "Babu shawarwarin da ke akwai."
            return {
                "recommendation": rec_text,
                "is_actionable": False,
                "type": "unknown",
                "severity_label": severity
            }
            
    def get_quality_warning(self, reasons, lang="English"):
        """
        Generates user-friendly warnings for failing quality gates.
        """
        lang = "English" if lang not in ["English", "Hausa"] else lang
        if not reasons:
            return ""
            
        warning_map = {
            "blurry": {
                "English": "The image appears blurry. Please make sure the camera is in focus.",
                "Hausa": "Hoton ya yi dishi-dishi (blurry). Tabbatar cyamarar tana kan ganye sosai."
            },
            "too_dark": {
                "English": "The image is too dark. Try turning on the flash or moving to a brighter area.",
                "Hausa": "Hoton ya yi duhu da yawa. Kunna tocila ko ka koma inda akwai haske."
            },
            "too_bright": {
                "English": "The image is too bright or has glare. Shield the leaf from direct sunlight.",
                "Hausa": "Hoton yana da haske ko hasken rana ya yi yawa. Rufe ganyen daga hasken rana kai tsaye."
            },
            "poor_framing": {
                "English": "Insufficient green leaf area visible. Move closer and center a single leaf in the photo.",
                "Hausa": "Babu isasshen koran ganye a hoton. Matso kusa sannan ka saita ganye guda daya a tsakiya."
            },
            "leaf_too_small": {
                "English": "The leaf appears too small or too far away. Please move closer and reposition the camera so the leaf fills more of the screen.",
                "Hausa": "Ganyen ya yi kankanta ko kuma ya yi nesa sosai a cikin hoton. Da fatan a matso kusa sannan a sake dauka domin ganyen ya cika hoton."
            }
        }
        
        translated_reasons = []
        for r in reasons:
            if r in warning_map:
                translated_reasons.append(warning_map[r][lang])
                
        return " \n".join(translated_reasons)
