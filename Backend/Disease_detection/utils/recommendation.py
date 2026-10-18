# =========================================================
# POULTRY DISEASE RECOMMENDATIONS
# =========================================================


DISEASE_RECOMMENDATIONS = {


    "Coccidiosis": {

        "severity": "Medium",

        "actions": [

            "Separate affected birds from healthy birds",

            "Keep litter dry and clean",

            "Monitor droppings regularly",

            "Consult a veterinarian for proper treatment"

        ],

        "prevention": [

            "Maintain clean drinking water",

            "Avoid overcrowding",

            "Clean feeders and drinkers regularly",

            "Improve farm hygiene"

        ],


        "warning": 
        "Early treatment is recommended to reduce spread."

    },



    "Newcastle": {

        "severity": "High",

        "actions": [

            "Immediately isolate suspected birds",

            "Reduce movement between poultry areas",

            "Contact a veterinarian for confirmation",

            "Monitor other birds for symptoms"

        ],

        "prevention": [

            "Maintain vaccination schedules",

            "Control farm visitors",

            "Disinfect equipment regularly",

            "Maintain biosecurity practices"

        ],


        "warning":
        "Newcastle disease can spread quickly among birds."

    },



    "Fowlpox": {

        "severity": "Medium",

        "actions": [

            "Separate affected birds",

            "Check skin lesions regularly",

            "Reduce mosquito exposure",

            "Seek veterinary guidance"

        ],

        "prevention": [

            "Control insects around poultry areas",

            "Maintain clean housing",

            "Follow vaccination programs",

            "Avoid contact with infected birds"

        ],


        "warning":
        "Monitor birds closely to prevent further spread."

    },


    "Healthy": {

        "severity": "Low",

        "actions": [

            "Continue regular monitoring",

            "Maintain current feeding and hygiene practices"

        ],

        "prevention": [

            "Keep water and feeders clean",

            "Maintain good farm management"

        ],


        "warning":
        "No disease risk detected."

    }

}



# =========================================================
# GET RECOMMENDATION
# =========================================================

def get_recommendation(disease_name):

    return DISEASE_RECOMMENDATIONS.get(

        disease_name,

        {

            "severity": "Unknown",

            "actions": [

                "Capture another image",

                "Consult veterinarian"

            ],

            "prevention": [

                "Maintain farm hygiene"

            ],

            "warning":

            "No specific recommendation available."

        }

    )