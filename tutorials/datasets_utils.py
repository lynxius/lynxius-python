def your_llm_application(query: str) -> str:
    """
    This is a very rudimentary implementation of an LLM :)
    For the sake of brevity of this demo, we simply store the LLM answers.
    This function is meant to abstract the functionality of **your** LLM applications
    that Lynxius never has access to.
    """

    data = {
        "What are the members of the famous rock band Queen?": """
The members of the iconic rock band Queen are:

1. Freddie Mercury (lead vocals, piano)
2. Brian May (guitar, vocals)
3. Roger Taylor (drums, vocals)
4. John Deacon (bass guitar)
These four musicians formed the core of Queen and contributed to the band's immense success with their distinctive talents and styles.
""",
        "What is the difference between virtual reality and augmented reality? Can you provide examples of both?": """
Virtual Reality (VR) fully immerses users in a simulated environment, while Augmented Reality (AR) overlays digital content onto the real world.

Examples:

VR: Oculus Rift for gaming.
AR: Pokémon GO for mobile gaming.
""",
        "What planets are part of the solar system?": """
The eight planets that are part of our solar system, in order of their distance from the Sun, are:

Mercury
Venus
Earth
Mars
Jupiter
Saturn
Uranus
Neptune
Additionally, there is a category of smaller celestial bodies called "dwarf planets," which includes Pluto, Eris, Haumea, Makemake, and Ceres. These objects are not considered full-fledged planets but are still part of the solar system.
""",
        #         "Name 3 highest mountains in Switzerland.": """
        # 3 highest mountains in Switzerland:
        # 1. Mount Everest
        # 2. Mont Blanc
        # 3. Mount Kilimanjaro
        # """,
        "Name 3 highest mountains in Switzerland.": """
3 highest mountains in Switzerland:

1. Dufourspitze (4,634 meters)
2. Dom (4,545 meters)
3. Weisshorn (4,506 meters)
""",
    }

    if query in data:
        return data[query]
    else:
        return "I don't know"


def chatdoctor_v1(query: str) -> str:
    """
    This is a very rudimentary implementation of an LLM Chat Doctor :)
    For the sake of brevity of this demo, we simply store the LLM answers.
    This function is meant to abstract the functionality of **your** LLM applications
    that Lynxius never has access to.
    """

    data = {
        "How can I prevent the flu?": "The annual flu vaccine is best way to prevent the flu.",
        "What are the early signs of diabetes?": "Diabetes can be identified by increased thirst, irregular urination, weight loss, blurred vision, tireness, and slow-healing sores or infections.",
        "How do I know if I have a food allergy?": "Pay attention to indicators such as itching or eczema, swelling of the lips, or throat, nausea or vomiting, diarrhea, or difficulty breathing shortly after consuming certain foods.",
        "What should I do if I get a sunburn?": "If you get a sunburn, you should cool the skin with a cold compress or cool bath, apply moisturizer or aloe vera gel to soothe the skin, drink plenty of water to stay hydrated, and avoid further sun exposure until the burn heals.",
        "What are the symptoms of a migraine headache?": "Migraines are characterized by a gradual increase in pain over several days.",
        "What are the symptoms of the common cold?": "Symptoms of the common cold include a high fever and severe headache.",
        "What are the symptoms of a urinary tract infection (UTI)?": "UTIs are mostly asymptomatic and do not cause any noticeable symptoms.",
    }

    if query in data:
        return data[query]
    else:
        return "I don't know"


def chatdoctor_v2(query: str) -> str:
    """
    This is a very rudimentary implementation of an LLM Chat Doctor :)
    For the sake of brevity of this demo, we simply store the LLM answers.
    This function is meant to abstract the functionality of **your** LLM applications
    that Lynxius never has access to.
    """

    data = {
        "How can I prevent the flu?": "The annual flu vaccine is best way to prevent the flu.",
        "What are the early signs of diabetes?": "Diabetes can be identified by increased thirst, irregular urination, weight loss, blurred vision, tireness, and slow-healing sores or infections.",
        "How do I know if I have a food allergy?": "Pay attention to indicators such as itching or eczema, swelling of the lips, or throat, nausea or vomiting, diarrhea, or difficulty breathing shortly after consuming certain foods.",
        "What should I do if I get a sunburn?": "If you get a sunburn, you should cool the skin with a cold compress or cool bath, apply moisturizer or aloe vera gel to soothe the skin, drink plenty of water to stay hydrated, and avoid further sun exposure until the burn heals.",
        "What are the symptoms of a migraine headache?": "Symptoms of a migraine headache can include intense throbbing or pulsing pain, usually on one side of the head, sensitivity to light, sound, and sometimes smell, nausea, and vomiting. Some people also experience visual disturbances called aura before the headache begins.",
        "What are the symptoms of the common cold?": "Common cold symptoms are runny or stuffy nose, sneezing, sore throat, cough, mild body aches, and mild fatigue.",
        "What are the symptoms of a urinary tract infection (UTI)?": "Symptoms of a UTI may include a strong, persistent urge to urinate, a burning sensation when urinating, passing frequent, small amounts of urine, cloudy or strong-smelling urine, and pelvic pain in women.",
    }

    if query in data:
        return data[query]
    else:
        return "I don't know"
