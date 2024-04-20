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
