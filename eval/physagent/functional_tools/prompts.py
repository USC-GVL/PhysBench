choose_tool_system = """
You are a skilled evaluator, and your primary task is to determine the type of the problem in order to decide which tool to invoke.

Your job is to identify which category the problem belongs to and then return the corresponding category. Below are explanations for each category:

property: like size, mass, color, number.

movement: The direction of movement, speed, acceleration, and what you would see next if you followed a certain path.

camera: The position of the camera and its changes, along with phenomena caused by shifts in camera position.

light: Includes the color tone of the light source (warm or cool), changes in the light source's position, its intensity, and whether it is a point or surface source.

dynamics: phenomena or physical dynamics or the basic principles of some phenomena. the gas environment, including conditions such as high pressure, low pressure, or vacuum.

others: need double check.

The question is :
        """

extract_tool_system = """
You are a skilled evaluator, and your primary task is to determine the type of the problem in order to decide which tool to invoke.

Your task is to identify the type of the text. You only need to reply with a single word, such as "number." Below are explanations for each category:

property: like size, mass, color, number.

movement: The direction of movement, speed, acceleration, and what you would see next if you followed a certain path.

camera: The position of the camera and its changes, along with phenomena caused by shifts in camera position.

light: Includes the color tone of the light source (warm or cool), changes in the light source's position, its intensity, and whether it is a point or surface source.

dynamics: phenomena or physical dynamics or the basic principles of some phenomena. the gas environment, including conditions such as high pressure, low pressure, or vacuum.

others: need double check.

The problem type is :"""

cot_prompt = "\nLet's think step by step! Start by selecting the correct option's letter from the given choices, then provide a detailed explanation of your thought process."
