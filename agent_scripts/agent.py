import openai

client = openai.AzureOpenAI(
        api_version="2024-06-01",
        azure_endpoint="https://genai-nexus.int.api.corpinter.net/apikey/",
        api_key="06c9643f-90c3-4b7a-8560-5b5e5a0d2730",
    )


system_prompt = """
You are an assistant whose task is to compare product attributes found by a vision agent with the seller's description and report any mismatches.

Inputs:
1) Vision agent attributes are provided as a single-line or multi-line string (attribute names vary by product).
   Example tokens: color, pattern, fabric, fit, sleeve, closure, no_of_pockets, Blouse, Blouse Fabric, Net Quantity, etc.
   NOTE: attribute values can be empty strings or values like "No Blouse", "Without Blouse", "None", "N/A" to indicate absence.

2) Description: free-form seller text.

Rules (strict):
- Treat the following (case-insensitive) as **absence indicators** when they appear as attribute values in the vision agent: "no", "none", "without", "no blouse", "no_b blouse", "n/a", "na", "not included", "no blouse piece".
- Treat the description as **indicating presence** if it contains tokens/phrases like "blouse", "blouse piece", "unstitched blouse", "stitched blouse", "comes with blouse", "paired with * blouse", "with matching blouse", "includes blouse", or synonyms. Use substring matching and be case-insensitive; allow minor punctuation/spacing differences.
- If vision agent indicates **absent** (per absence indicators above) but the description mentions a corresponding item (e.g., blouse), that is a **mismatch**.
- If vision agent gives a value (e.g., color=Blue) and the description explicitly states a different value (e.g., "red"), that is a **mismatch**.
- If an attribute exists in only one source (vision OR description) and the other side makes no claim, **ignore** it (do not count as mismatch).
- For numeric/quantity attributes (e.g., "Net Quantity (N) : Single", "no_of_pockets: 2"), parse numeric words if obvious and compare (e.g., "single" -> 1). If parsing is ambiguous, do not declare mismatch.
- The assistant must only return a single JSON object (no extra text, no explanation). Use valid JSON.

Output JSON format (exact keys):
{
  "match": true|false,                     // boolean JSON true/false
  "mismatched_attributes": ["attr1", ...],// list of attribute names that mismatched (empty list if match = true)
  "reason": "detailed reason or 'All attributes match'"
}

When producing the reason, be specific (mention attribute names, vision values and description snippets that caused the mismatch).
"""

user_input = """
vision agent input: Saree Fabric : Georgette, Blouse : Without Blouse, Blouse Fabric : No Blouse, Pattern : Printed, Blouse Pattern : Printed, Net Quantity (N) : Single

description provided by seller: Give in to the exotic confluence of today and tomorrow in this lovely casual wear saree. This sarees is made of georgette fabric which is highlighted with elegant printed and pallu which makes this more demanding among people. This sarees is comfortable to wear and care. Paired with matching Printed unstitched blouse piece. You can wear this saree at casual, semi parties, family functions. Pair this with beautiful jewellery and pair of heels and you will look beautiful.

"""

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt}
    ],
    response_format={"type": "json_object"}

)
out_text = completion.choices[0].message.content

print(out_text)