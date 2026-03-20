/**
 * aiService.ts — Chanoch AI Service Layer
 *
 * All Gemini API calls are centralised here. Each function follows the same
 * three-layer contract:
 *   1. Application-layer intent extraction  (never delegate to the LLM)
 *   2. XML-structured system prompt         (Anthropic-style, priority-ordered rules)
 *   3. Hardened response schema             (required fields, correct types, no nulls
 *                                            where a value is always expected)
 *
 * Model selection rationale is documented inline for every call site.
 */

import { GoogleGenAI, Type, ThinkingLevel } from "@google/genai";
import {
  SaleItem,
  HealthProfile,
  ScannedItem,
  MealPlan,
  GroceryItem,
  Meal,
} from "../types";
import { sanitizePromptInput } from "../utils/security";

// ─── Model constants ──────────────────────────────────────────────────────────
// Centralised so a version bump touches one line, not six call sites.
const MODEL = {
  /** Multi-modal + vision tasks, single-meal generation, cultural recipe fidelity */
  PRO: "gemini-3.1-pro-preview",
  /** Multi-day plan generation, sales search — fast, cost-effective */
  FLASH: "gemini-3-flash-preview",
  /** Lightweight regional availability check — minimal reasoning needed */
  FLASH_LITE: "gemini-3.1-flash-lite-preview",
} as const;

// ─── Client factory ───────────────────────────────────────────────────────────
function getAIClient(): GoogleGenAI {
  const env = (window as any).__ENV__ || {};
  const apiKey = env.GEMINI_API_KEY || (typeof process !== 'undefined' && process.env ? process.env.GEMINI_API_KEY : undefined) || env.API_KEY || (typeof process !== 'undefined' && process.env ? process.env.API_KEY : undefined);

  if (!apiKey) throw new Error("No Gemini API key found in environment.");
  return new GoogleGenAI({ apiKey });
}

// ─── Application-layer intent extraction ─────────────────────────────────────
// These run BEFORE the system prompt is built.  The LLM never sees raw
// free-text and is never asked to parse numbers out of preferences strings.

/** Extracts "3" from "3-day", "3 days", "three days", etc. */
function extractDays(text?: string): number | undefined {
  if (!text) return undefined;
  const wordMap: Record<string, number> = {
    one: 1, two: 2, three: 3, four: 4, five: 5,
    six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  };
  const wordMatch = text.match(/\b(one|two|three|four|five|six|seven|eight|nine|ten)\b[- ]?day/i);
  if (wordMatch) return wordMap[wordMatch[1].toLowerCase()];
  const numMatch = text.match(/(\d+)[- ]?day/i);
  return numMatch ? parseInt(numMatch[1], 10) : undefined;
}

/** Extracts "$80", "80 dollars", "under 80", "budget of 80", etc. */
function extractBudget(text?: string): number | undefined {
  if (!text) return undefined;
  const match = text.match(/(?:\$|budget[^\d]*|under[^\d]*)(\d+(?:\.\d{1,2})?)/i);
  return match ? parseFloat(match[1]) : undefined;
}

/** Extracts "for 2 people", "2 people", "2 persons", "family of 4", etc. */
function extractPeople(text?: string): number | undefined {
  if (!text) return undefined;
  const match = text.match(/(?:for\s+)?(\d+)\s*(?:people|persons?|adults?)|family\s+of\s+(\d+)/i);
  if (match) return parseInt(match[1] ?? match[2], 10);
  return undefined;
}

// ─── Robust JSON parser ───────────────────────────────────────────────────────
function robustJsonParse<T>(jsonText: string, fallback: T): T {
  if (!jsonText) return fallback;
  const cleanedText = jsonText.replace(/^```json\s*/i, "").replace(/```$/, "").trim();
  if (!cleanedText) return fallback;

  // Pass 0: fast-path
  try { return JSON.parse(cleanedText) as T; } catch (_) { /* continue */ }

  console.warn("[aiService] JSON fast-path failed — attempting salvage");

  // Pass 1: escape unescaped control characters inside strings
  let cleaned = "";
  let inString = false;
  let escaped = false;
  const stack: string[] = [];

  for (const char of cleanedText) {
    if (escaped) { cleaned += char; escaped = false; continue; }
    if (char === "\\") { cleaned += char; escaped = true; continue; }
    if (char === '"') { inString = !inString; cleaned += char; continue; }

    if (inString) {
      if      (char === "\n") cleaned += "\\n";
      else if (char === "\r") cleaned += "\\r";
      else if (char === "\t") cleaned += "\\t";
      else                    cleaned += char;
    } else {
      cleaned += char;
      if      (char === "{" || char === "[") stack.push(char);
      else if (char === "}" && stack.at(-1) === "{") stack.pop();
      else if (char === "]" && stack.at(-1) === "[") stack.pop();
    }
  }

  try { return JSON.parse(cleaned) as T; } catch (_) { /* continue */ }

  // Pass 2: close truncated structures
  let fixed = cleaned;
  if (inString) fixed += '"';
  for (let i = stack.length - 1; i >= 0; i--)
    fixed += stack[i] === "{" ? "}" : "]";

  try { return JSON.parse(fixed) as T; } catch (_) { /* continue */ }

  // Pass 3: for array fallbacks, salvage the last complete object
  if (Array.isArray(fallback)) {
    let text = cleaned;
    while (text.length > 0) {
      const last = text.lastIndexOf("}");
      if (last === -1) break;
      text = text.slice(0, last + 1);
      try {
        const wrapped = text.trim().startsWith("[") ? text : `[${text}]`;
        const parsed = JSON.parse(wrapped.trim().endsWith("]") ? wrapped : wrapped + "]");
        if (Array.isArray(parsed)) {
          console.warn("[aiService] Salvaged JSON by truncating to last complete object.");
          return parsed as unknown as T;
        }
      } catch (_) { text = text.slice(0, last); }
    }
  }

  console.error("[aiService] All JSON salvage passes failed — returning fallback.");
  return fallback;
}

// ─── Shared sub-schemas ───────────────────────────────────────────────────────

const macrosSchema = {
  type: Type.OBJECT,
  description: "Estimated macronutrient breakdown per serving",
  properties: {
    calories: { type: Type.NUMBER, description: "Total calories (kcal)" },
    protein:  { type: Type.NUMBER, description: "Protein in grams" },
    carbs:    { type: Type.NUMBER, description: "Total carbohydrates in grams" },
    fat:      { type: Type.NUMBER, description: "Total fat in grams" },
    fiber:    { type: Type.NUMBER, description: "Dietary fibre in grams" },
    sugar:    { type: Type.NUMBER, description: "Total sugars in grams" },
  },
  required: ["calories", "protein", "carbs", "fat"],
} as const;

const mealSchema = {
  type: Type.OBJECT,
  description: "A single meal",
  properties: {
    name:             { type: Type.STRING, description: "Short meal name, e.g. 'Grilled Salmon Bowl'" },
    description:      { type: Type.STRING, description: "One-sentence description, max 20 words" },
    cuisine:          { type: Type.STRING, description: "Cuisine origin, e.g. 'Mediterranean'" },
    prepTimeMinutes:  { type: Type.NUMBER, description: "Preparation time in minutes" },
    cookTimeMinutes:  { type: Type.NUMBER, description: "Cook time in minutes" },
    tags: {
      type: Type.ARRAY,
      description: "Descriptive tags e.g. ['high-protein','gluten-free']",
      items: { type: Type.STRING },
    },
    ingredients: {
      type: Type.ARRAY,
      description: "Ingredient list with quantities, e.g. '200g chicken breast'",
      items: { type: Type.STRING },
    },
    usesGroceries: {
      type: Type.ARRAY,
      description: "Names of grocery items from the user's list used in this meal",
      items: { type: Type.STRING },
    },
    macros: macrosSchema,
  },
  required: ["name", "description", "ingredients", "macros"],
} as const;

// ─── analyzeGroceryItem ───────────────────────────────────────────────────────
/**
 * Vision-based grocery item analysis.
 *
 * Model: PRO — multi-modal input, health-profile cross-referencing, and
 * alternative recommendation require richer reasoning than Flash-Lite offers.
 */
export async function analyzeGroceryItem(
  imageBase64: string,
  profile: HealthProfile,
): Promise<ScannedItem | null> {
  try {
    const profileContext = `
      Diet Types            : ${profile.dietTypes?.join(", ")          || "None"}
      Allergies             : ${profile.allergies?.join(", ")           || "None"}
      Health Goals          : ${profile.goals?.join(", ")               || "None"}
      Disliked Ingredients  : ${profile.dislikedIngredients?.join(", ") || "None"}
    `.trim();

    const systemInstruction = `
<role>
You are Chanoch, an expert nutritionist and computer vision assistant.
Analyse the provided grocery item image and return a structured health assessment.
</role>

<rules>
RULE 1 — SCOPE ENFORCEMENT (highest priority)
Only process images of food, beverages, or grocery products.
If the image does not contain a grocery item, return:
{ "name": "Unknown", "category": "Unknown", "nutritionalInfo": { "calories": "0", "protein": "0g", "carbs": "0g", "fat": "0g" }, "isAligned": false, "reason": "Image does not contain a recognisable grocery item.", "healthierAlternative": null }

RULE 2 — ALLERGY SAFETY (non-negotiable)
If the identified item contains ANY ingredient that matches the user's listed allergies,
you MUST set isAligned to false and explain the conflict in the reason field.

RULE 3 — ALTERNATIVE RECOMMENDATION
Only populate healthierAlternative if:
  a) the item conflicts with the health profile, OR
  b) a meaningfully healthier product in the same category exists.
Do not invent brand names. If unsure, omit the field.

RULE 4 — NUTRITIONAL VALUES
All nutritional values are per standard serving size (as printed on the packaging if visible).
Values must be realistic — do not hallucinate macros for unknown items.
</rules>

<context>
User Health Profile:
${profileContext}
</context>

<self_check>
Before responding, verify:
[ ] isAligned correctly reflects allergy and diet constraints
[ ] All nutritional values are per serving, not per 100g unless that IS the serving
[ ] healthierAlternative is a real product, not a hallucination
[ ] Response is raw JSON matching the schema — no markdown or preamble
</self_check>
    `.trim();

    const response = await getAIClient().models.generateContent({
      model: MODEL.PRO,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: imageBase64.split(",")[1] ?? imageBase64,
            },
          },
          { text: "Analyse this grocery item based on my health profile." },
        ],
      },
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            name:     { type: Type.STRING, description: "Identified item name including brand if visible" },
            category: { type: Type.STRING, description: "Category, e.g. 'Dairy', 'Produce', 'Snacks'" },
            servingSize: { type: Type.STRING, description: "Serving size as printed on packaging, e.g. '30g (1 cup)'" },
            nutritionalInfo: {
              type: Type.OBJECT,
              description: "Macros per serving — all values as numeric strings with units",
              properties: {
                calories: { type: Type.STRING, description: "e.g. '150 kcal'" },
                protein:  { type: Type.STRING, description: "e.g. '8g'" },
                carbs:    { type: Type.STRING, description: "e.g. '22g'" },
                fat:      { type: Type.STRING, description: "e.g. '5g'" },
                fiber:    { type: Type.STRING, description: "e.g. '3g'" },
                sugar:    { type: Type.STRING, description: "e.g. '10g'" },
              },
              required: ["calories", "protein", "carbs", "fat"],
            },
            isAligned:          { type: Type.BOOLEAN, description: "True if item fits the user's health profile" },
            reason:             { type: Type.STRING,  description: "Clear explanation of alignment or conflict" },
            healthierAlternative: { type: Type.STRING, description: "A specific healthier product if applicable; omit if not relevant" },
            allergenWarnings: {
              type: Type.ARRAY,
              description: "Any detected allergens that conflict with the user's profile",
              items: { type: Type.STRING },
            },
          },
          required: ["name", "category", "nutritionalInfo", "isAligned", "reason"],
        },
      },
    });

    return robustJsonParse<ScannedItem | null>(response.text ?? "{}", null);
  } catch (error) {
    console.error("[aiService] analyzeGroceryItem error:", error);
    return null;
  }
}

// ─── generateMealPlan ─────────────────────────────────────────────────────────
/**
 * Generates a full multi-day meal plan.
 *
 * Model: FLASH — structured JSON generation with a well-defined schema.
 * No vision input, no multi-hop web search, no cultural recipe fidelity needed.
 * FLASH handles 7-day plans comfortably within the 8192-token output budget.
 *
 * Intent extraction happens HERE, in application code, before the prompt is built.
 * The LLM is NEVER asked to parse numbers from free text.
 */
export async function generateMealPlan(
  groceries: GroceryItem[],
  profile: HealthProfile,
  days?: number,
  people?: number,
  preferences?: string,
  budget?: number,
): Promise<MealPlan | null> {
  try {
    // ── Resolve parameters before touching the prompt ──────────────────────
    const sanitizedPrefs  = preferences ? sanitizePromptInput(preferences) : undefined;
    const resolvedDays    = days   ?? extractDays(sanitizedPrefs)   ?? 1;
    const resolvedBudget  = budget ?? extractBudget(sanitizedPrefs) ?? null;
    const resolvedPeople  = people ?? extractPeople(sanitizedPrefs) ?? 1;

    const today      = new Date();
    const currentDay = today.toLocaleDateString("en-US", { weekday: "long" });

    const safeGroceries = groceries ?? [];
    const groceryList   = safeGroceries.length > 0
      ? safeGroceries.map(g => `- ${g.quantity}x ${g.name}`).join("\n")
      : "None";

    const profileContext = `
      Diet Types  : ${profile.dietTypes?.join(", ") || "None"}
      Allergies   : ${profile.allergies?.join(", ")  || "None"}
      Health Goals: ${profile.goals?.join(", ")      || "None"}
    `.trim();

    const systemInstruction = `
<role>
You are Chanoch, an expert meal planner and clinical nutritionist.
Generate a structured, personalised meal plan grounded in the user's health profile,
available groceries, and any explicit preferences they have stated.
</role>

<context>
Today is ${currentDay}. Start the plan from ${currentDay} unless the user's preferences
explicitly name a different starting day.

Resolved Meal Plan Parameters (authoritative — do NOT re-parse from preferences):
  Days    : ${resolvedDays}
  People  : ${resolvedPeople}
  Budget  : ${resolvedBudget !== null ? `$${resolvedBudget} USD (strict)` : "No constraint"}

User Health Profile:
${profileContext}

Available Groceries:
${groceryList}

${sanitizedPrefs ? `User Raw Preferences:\n"${sanitizedPrefs}"` : ""}
</context>

<rules>
RULE 1 — SCOPE ENFORCEMENT (highest priority)
Only respond to requests about food, meals, and nutrition.
If the request is entirely unrelated to these topics, return:
{ "days": [], "error": "Request is outside the meal planning scope." }

RULE 2 — DAY COUNT (non-negotiable)
Generate EXACTLY ${resolvedDays} day(s). The days array MUST have exactly ${resolvedDays} entries.
Generating the wrong number of days is a critical failure.

RULE 3 — ALLERGY AND DIET SAFETY (non-negotiable)
NEVER include ingredients that conflict with:
  Allergies  : ${profile.allergies?.join(", ")  || "none"}
  Diet Types : ${profile.dietTypes?.join(", ") || "none"}
If a preference conflicts with a safety rule, add a warning to the warnings array
and substitute a compliant alternative. Do NOT silently honour a dangerous preference.

RULE 4 — SNACK POLICY
Do NOT include a snack slot UNLESS the user explicitly requests it in their preferences.
The key must be entirely absent from days without a snack request — not null, not empty.

RULE 5 — GROCERY UTILISATION
Prioritise groceries from the list above. Assume standard pantry staples are available:
oil, salt, pepper, garlic, onion, basic spices.
If the grocery list is empty, generate the plan normally; the user will shop later.

RULE 6 — BUDGET ENFORCEMENT
${resolvedBudget !== null
  ? `Strict budget: $${resolvedBudget} across the full plan.
  - Populate estimatedCost with the total ingredient cost.
  - If estimatedCost > ${resolvedBudget}, set budgetExceeded: true and populate
    budgetWarning with specific cheaper substitutions.
  - If within budget, set budgetExceeded: false, budgetWarning: null.`
  : "No budget constraint. Omit estimatedCost, budgetExceeded, and budgetWarning."}

RULE 7 — SPECIFIC DISH REQUESTS
If preferences name a specific dish for a specific day or meal slot, place that
exact dish in the correct slot. This overrides nutritional optimisation.
Never silently ignore an explicit dish request.

RULE 8 — RESPONSE BREVITY
Meal descriptions: max 20 words. Ingredient lists: max 8 items per meal.
Do not include cooking instructions unless explicitly asked.
</rules>

<self_check>
Before emitting, verify:
[ ] days array has EXACTLY ${resolvedDays} entries
[ ] No meal contains an ingredient from the allergy list
[ ] Diet type constraints are respected throughout
[ ] Snack slot is present ONLY if requested
[ ] Response is raw JSON with no markdown or preamble
${resolvedBudget !== null ? "[ ] estimatedCost and budgetExceeded are both populated" : ""}
</self_check>
    `.trim();

    const response = await getAIClient().models.generateContent({
      model: MODEL.FLASH,
      contents: "Generate the meal plan.",
      config: {
        systemInstruction,
        maxOutputTokens: 8192,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            startDay:       { type: Type.STRING,  description: "Day the plan starts, e.g. 'Monday'" },
            totalDays:      { type: Type.NUMBER,  description: `MUST equal ${resolvedDays}` },
            servings:       { type: Type.NUMBER,  description: `MUST equal ${resolvedPeople}` },
            estimatedCost:  { type: Type.NUMBER,  description: "Total estimated ingredient cost in USD" },
            budgetExceeded: { type: Type.BOOLEAN, description: "True if estimatedCost exceeds the budget" },
            budgetWarning:  { type: Type.STRING,  description: "Explanation + cheaper alternatives if over budget" },
            warnings: {
              type: Type.ARRAY,
              description: "Non-fatal issues: allergy substitutions, preference conflicts",
              items: { type: Type.STRING },
            },
            error: {
              type: Type.STRING,
              description: "Populated only if the request is out of scope or unresolvable",
            },
            days: {
              type: Type.ARRAY,
              description: `One entry per day — MUST have exactly ${resolvedDays} entries`,
              items: {
                type: Type.OBJECT,
                properties: {
                  day:  { type: Type.STRING, description: "Day of week, e.g. 'Monday'" },
                  date: { type: Type.STRING, description: "ISO date, e.g. '2026-03-16'" },
                  meals: {
                    type: Type.OBJECT,
                    properties: {
                      breakfast: mealSchema,
                      lunch:     mealSchema,
                      dinner:    mealSchema,
                      snack: {
                        ...mealSchema,
                        description: "Only present if user explicitly requested a snack",
                      },
                    },
                    required: ["breakfast", "lunch", "dinner"],
                  },
                  dailyTotals: {
                    type: Type.OBJECT,
                    description: "Summed macros across all meals for this day",
                    properties: {
                      calories: { type: Type.NUMBER },
                      protein:  { type: Type.NUMBER },
                      carbs:    { type: Type.NUMBER },
                      fat:      { type: Type.NUMBER },
                    },
                    required: ["calories", "protein", "carbs", "fat"],
                  },
                },
                required: ["day", "date", "meals", "dailyTotals"],
              },
            },
            planSummary: {
              type: Type.OBJECT,
              description: "Aggregate stats across the entire plan for dashboard KPIs",
              properties: {
                avgDailyCalories: { type: Type.NUMBER },
                avgDailyProtein:  { type: Type.NUMBER },
                avgDailyCarbs:    { type: Type.NUMBER },
                avgDailyFat:      { type: Type.NUMBER },
                cuisinesUsed: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: "Unique cuisine types across all meals",
                },
                groceriesUsed: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: "Grocery items from the user's list that were used",
                },
              },
              required: ["avgDailyCalories", "avgDailyProtein", "avgDailyCarbs", "avgDailyFat"],
            },
          },
          required: ["startDay", "totalDays", "servings", "days", "planSummary"],
        },
      },
    });

    const result = robustJsonParse<MealPlan | null>(response.text ?? "{}", null);

    // Post-parse guard: enforce day count even if the model short-changed us
    if (result && Array.isArray(result.days) && result.days.length !== resolvedDays) {
      console.warn(
        `[aiService] generateMealPlan: expected ${resolvedDays} days, got ${result.days.length}. ` +
        "Returning partial result — caller should re-try or notify the user.",
      );
    }

    return result;
  } catch (error) {
    console.error("[aiService] generateMealPlan error:", error);
    return null;
  }
}

// ─── generateSingleMeal ───────────────────────────────────────────────────────
/**
 * Generates exactly ONE meal for a specific slot and day.
 *
 * Use this when:
 *   • The user requests a named dish for a specific day/slot
 *     ("Ghanaian fufu and groundnut soup for Tuesday lunch")
 *   • The user wants an AI-suggested meal for one slot without regenerating
 *     the whole plan
 *
 * Do NOT use this for multi-day plan generation — use generateMealPlan.
 *
 * Model: PRO — cultural recipe fidelity and authentic ingredient lists require
 * the stronger reasoning model. Flash hallucinates ingredients for less common
 * dishes (e.g. it omits fermented locust beans from egusi soup).
 */
export async function generateSingleMeal(
  mealName: string,
  mealType: "breakfast" | "lunch" | "dinner" | "snack",
  dayLabel: string,
  groceries: GroceryItem[],
  profile: HealthProfile,
  additionalNotes?: string,
  budget?: number,
): Promise<Meal | null> {
  try {
    const sanitizedName  = sanitizePromptInput(mealName,  100);
    const sanitizedNotes = additionalNotes ? sanitizePromptInput(additionalNotes, 300) : undefined;

    const safeGroceries = groceries ?? [];
    const groceryList   = safeGroceries.length > 0
      ? safeGroceries.map(g => `- ${g.quantity}x ${g.name}`).join("\n")
      : "None";

    const profileContext = `
      Diet Types           : ${profile.dietTypes?.join(", ")          || "None"}
      Allergies            : ${profile.allergies?.join(", ")           || "None"}
      Health Goals         : ${profile.goals?.join(", ")               || "None"}
      Disliked Ingredients : ${profile.dislikedIngredients?.join(", ") || "None"}
    `.trim();

    const systemInstruction = `
<role>
You are Chanoch, an expert chef and nutritionist specialising in international cuisines.
Your task is to generate a single, complete, authentic meal recipe.
</role>

<context>
Target Meal   : "${sanitizedName}"
Meal Type     : ${mealType}
Day           : ${dayLabel}
${sanitizedNotes ? `User Notes    : ${sanitizedNotes}` : ""}
${budget       ? `Budget        : $${budget} USD (strict)` : ""}

User Health Profile:
${profileContext}

Available Groceries (use where possible; assume oil, salt, pepper, garlic, spices are available):
${groceryList}
</context>

<rules>
RULE 1 — SCOPE ENFORCEMENT (highest priority)
Only generate food/meal content. If the meal name is entirely unrelated to food,
return: { "name": "${sanitizedName}", "description": "Not a food item.", "ingredients": [], "macros": { "calories": 0, "protein": 0, "carbs": 0, "fat": 0 } }

RULE 2 — EXACT DISH FIDELITY (non-negotiable)
The meal MUST be exactly "${sanitizedName}".
Do NOT rename it, simplify it, or substitute it with something easier or more common.
If it is a specific cultural dish (Ghanaian fufu, Japanese ramen, Nigerian jollof),
generate the AUTHENTIC version with authentic ingredients, technique, and name.

RULE 3 — ALLERGY AND DIET SAFETY (non-negotiable)
NEVER include ingredients that conflict with:
  Allergies  : ${profile.allergies?.join(", ")  || "none"}
  Diet Types : ${profile.dietTypes?.join(", ") || "none"}
If the requested dish inherently requires an allergen, add a prepNotes warning
and suggest a safe authentic substitution. Never silently include a dangerous ingredient.

RULE 4 — INGREDIENT QUANTITIES
EVERY ingredient MUST include a quantity with units.
e.g. "500g chicken thighs", "2 cups cassava flour", "3 tbsp peanut butter".
Bare ingredient names without amounts are not acceptable.

RULE 5 — BUDGET ENFORCEMENT
${budget
  ? `Budget: $${budget}. Estimate ingredient cost. If over budget, explain in prepNotes
    and suggest cheaper alternatives. Populate budgetWarning.`
  : "No budget constraint."}

RULE 6 — MACRO ACCURACY
Macros must be realistic and calculated per serving, not per 100g.
Do not copy generic database values — adjust for the specified quantities.
</rules>

<self_check>
Before emitting, verify:
[ ] Meal name matches "${sanitizedName}" exactly
[ ] Every ingredient has a quantity with units
[ ] No allergens from the profile are present
[ ] Macros are per serving, not per 100g
[ ] Response is raw JSON with no markdown or preamble
</self_check>
    `.trim();

    const response = await getAIClient().models.generateContent({
      model: MODEL.PRO,
      contents: `Generate a complete recipe for: ${sanitizedName}`,
      config: {
        systemInstruction,
        // LOW thinking: single well-defined recipe, schema is strict.
        // Reduces latency and token usage without sacrificing output quality.
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        maxOutputTokens: 8192,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            name: {
              type: Type.STRING,
              description: "Exact meal name as requested — do NOT rename or shorten",
            },
            description: {
              type: Type.STRING,
              description: "1–2 vivid sentences including cultural origin",
            },
            cuisine: {
              type: Type.STRING,
              description: "Cuisine origin, e.g. 'Ghanaian', 'Japanese', 'Italian'",
            },
            prepTimeMinutes: { type: Type.NUMBER, description: "Prep time in minutes" },
            cookTimeMinutes: { type: Type.NUMBER, description: "Cook time in minutes" },
            servings:        { type: Type.NUMBER, description: "Number of servings this recipe yields" },
            tags: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "e.g. ['High Protein', 'West African', 'Gluten-Free']",
            },
            ingredients: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Every ingredient with quantity + unit. e.g. '500g chicken thighs'",
            },
            usesGroceries: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Grocery items from the user's list used in this recipe",
            },
            recipe: {
              type: Type.STRING,
              description: "Numbered step-by-step instructions, detailed enough to cook the dish",
            },
            prepNotes: {
              type: Type.STRING,
              description: "Cultural context, make-ahead tips, substitution notes, or serving suggestions",
            },
            budgetWarning: {
              type: Type.STRING,
              description: "Populated only if estimated cost exceeds the budget",
            },
            macros: macrosSchema,
          },
          required: ["name", "description", "cuisine", "ingredients", "recipe", "macros"],
        },
      },
    });

    return robustJsonParse<Meal | null>(response.text ?? "{}", null);
  } catch (error) {
    console.error("[aiService] generateSingleMeal error:", error);
    return null;
  }
}

// ─── searchSales ──────────────────────────────────────────────────────────────
/**
 * Searches grocery flyers and store websites for current prices.
 *
 * Model: FLASH with MEDIUM thinking — multi-step reasoning chain:
 *   1. Query flyer aggregators (Flipp, Reebee)
 *   2. Query official store websites for regular prices
 *   3. Geolocate the closest branch to the user's coordinates
 *   4. De-duplicate and rank by price
 * LOW thinking drops steps 3–4, producing hallucinated addresses and estimated
 * prices. MEDIUM provides the necessary depth without excessive latency.
 */
export async function searchSales(
  query: string,
  store?: string,
  category?: string,
  lat?: number,
  lng?: number,
  accuracy?: number,
  postalCode?: string,
): Promise<SaleItem[]> {
  const sanitizedQuery      = sanitizePromptInput(query,    200);
  const sanitizedStore      = store      ? sanitizePromptInput(store,      50) : undefined;
  const sanitizedCategory   = category   ? sanitizePromptInput(category,   50) : undefined;
  const sanitizedPostalCode = postalCode ? sanitizePromptInput(postalCode, 10) : undefined;

  const systemInstruction = `
<role>
You are Chanoch, a comprehensive global grocery price finder.
Your goal is to surface ALL available prices for the requested items — both current
promotional prices (from digital flyers) and regular shelf prices (from store websites).
</role>

<rules>
RULE 1 — SCOPE ENFORCEMENT (highest priority)
Only process queries about groceries, food, beverages, household consumables, or store locations.
If the query is entirely unrelated, return an empty array: []

RULE 2 — NO HALLUCINATION (non-negotiable)
NEVER invent or estimate prices. Only return prices you found in a current flyer or on
an official store website. If you cannot verify a price, omit the item entirely.

RULE 3 — STORE ADDRESS ACCURACY (non-negotiable)
You MUST provide the EXACT address of the CLOSEST branch to the user's location.
Use Google Search to verify the actual address. DO NOT use generic city-centre addresses
if a closer branch exists. DO NOT fabricate addresses.
Generate mapsUri as: https://www.google.com/maps/search/?api=1&query=<store+address>
Never trust AI-generated Google Maps place IDs — they are frequently hallucinated.

RULE 4 — PRODUCT SIZE REQUIREMENT
ALWAYS include size, weight, or volume in the name field.
Correct  : "Kirkland Organic Whole Milk (2L)"
Incorrect: "Kirkland Organic Whole Milk"

RULE 5 — MULTI-STORE COMPARISON
${sanitizedStore
  ? `Return items ONLY from: ${sanitizedStore}. Do NOT include other stores.`
  : `Return items from MULTIPLE different store chains so the user can compare prices
     (e.g. local equivalents of Walmart, Aldi, Tesco, Loblaws, Woolworths).`}

RULE 6 — CHEAPEST PRICE WINS
If you find multiple prices for the same item, return ONLY the cheapest one.

RULE 7 — COMPREHENSIVENESS
Return ALL relevant items found. Do not artificially limit the result count.
</rules>
  `.trim();

  // Build user prompt
  const parts: string[] = [];

  if (sanitizedQuery) {
    parts.push(
      `Find all available prices (both flyer sales and regular prices) for: ${sanitizedQuery}. ` +
      `For items not on sale, search official store websites for the verified regular price. ` +
      `DO NOT estimate. Work hard to find real prices.`,
    );
  } else if (sanitizedStore) {
    parts.push(`Find ALL items currently on sale in the weekly flyer for ${sanitizedStore}.`);
  } else if (sanitizedCategory) {
    parts.push(`Find ALL items currently on sale in the category: ${sanitizedCategory}.`);
  } else {
    parts.push("Find ALL items currently on sale in weekly flyers across the user's local area.");
  }

  if (sanitizedCategory && sanitizedQuery) parts.push(`Category filter: ${sanitizedCategory}.`);
  if (sanitizedPostalCode) parts.push(`User postal/zip code: ${sanitizedPostalCode}.`);
  if (lat && lng) {
    parts.push(
      `User coordinates: (${lat}, ${lng})${accuracy ? ` ±${accuracy}m accuracy` : ""}. ` +
      `Use Google Search to find the absolute closest physical store branch. ` +
      `Verify the address exists — do not return a city-centre default.`,
    );
  }

  const userPrompt = parts.join(" ");

  try {
    const response = await getAIClient().models.generateContent({
      model: MODEL.FLASH,
      contents: userPrompt,
      config: {
        systemInstruction,
        tools: [{ googleSearch: {} }],
        thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM },
        responseMimeType: "application/json",
        maxOutputTokens: 8192,
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              name:          { type: Type.STRING,  description: "Product name including size/weight/volume" },
              price:         { type: Type.STRING,  description: "Current price, e.g. '$2.99' or '2 for $5.00'" },
              originalPrice: { type: Type.STRING,  description: "Regular shelf price if available" },
              isOnSale:      { type: Type.BOOLEAN, description: "True if promotional price, false if regular" },
              validFrom:     { type: Type.STRING,  description: "Sale start date, e.g. 'Mar 16'" },
              validUntil:    { type: Type.STRING,  description: "Sale end date, e.g. 'Mar 22'" },
              store:         { type: Type.STRING,  description: "Store chain name" },
              category:      { type: Type.STRING,  description: "Product category" },
              address:       { type: Type.STRING,  description: "Verified street address of closest branch" },
              distance:      { type: Type.STRING,  description: "Estimated distance, e.g. '1.2 km'" },
              unit:          { type: Type.STRING,  description: "Unit of measurement: lb, kg, pieces, pack, etc." },
              description:   { type: Type.STRING,  description: "Brief deal details, max 100 characters" },
            },
            required: ["name", "price", "store", "category", "address", "isOnSale"],
          },
        },
      },
    });

    const items = robustJsonParse<any[]>(response.text ?? "[]", []);

    return items.map((item: any, index: number) => {
      // Always build the Maps URL client-side — AI-generated place IDs are unreliable
      const q = encodeURIComponent(`${item.store ?? ""} ${item.address ?? ""}`.trim());
      return {
        ...item,
        mapsUri: `https://www.google.com/maps/search/?api=1&query=${q}`,
        id: `item-${Date.now()}-${index}`,
      };
    });
  } catch (error) {
    console.error("[aiService] searchSales error:", error);
    return [];
  }
}

// ─── filterStoresByLocation ───────────────────────────────────────────────────
/**
 * Filters a store list down to chains that operate near the user's location.
 *
 * Model: FLASH_LITE — single-step binary classification (does chain X operate
 * in region Y?).  No price verification, no multi-hop reasoning.
 * Results are cached in localStorage to eliminate redundant API calls.
 */
export async function filterStoresByLocation(
  stores: { name: string; logo: string }[],
  lat?: number,
  lng?: number,
  postalCode?: string,
): Promise<{ name: string; logo: string }[]> {
  if (!lat && !lng && !postalCode) return stores;

  // Cache keyed on rounded coordinates to tolerate minor GPS drift
  const cacheKey = `chanoch_storeFilter_${lat?.toFixed(2)},${lng?.toFixed(2)},${postalCode ?? ""}`;
  try {
    const cached = localStorage.getItem(cacheKey);
    if (cached) return JSON.parse(cached);
  } catch (_) { /* ignore localStorage errors */ }

  const systemInstruction = `
<role>
You are a helpful assistant that determines which grocery store chains have physical
locations in or near a specified area.
</role>

<rules>
RULE 1 — REGIONAL ACCURACY
Return ONLY stores from the provided list that actually operate in or near the user's area.
Do NOT include Canadian chains for a US location, or US chains for a Canadian location.
Cross-border exceptions apply only if a chain genuinely operates in both countries near the border.

RULE 2 — FALLBACK
If location data is ambiguous or you cannot determine regional availability for a store,
include it in the results (fail open to avoid hiding valid stores).
</rules>
  `.trim();

  const locationClause = lat && lng
    ? `User coordinates: (${lat}, ${lng})`
    : `User postal/zip code: ${postalCode}`;

  const userPrompt =
    `Store list: ${stores.map(s => s.name).join(", ")}\n` +
    `${locationClause}\n` +
    `Return a JSON array of store names from the list that operate near this location.`;

  try {
    const response = await getAIClient().models.generateContent({
      model: MODEL.FLASH_LITE,
      contents: userPrompt,
      config: {
        systemInstruction,
        tools: [{ googleSearch: {} }],
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
        },
      },
    });

    const validNames = robustJsonParse<string[]>(response.text ?? "[]", []);

    if (!validNames || validNames.length === 0) return stores; // fail open

    const validLower  = new Set(validNames.map(n => n.toLowerCase()));
    const filtered    = stores.filter(s => validLower.has(s.name.toLowerCase()));
    const result      = filtered.length > 0 ? filtered : stores; // guard against over-filtering

    try { localStorage.setItem(cacheKey, JSON.stringify(result)); } catch (_) { /* ignore */ }

    return result;
  } catch (error) {
    console.error("[aiService] filterStoresByLocation error:", error);
    return stores; // fail open on error
  }
}
