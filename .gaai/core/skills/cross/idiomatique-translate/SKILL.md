---
type: skill
id: idiomatique-translate
name: idiomatique-translate
description: Translate strings idiomatically across multiple target languages using a project glossary, preserving tone, domain terminology, and format placeholders while flagging untranslatable content.
layer: cross
category: content
created_at: 2026-03-02
updated_at: 2026-03-02
---

# Skill: Idiomatique Translate (Multi-Language)

## Purpose

Translate copy from any source language to any target language with idiomatic fluency, cultural awareness, domain expertise, and consistent voice — language-agnostic pattern optimized for B2B SaaS platforms.

> **Usage context:** Bulk translation of UI copy, help text, error messages, and onboarding content. Automatically detects source language. Replaces mechanical translation with culturally-aware, tonally-consistent output in the target language.

---

## Input

```json
{
  "source_string": "You pay $0.50 per direct link request",
  "source_language": null, // null = auto-detect, or explicit: "en", "fr", "de", "es"
  "target_language": "fr", // required: "en", "fr", "de", "es", "it", "pt", "nl", etc.
  "context": {
    "domain": "billing", // domain expertise context
    "component": "PricingCard",
    "tone": "transparent", // transparent, benevolent, precise, casual, formal, technical
    "audience": "expert", // expert, prospect, admin, user
    "key": "billing.directLink.perRequest"
  },
  "glossary": {
    "direct link": "lien direct", // source term : target term (or multi-lang)
    "request": "demande",
    "lead": "prospect",
    "milestone": "jalon"
    // Can also use language-keyed format:
    // "direct link": { "fr": "lien direct", "de": "direkter Link", "es": "enlace directo" }
  },
  "regional_preferences": {
    "currency": "EUR", // override default currency ($→€, etc.)
    "date_format": "DD/MM/YYYY",
    "number_format": "de_DE" // locale code for number formatting
  },
  "notes": "Avoid formal 'vous' / use informal 'tu', clarify that €0.50 is machine cost not per-lead"
}
```

## Output

```json
{
  "translated_string": "Tu paies 0,50 € par demande traitée",
  "source_language_detected": "en",
  "target_language": "fr",
  "confidence": "high", // high, medium, low
  "idiomaticity_score": 4.5, // 1-5 scale (5 = native speaker fluency)
  "notes": "Changed 'requête' to 'demande' (more natural in billing context). Used '€' instead of '$' for FR audience. Used informal 'tu' per context. Tone: factual, reassuring.",
  "alternatives": [
    {
      "text": "Tu paies 0,50 € par traitement de demande",
      "note": "More formal variant"
    },
    {
      "text": "Chaque demande te coûte 0,50 €",
      "note": "More casual variant"
    }
  ],
  "cultural_notes": "French audience expects € currency and informal 'tu' in UX context. 'Demande' is more natural than 'requête' in billing."
}
```

---

## Translation Principles (Universal)

### 1. **Idiomatic > Literal**

# <<<<<<< HEAD

> > > > > > > origin/staging

- ❌ "You pay $0.50 per request" → literal French: "Vous payez 0,50 $ par requête"
- ✅ "You pay $0.50 per request" → idiomatic French: "Vous payez 0,50 € par demande"

(Reasoning: FR audience expects €, not $. "Demande" is more natural than "requête" in billing context.)

**For any language pair:** Prioritize cultural norms, expected terminology, and natural phrasing over word-for-word mapping.

### 2. **Domain-Aware Glossary Alignment**

Glossary can be:
