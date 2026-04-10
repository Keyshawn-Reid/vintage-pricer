Building a pricer modeling tool that using ebay api to reference old sold items to determine the value of new items
 
 right now we are in the process of :
 1)getting the api account request accepted by e bay
 2) using sythetic data building a working model, preparing for the api



2/9/2026

🚀 Live at: https://vintage-pricer.onrender.com

## Product Vision

### Current State (v1.0)
- Harley Davidson vintage tee pricing
- Manual entry via dropdowns
- Photo analysis via GPT-4o vision
- XGBoost ML model trained on 114 real eBay sold listings
- Live at: https://vintage-pricer.onrender.com

### Roadmap

#### v1.1 — Unified Photo + Form Flow
Replace two-tab layout with a single unified flow:
- Upload photos (tag, front/back graphic, care label)
- AI auto-fills signals from photos
- User reviews and adjusts before pricing
- Photo informs the form, form is source of truth

#### v1.2 — Brand Adaptive Signals
Each brand has its own premium signal set:
- Harley Davidson: 3D Emblem, single stitch, location name, era, event
- Ed Hardy: rhinestones, collab, designer series
- Hysteric Glamour: graphic type, Japanese market tag, era
- UI adapts dynamically based on brand selected

#### v1.3 — Multi-Brand Model
- Separate XGBoost model per brand
- Each trained on brand-specific eBay sold data
- Shared pipeline, brand-specific features

#### v1.4 — Generics & Hype Pricing
- Separate pricing track for non-brand items
- Signals: graphic style, color, condition, fabric, trend
- Store reputation premium factored in
- Collab detection (e.g. Ken Carson x Ed Hardy)

#### v1.5 — UX Polish
- Hover tooltips explaining each signal
- How to identify: single stitch, 3D emblem, etc.
- Confidence score on price output
- Mobile optimized for in-store use

### Notes
- Generics pricing is the hardest problem — vibes-based, needs own research
- Collab items (e.g. Ed Hardy x Ken Carson $400) are hype-priced, not vintage ML
- Multi-brand system needs one data collection session per brand