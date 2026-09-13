---
title: literature-review-crop-phenology-ndvi
type: note
permalink: timeseries-final-crop-phenology/literature-review-crop-phenology-ndvi
---

# Literature Review: Satellite NDVI Time-Series for Crop Phenology Detection — Research Landscape, Gaps, and Near-SOTA

> Scope note: this document is the **pre-data-collection literature review** step only. It does not decide a dataset, sensor, region, crop, or model — those decisions are explicitly deferred until the team confirms data availability. Builds on `nghien-cuu-nen-tang-crop-phenology-ndvi.md` (background/foundations, in Vietnamese); this document goes deeper on the standard research pipeline, a world-scale comparative assessment, near-SOTA works, and an explicit gap analysis, following the `deep-research` skill's Investigation + Analysis (gap-analysis) methodology.

## 1. The standard research pipeline in this field

Across essentially every paper surveyed — from the classic reviews (Zeng, Wardlow, Xiang, Hu, & Li, 2020, *Remote Sensing of Environment*, 237) to the most recent 2024–2025 syntheses — vegetation/crop phenology retrieval from satellite time series follows the same four-stage pipeline:

| Stage | What happens | Typical techniques / choices |
|---|---|---|
| A. Data acquisition | Choose sensor(s) trading off spatial vs. temporal resolution | MODIS (250 m, daily/8–16-day composite), Landsat (30 m, 16-day), Sentinel-2 (10 m, 5-day), PlanetScope (3 m, near-daily), Sentinel-1 SAR (cloud-independent) |
| B. Preprocessing & reconstruction | Remove cloud/noise contamination, fill gaps, smooth | Cloud masking, HANTS/Fourier gap-filling, Savitzky–Golay, Whittaker, double-logistic/asymmetric-Gaussian fitting |
| C. Phenological metric extraction | Turn a clean curve into SOS/POS/EOS dates | Threshold-based, inflection/derivative-based (curve-fitting), breakpoint/decomposition-based (BFAST-style), ML/DL (LSTM, ConvLSTM, foundation-model embeddings) |
| D. Validation & application | Check dates against ground truth, then use them | Field survey, PhenoCam/near-surface imagery, official crop-calendar statistics; downstream: yield forecasting, crop-type mapping, early-warning, climate-adaptation studies |

This is essentially the same shape as the course's own workflow (preprocess → decompose/smooth → model → evaluate), which is useful framing for the literature-review chapter itself.

Sources: Zeng et al. (2020); [Monitoring and Prediction of Land Surface Phenology Using Satellite Earth Observations — A Brief Review](https://www.mdpi.com/2076-3417/14/24/12020) (2024); [Satellite remote sensing of vegetation phenology: Progress, challenges, and opportunities](https://www.researchgate.net/publication/384606053_Satellite_remote_sensing_of_vegetation_phenology_Progress_challenges_and_opportunities) (2024).

---

## 2. World-scale comparative assessment — where the method actually works, and where it doesn't

### 2.1 Data-rich temperate regions (North America, Europe, China): mature and mostly working
- Dense Sentinel-2/PlanetScope coverage plus PhenoCam near-surface camera networks make direct satellite-vs-ground validation routine. PlanetScope-derived green-up dates match PhenoCam-observed emergence at RMSE ≈ 8.35 days (bias 5.28, R²=0.64); EOS/maturity at RMSE ≈ 9.92 days ([Near-Surface and High-Resolution Satellite Time Series for Detecting Crop Phenology](https://doi.org/10.3390/rs14091957), 2022).
- But even here, accuracy is **crop-type dependent, not uniform**: Sentinel-1/Sentinel-2 SOS agreement ranges from r = 0.89 (MAE = 10 days) for some crops down to r = 0.15 (MAE = 53 days) for others ([Comparing land surface phenology of major European crops from Sentinel-1/-2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7841528/), 2021) — i.e., "solved" is an overstatement even for the best-resourced case.

### 2.2 Cloud-dominated monsoon/tropical rice systems (e.g., Mekong Delta, Southeast Asia): partially solved with region-specific workarounds
- Persistent cloud cover degrades optical (NDVI) time series continuity during the growing season.
- The field's answer has been **crop-specific rule-based algorithms**, not general-purpose ones: PhenoRice (Boschetti et al., 2017) exploits the flooding signal unique to paddy rice (via NDFI+EVI) rather than a generic phenology curve model; SAR (Sentinel-1, cloud-independent) is increasingly used as a complement/substitute.
- Reported accuracy is still reasonable in absolute terms (<12% error vs. official statistics in the Mekong Delta case), but the *method* is rice-specific and does not transfer to other crops in the same cloudy environment without redesign.

### 2.3 Smallholder / Global South agriculture (Sub-Saharan Africa, much of smallholder South/Southeast Asia): the least solved case
- Fragmented, small fields create a **mixed-pixel problem** that persists even at 10–30 m resolution, not just at MODIS's 250 m ([Extraction of multiple cropping information at sub-pixel scale, Henan](https://www.tandfonline.com/doi/full/10.1080/10106049.2022.2104390), 2022).
- Ground-truth and ground-observation infrastructure (PhenoCam-type networks, systematic field surveys) is sparse compared to North America/Europe/China, so validation numbers from §2.1 may not generalize to these settings ([GROW-Africa gaps](https://www.nature.com/articles/s41597-025-05257-5), 2025; [EO-NAM Africa framework](https://www.nature.com/articles/s44264-025-00083-z), 2025).
- The dominant global operational product, MODIS MCD12Q2, has documented **spatial gaps concentrated exactly in rainfall-driven (rather than temperature-driven) phenology regions** — which describes most tropical smallholder cropland ([Remote Sensing of Land Surface Phenology: Progress, Challenges, Prospects](https://www.researchgate.net/publication/387850741_Remote_Sensing_of_Land_Surface_Phenology_Progress_Challenges_Prospects), 2024/2025).

**Net picture:** the methodological toolkit is mature in data-rich temperate regions but its validated accuracy drops — and the supporting ground-truth infrastructure nearly disappears — precisely where the world's food-security-critical smallholder and monsoon-rice agriculture is concentrated. This asymmetry is the single clearest "good vs. bad" pattern across the literature.

---

## 3. Near-SOTA (2023–2026)

- **Remote-sensing foundation models (RSFMs):** Prithvi (IBM/NASA, pretrained on >1 TB Harmonized Landsat-Sentinel-2 via masked-autoencoder with a 3D time-aware embedding), SatMAE (ViT-based, Landsat-8), Presto (pixel-time-series pretraining across Sentinel-2/Sentinel-1/DEM/Dynamic World), and AgriFM (2025, multi-source temporal RSFM for agriculture mapping). Reported effect: tasks that needed ~50,000 labeled tiles in 2022 can now be fine-tuned with 500–5,000 ([Genealogy of Foundation Models in Remote Sensing](https://arxiv.org/pdf/2504.17177), 2025; [AgriFM](https://arxiv.org/pdf/2505.21357), 2025).
  - **Important caveat:** these models are demonstrated mainly for crop-**type mapping**/general EO tasks. Rigorous benchmarking of FM embeddings specifically for **phenology-date extraction accuracy** (SOS/POS/EOS vs. ground truth) is still sparse — this is a gap, not a solved problem (see §4.5).
- **Super-resolution reconstruction for phenology extraction** (2025): targets the spatial-resolution/mixed-pixel bottleneck directly by upsampling coarse time series before metric extraction ([Frontiers in Plant Science](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1687246/full), 2025).
- **LSTM/ConvLSTM-based forecasting-then-extraction** (2022–2025): still validated mainly on single-crop, single-region studies; cross-region generalization is largely untested.
- **Multi-sensor fusion (Sentinel-1 SAR + Sentinel-2 optical):** actively researched to overcome cloud gaps, but effectiveness is crop-type dependent (§2.1), not a universal fix.

---

## 4. Explicit research gaps

1. **Definition mismatch between "satellite land-surface phenology" and "agronomic phenology."** Spectral inflection points (SOS/POS/EOS) are a statistical construct of the vegetation-index curve, not the same thing as farmer-observed/BBCH-scale growth stages — repeatedly flagged as a validation confound (Zeng et al., 2020; [Progress, Challenges, Prospects](https://www.researchgate.net/publication/387850741_Remote_Sensing_of_Land_Surface_Phenology_Progress_Challenges_Prospects), 2024/2025).
2. **Unresolved scale/mixed-pixel problem for smallholder and multi-cropping systems**, persisting even at 10 m resolution, not just coarse MODIS pixels (§2.3).
3. **Geographically uneven validation infrastructure.** PhenoCam/near-surface networks and dense field surveys are concentrated in North America/Europe/China; reported accuracy figures (RMSE ~8–10 days) may not hold in tropical smallholder regions where such ground truth barely exists.
4. **No standardized benchmark/protocol for algorithm intercomparison.** Reviews explicitly note the field lacks a common dataset/protocol to fairly compare threshold vs. curve-fitting vs. breakpoint vs. ML/DL methods head-to-head across crops and regions — echoing the broader crop-phenomics standardization gap ([Standard Framework Construction of Technology and Equipment for Big Data in Crop Phenomics](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2024.06.001), 2024).
5. **Foundation models are under-validated specifically for phenology-date extraction.** Current FM literature targets crop-type classification; rigorous SOS/POS/EOS accuracy benchmarking against ground truth is sparse — a clear opening for smaller-scale, well-scoped work.
6. **Cloud-dominated monsoon regions still rely on crop-specific rule-based workarounds** (e.g., PhenoRice's flood signal for paddy rice) rather than general-purpose robust methods, limiting transfer to other crops in the same climate.
7. **Cross-sensor fusion benefit is crop-type dependent, with no unified theory yet** for when Sentinel-1+Sentinel-2 fusion helps vs. hurts (§2.1).

---

## 5. Implication for the course project (orientation only — no data/model decision yet)

- Gaps #2, #3, and #6 all point toward the same underserved area: **validation/ground-truth scarcity and cloud-driven data gaps in tropical smallholder/monsoon rice systems** — which includes Vietnam. A project oriented there sits on a literature-acknowledged real gap rather than re-solving an already well-validated US-Corn-Belt-style problem.
- This is an orientation for how the literature-review chapter's "gap statement" could be framed — **not** a data-source or modeling decision. Sensor choice, region, crop, and dataset remain open until the team confirms data availability, per your instruction.

---

## 6. Annotated bibliography (new sources from this pass)

| # | Source | Contribution | Evidence tier |
|---|---|---|---|
| 1 | Zeng, L., Wardlow, B. D., Xiang, D., Hu, S., & Li, D. (2020). A review of vegetation phenological metrics extraction using time-series, multispectral satellite data. *Remote Sensing of Environment, 237*. | Master review: methods, error sources, opportunities/challenges | Peer-reviewed review, highly cited |
| 2 | Satellite remote sensing of vegetation phenology: Progress, challenges, and opportunities (2024). *ISPRS J. Photogrammetry & Remote Sensing* (via ScienceDirect/ResearchGate). | Recent (2024) synthesis of global progress + challenges | Peer-reviewed review |
| 3 | Remote Sensing of Land Surface Phenology: Progress, Challenges, Prospects (2024/2025, Springer book chapter). | Terminology/scale-effect critique, MCD12Q2 spatial-gap finding | Peer-reviewed |
| 4 | Monitoring and Prediction of Land Surface Phenology Using Satellite Earth Observations — A Brief Review (2024). *Applied Sciences* (MDPI). | Recent brief review confirming pipeline stages | Peer-reviewed |
| 5 | Near-Surface and High-Resolution Satellite Time Series for Detecting Crop Phenology (2022). *Remote Sensing* (MDPI). | PlanetScope–PhenoCam validation numbers | Peer-reviewed |
| 6 | Comparing land surface phenology of major European crops from Sentinel-1/-2 (2021). PMC7841528. | Crop-type-dependent sensor fusion accuracy (r 0.15–0.89) | Peer-reviewed |
| 7 | Extraction of multiple cropping information at sub-pixel scale, Henan (2022). *Geocarto International*. | Mixed-pixel problem in multi-cropping systems | Peer-reviewed |
| 8 | A Genealogy of Foundation Models in Remote Sensing (2025). arXiv:2504.17177. | Survey of Prithvi/SatMAE/Presto and effect on labeled-data needs | Preprint (arXiv), not yet peer-reviewed — verify before formal citation |
| 9 | AgriFM: A Multi-source Temporal Remote Sensing Foundation Model for Agriculture Mapping (2025). arXiv:2505.21357. | Recent agriculture-specific foundation model | Preprint (arXiv) |
| 10 | Research on the application of remote sensing image super-resolution reconstruction techniques in crop phenology extraction (2025). *Frontiers in Plant Science*. | Near-SOTA attack on mixed-pixel/resolution gap | Peer-reviewed |
| 11 | GROW-Africa database gaps (2025). *Scientific Data* (Nature). | Quantifies ground-truth/data scarcity in Africa | Peer-reviewed |
| 12 | Standard Framework Construction of Technology and Equipment for Big Data in Crop Phenomics (2024). *Engineering*. | Confirms lack of standardization/benchmark protocol | Peer-reviewed |

(Carries forward, not repeated: the 12 sources in `nghien-cuu-nen-tang-crop-phenology-ndvi.md` — TIMESAT, Zhang 2003, BFAST/Hyndman, PhenoRice, HANTS, Savitzky–Golay, Whittaker, LSTM phenology paper, Mekong Delta sowing-date study.)

---

## 7. Limitations of this literature scan & AI disclosure

- This is a rapid, WebSearch-driven literature scan (skill `deep-research`, `lit-review`-depth Investigation + Analysis stages, applied manually — the `academic-research-skills` plugin itself is not installed in this session). It has not gone through the full source-verification / devil's-advocate checkpoint of the skill's `full` pipeline mode.
- Several sources were read only via search snippets/abstracts, not full text (notably #8, #9 which are unreviewed arXiv preprints — flag before citing formally). ScienceDirect full-text pages returned HTTP 403 and could not be fetched directly; conclusions for those items rely on search-result summaries of the abstract.
- No independent fact-check or cross-model verification pass was run on these claims (that is a distinct, heavier step in the skill's methodology, not performed here).
- Produced with the assistance of Claude (Anthropic) using WebSearch, session date 2026-09-12.