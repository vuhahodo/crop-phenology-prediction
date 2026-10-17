"""Generates a pixel-perfect Springer LNCS PDF manuscript matching the instructor's template (splnproc2510.docm) exactly.
- Page geometry: Text frame 122 mm x 193 mm, 44 mm margins left/right, 52 mm top/bottom on A4 (210 mm x 297 mm).
- Running headers stamped via PyMuPDF:
  * Page 1: NO header, NO footer (clean).
  * Even pages (2, 4, ...): Left-aligned page number, Centered author name ('Ha Do-Phuc-Vu').
  * Odd pages (3, 5, ...): Centered short title ('Crop Phenology Detection from Sentinel-2 NDVI Time Series'), Right-aligned page number.
  * Footers: Empty on all pages.
- Visual elements: Exact 14pt title, 10pt author, 9pt affiliation, 8.5pt Courier email, 9pt indented abstract,
  Times New Roman body, academic prose (no bullet-list slide formatting), booktabs tables, captions, equations.
"""
import base64
from pathlib import Path
import fitz  # PyMuPDF
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "results" / "plots"
DOCS_DIR = BASE_DIR / "docs"

def image_to_base64(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>A Temporal Segmentation Benchmark and Calendar-Conditioned Attention for Sentinel-2 NDVI Phenology Extraction on BreizhCrops</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {
    output: 'html',
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$', right: '$', display: false}
    ]
  });"></script>
<style>
  .katex-mathml {
    display: none !important;
  }
  /* Exact Springer LNCS geometry: Text frame 122mm x 193mm on A4 */
  @page {
    size: 210mm 297mm;
    margin-top: 52mm;
    margin-bottom: 52mm;
    margin-left: 44mm;
    margin-right: 44mm;
  }

  body {
    font-family: "Times New Roman", Times, "Nimbus Roman No9 L", serif;
    font-size: 10pt;
    line-height: 12.2pt;
    color: #000000;
    text-align: justify;
    margin: 0;
    padding: 0;
    background: #ffffff;
    -webkit-font-smoothing: antialiased;
  }

  /* Page 1 Title block: starts ~53mm from top edge, generous whitespace */
  .title-block {
    padding-top: 4pt;
    margin-bottom: 24pt;
    text-align: center;
  }

  .title {
    font-size: 14pt;
    font-weight: bold;
    line-height: 17pt;
    margin-bottom: 20pt;
    text-align: center;
  }

  .authors {
    font-size: 10pt;
    line-height: 13pt;
    margin-bottom: 12pt;
    text-align: center;
  }

  .authors sup {
    font-size: 8pt;
  }

  .affiliation {
    font-size: 9pt;
    line-height: 12pt;
    color: #111111;
    margin-bottom: 22pt;
    text-align: center;
  }

  .email {
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
    color: #000000;
  }

  /* Abstract: indented by 10mm (~28.3pt) on both left and right (width = 102mm) */
  .abstract-container {
    margin-left: 28.3pt;
    margin-right: 28.3pt;
    margin-bottom: 22pt;
    font-size: 9pt;
    line-height: 11.5pt;
    text-align: justify;
  }

  .abstract-container strong.abstract-title {
    font-weight: bold;
  }

  .keywords-block {
    margin-top: 10pt;
    font-size: 9pt;
    line-height: 11.5pt;
  }

  .keywords-block strong {
    font-weight: bold;
  }

  /* Section Headings */
  h1 {
    font-size: 12pt;
    font-weight: bold;
    line-height: 14pt;
    margin-top: 11pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
  }

  h2 {
    font-size: 10pt;
    font-weight: bold;
    line-height: 13pt;
    margin-top: 10pt;
    margin-bottom: 3pt;
    page-break-after: avoid;
  }

  h3 {
    font-size: 10pt;
    font-weight: bold;
    display: inline;
    margin-right: 4pt;
  }

  .sec-num {
    display: inline-block;
    min-width: 14pt;
  }

  .subsec-num {
    display: inline-block;
    min-width: 22pt;
  }

  /* Paragraph indents: 0.4cm (~11.3pt) */
  p {
    margin-top: 0;
    margin-bottom: 0;
    text-indent: 11.5pt;
  }

  /* First paragraph of section/subsection/table/figure has NO indent */
  h1 + p, h2 + p, .no-indent, .figure + p, .table-box + p, .equation + p {
    text-indent: 0 !important;
  }

  /* Figures */
  .figure {
    margin: 7pt auto 4pt auto;
    text-align: center;
    page-break-inside: avoid;
  }

  .figure img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto 3pt auto;
  }

  .fig-caption {
    font-size: 9pt;
    line-height: 11.5pt;
    text-align: justify;
    margin: 2.5pt 0 4.5pt 0;
  }

  .fig-caption strong {
    font-weight: bold;
  }

  /* Tables (Classic Springer Booktabs) */
  .table-box {
    margin: 7pt auto 4pt auto;
    text-align: center;
    page-break-inside: avoid;
  }

  .table-caption {
    font-size: 9pt;
    text-align: justify;
    margin-bottom: 3.5pt;
    line-height: 11.5pt;
  }

  .table-caption strong {
    font-weight: bold;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    line-height: 11pt;
    margin: 0 auto;
    border-top: 1.2pt solid #000;
    border-bottom: 1.2pt solid #000;
  }

  th {
    font-weight: bold;
    border-bottom: 0.6pt solid #000;
    padding: 2.5pt 3pt;
    text-align: center;
  }

  td {
    padding: 2.0pt 3pt;
    text-align: center;
    border: none;
  }

  td.left, th.left {
    text-align: left;
    padding-left: 4pt;
  }

  /* Equations: centered formula with right-aligned numbering */
  .equation {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 3pt 0;
    padding: 0;
    page-break-inside: avoid;
  }

  .eq-content {
    flex-grow: 1;
    text-align: center;
  }

  .eq-num {
    width: 30pt;
    text-align: right;
    font-size: 10pt;
  }

  /* References: Unnumbered H1, numbered 1., 2. with hanging indent */
  .ref-list {
    margin-top: 3pt;
  }

  .ref-item {
    font-size: 8.5pt;
    line-height: 10.5pt;
    text-align: justify;
    padding-left: 16pt;
    text-indent: -16pt;
    margin-bottom: 1.2pt;
  }

  .disclosure {
    font-size: 8.5pt;
    line-height: 10.5pt;
    margin-top: 5pt;
    margin-bottom: 3pt;
  }
</style>
</head>
<body>

<!-- HEADER & TITLE -->
<div class="title-block">
  <div class="title">
    A Temporal Segmentation Benchmark and Calendar-Conditioned Attention for Sentinel-2 NDVI Phenology Extraction on BreizhCrops
  </div>

  <div class="authors">
    Ha Do-Phuc-Vu<sup>[0009-0004-8291-3563]</sup>
  </div>

  <div class="affiliation">
    Faculty of Computer Science, Vietnam - Korea University of Information and Communication Technology (VKU), The University of Danang, Da Nang, Vietnam<br>
    <span class="email">hadpv.23ai@vku.udn.vn</span>
  </div>
</div>

<!-- ABSTRACT -->
<div class="abstract-container">
  <p class="no-indent"><strong class="abstract-title">Abstract.</strong> Accurate extraction of crop phenological transition dates&mdash;specifically Start of Season (SOS / green-up), Peak of Season (POS), and End of Season (EOS / senescence)&mdash;from satellite vegetation index time series is vital for automated agricultural monitoring. However, conventional operational pipelines rely on heuristic smoothing and ad-hoc peak-finding rules, while the deep learning literature lacks standardized benchmarks evaluating sequential architectures against identical criteria. In this study, we formulate phenology extraction as a dense, point-wise 4-class temporal segmentation task (Fallow, Vegetative, Reproductive, Senescence) from which calendar dates (DOY) are deterministically decoded. We evaluate six architectures spanning classical tabular machine learning, recurrent, convolutional, and self-attention networks on 2,655 quality-filtered Sentinel-2 NDVI annual cycles (<em>T</em> = 73 regularized 5-day composite steps) from the BreizhCrops benchmark (frh01 region, 2017). Deep sequential models substantially outperform feature-engineered tabular baselines (Random Forest Macro-F1: 0.420; Bi-LSTM: 0.778; Transformer: 0.904 &plusmn; 0.009), driven primarily by their access to full-season bidirectional temporal context. Furthermore, we examine <strong>CBA-PhenoNet</strong> (Calendar-Biometric Attention), which augments a temporal Transformer with learned cyclical Day-of-Year positional encoding and a calendar-conditioned gating prior (+322 parameters). Across five random initialization seeds, CBA-PhenoNet yields segmentation performance comparable to the standard Transformer (0.897 &plusmn; 0.016 vs. 0.904 &plusmn; 0.009 Macro-F1), while demonstrating exploratory reductions in EOS timing error (MAE 7.0 &plusmn; 1.0 vs. 8.5 &plusmn; 0.7 days, achieving lower error in 4 of 5 seeds with 1 tie at 8.6 days; unadjusted paired <em>t</em>-test <em>p</em> = 0.036; Bonferroni-adjusted <em>p</em><sub>adj</sub> &asymp; 0.108 across three milestones). We disclose that reported milestone errors measure fidelity in reproducing rule-based amplitude threshold targets rather than ground-truth in-situ agronomic observations.</p>
  
  <div class="keywords-block">
    <strong>Keywords:</strong> Crop Phenology, Sentinel-2, NDVI Time Series, Temporal Segmentation, Attention Gating, BreizhCrops.
  </div>
</div>

<!-- SECTION 1 -->
<h1><span class="sec-num">1</span> Introduction</h1>

<p>Monitoring vegetation phenology using satellite Earth observation is a fundamental pillar of modern precision agriculture, global crop calendar assessment, and climate change resilience. Time series of Normalized Difference Vegetation Index (NDVI) derived from high-revisit multi-spectral constellations, such as Copernicus Sentinel-2, capture the dynamic spectral trajectory of crop growth from emergence to maturity. Accurately extracting key phenological transition dates&mdash;specifically the Start of Season (SOS / green-up), Peak of Season (POS / maximum canopy), and End of Season (EOS / canopy senescence)&mdash;enables timely agronomic interventions, parcel-level crop tracking [15], and accurate harvest yield estimations.</p>

<p>Despite decades of progress, conventional extraction approaches face two critical methodological bottlenecks. First, dominant operational pipelines remain heavily reliant on rule-based amplitude thresholds or semi-empirical curve-fitting tools such as TIMESAT [1], employing double-logistic functions [2, 7] and harmonic analysis [14]. These heuristics assume idealized bell-shaped trajectories and struggle to generalize across variable climates and crop types without exhaustive parameter tuning. Second, the Earth observation literature lacks a standardized benchmark protocol comparing classical machine learning algorithms (e.g., Random Forest [10], XGBoost [11]) against modern deep sequential architectures (1D-CNN, Bi-LSTM [12], and temporal Transformers [5, 6]) under identical data splits and evaluation criteria.</p>

<p>To systematically address these challenges, this study investigates two central research questions. First, does framing phenology extraction as a dense, point-wise temporal segmentation task allow bidirectional deep sequential models to capture full-season developmental trajectories and systematically outperform tabular machine learning built on local lag features (RQ1)? Second, to what extent do calendar-conditioned attention gating mechanisms improve phenological milestone accuracy relative to their computational training and inference footprints (RQ2)?</p>

<p>In resolving these questions, this paper makes four principal contributions: (1) <em>A Unified Open Benchmark Protocol:</em> We construct an open benchmark protocol on 2,655 quality-filtered Sentinel-2 crop seasons across four annual crop classes in BreizhCrops (frh01 region, 2017), establishing leak-free parcel-level partitions and a dual evaluation suite encompassing both segmentation metrics (Macro-F1) and calendar regression metrics (MAE/RMSE in days). (2) <em>Dense Point-Wise Segmentation Framing:</em> We formulate phenology extraction as a 4-class sequence labeling task (<em>T</em> = 73 regularized steps) from which calendar DOY dates are deterministically decoded, bypassing manual peak-searching heuristics and leveraging bidirectional temporal context. (3) <em>Investigation of Calendar-Conditioned Attention:</em> We investigate CBA-PhenoNet (Calendar-Biometric Attention), coupling cyclical continuous Day-of-Year positional encoding with a learned calendar-conditioned Phenology Gate that modulates attention outputs during key transition windows (+322 parameters). (4) <em>Multi-Seed Variance and Rigorous Reporting:</em> We conduct multi-seed evaluations across five random seeds for attention architectures, demonstrating that overall segmentation quality remains statistically comparable between the standard Transformer and CBA-PhenoNet, while CBA-PhenoNet achieves an exploratory advantage in senescence date extraction (EOS MAE 7.0 &plusmn; 1.0 days vs. 8.5 &plusmn; 0.7 days, lower error in 4 of 5 seeds with 1 tie at 8.6 days; unadjusted paired <em>t</em>-test <em>p</em> = 0.036, <em>p</em><sub>adj</sub> &asymp; 0.108) at 0.27 ms inference latency per season.</p>

<!-- SECTION 2 -->
<h1><span class="sec-num">2</span> Related Work</h1>

<p>Traditional phenological metric extraction predominantly utilizes semi-empirical smoothing and mathematical curve fitting. Software tools such as TIMESAT apply Savitzky-Golay filtering [13] or asymmetric Gaussian functions to reconstruct noise-contaminated vegetation index trajectories [1]. Phenological transition dates are subsequently identified using relative dynamic thresholds (e.g., 20% or 50% seasonal amplitude) or rate-of-change curvature extrema [2]. While computationally lightweight, these methods assume uniform symmetric growth patterns and degrade substantially under persistent cloud cover or atypical double-cropping cycles [3].</p>

<p>In recent years, deep sequence architectures have demonstrated remarkable capability in modeling sequential satellite observations. Temporal Convolutional Neural Networks (1D-CNN) and Recurrent Neural Networks (LSTM [12], Bi-LSTM [12]) have been widely adopted for crop classification and time-series modeling [4]. More recently, self-attention architectures have emerged as state-of-the-art paradigms for satellite image time series (SITS). Pixel-set and temporal self-attention models such as L-TAE [5] and TSViT [6] demonstrate that encoding real calendar Day-of-Year rather than ordinal sequence indices preserves vital temporal spacing across non-uniform revisit cycles. In parallel, domain-specific temporal priors and parametric models [7] demonstrate that constraining representations with seasonal calendars mitigates out-of-season spectral noise. Building on these advances, this work benchmarks these architectural paradigms under a unified temporal segmentation framework.</p>

<!-- SECTION 3 -->
<h1><span class="sec-num">3</span> Proposed Methodology</h1>

<h2><span class="subsec-num">3.1</span> Problem Formulation: Point-Wise Segmentation to Phenology DOY</h2>

<p>Let an observed crop season be represented as a multivariate temporal sequence $\mathbf{X} \in \mathbb{R}^{T \times C}$, where $T=73$ denotes sequence length (corresponding to a full 365-day annual cycle on a regular 5-day grid) and $C=3$ represents feature channels. Channel 0 contains the Savitzky-Golay smoothed NDVI trajectory, while channels 1 and 2 supply continuous cyclical Day-of-Year (DOY) encodings. Each timestep $t \in \{1, \dots, T\}$ is assigned an agronomic stage label $y_t \in \{0, 1, 2, 3\}$, corresponding to four non-overlapping developmental phases: $0$ &mdash; <strong>Fallow</strong> (bare soil or dormant inter-season period, $t < \text{SOS}$ or $t > \text{EOS}$); $1$ &mdash; <strong>Vegetative</strong> (emergence and green-up, $\text{SOS} \le t < \text{POS} - \delta$); $2$ &mdash; <strong>Reproductive</strong> (anthesis and peak biomass surrounding POS, $\text{POS} - \delta \le t \le \text{POS} + \delta$, where $\delta = \lfloor 0.12 \cdot (\text{EOS} - \text{SOS}) \rfloor$); and $3$ &mdash; <strong>Senescence</strong> (ripening and canopy drying, $\text{POS} + \delta < t \le \text{EOS}$).</p>

<p>From predicted sequence $\hat{\mathbf{y}} = (\hat{y}_1, \dots, \hat{y}_T)$, calendar phenology dates are deterministically decoded: $\text{SOS}_{\text{pred}}$ is the timestamp of the first transition from $0 \to 1$; $\text{POS}_{\text{pred}}$ is the timestamp maximizing NDVI within predicted stages $\{1, 2\}$; and $\text{EOS}_{\text{pred}}$ is the timestamp of the first transition from $3 \to 0$. If a predicted sequence does not exhibit a valid transition for a given milestone, that instance is excluded from milestone MAE/RMSE calculations and accounted for in the transition detection recall rate.</p>

<h2><span class="subsec-num">3.2</span> BreizhCrops Dataset and Preprocessing</h2>

<p>We utilize the public BreizhCrops benchmark [8], comprising bottom-of-atmosphere (Level-2A) Sentinel-2 time series acquired over Brittany, France (frh01 region, year 2017). From an initial balanced candidate cohort of 3,328 annual crop seasons across four crop categories (barley, wheat, rapeseed, and corn), quality filtering (removing flat NDVI trajectories with seasonal amplitude below 0.15 and seasons with edge-truncated peaks) retains 2,655 valid annual growth cycles. Each sample spans an annual cycle with $T=73$ interpolated 5-day composite steps (covering DOY 1 to 361). The dataset is partitioned into training (70%, 1,857 seasons), validation (15%, 399 seasons), and test sets (15%, 399 seasons) strictly at the field parcel level to prevent spatial autocorrelation data leakage across identical agricultural plots. Table 1 summarizes the dataset profile across the initial and retained subsets.</p>

<p><strong>Exploratory Data Analysis and Class Imbalance.</strong> An analysis of raw acquisition histories reveals an average of $78.5 \pm 25.5$ valid observations per parcel annually (with a $1\sigma$ interval of 53.0&ndash;104.0 and an observed range of 15&ndash;128). Due to pervasive cloud cover over Brittany, temporal revisits exhibit irregular sampling gaps, necessitating 5-day linear interpolation and Savitzky-Golay filtering (window length 9, polynomial order 2) across all seasons. Post-smoothing seasonal NDVI amplitude ($\Delta\text{NDVI} = \text{NDVI}_{\max} - \text{NDVI}_{\min}$) averages $0.369 \pm 0.091$, with corn exhibiting the highest dynamic range ($0.460 \pm 0.093$) and winter rapeseed the lowest ($0.333 \pm 0.061$). The quality threshold of $\Delta\text{NDVI} \ge 0.15$ effectively filters out dormant land parcels and perennial cover misclassifications (Fig. 1a). Crucially, point-wise stage assignment induces severe class imbalance: Fallow dominates 65.00% of all timesteps, whereas Vegetative (12.66%), Senescence (13.94%), and particularly Reproductive (8.40%) are substantially underrepresented (Fig. 1b). This temporal imbalance contributes to the acute sensitivity of non-sequential tabular baselines and highlights the necessity of bidirectional temporal context modeling.</p>

<div class="figure">
  <img src="IMAGE_EDA_PLACEHOLDER" alt="EDA Amplitude and Class Distribution" style="max-height: 125px;">
  <div class="fig-caption"><strong>Fig. 1.</strong> Exploratory data analysis on the 2,655 retained crop seasons from BreizhCrops (frh01 region, 2017): (a) Seasonal NDVI amplitude across the four crop types with the 0.15 quality filter threshold; (b) Point-wise stage label distribution showing pronounced temporal class imbalance.</div>
</div>

<p><strong>Pseudo-Ground-Truth Label Generation and Evaluation Scope.</strong> Because BreizhCrops provides only crop-type annotations and no field-validated phenological observations (e.g., ground BBCH stages), stage labels are derived using the classical 20%-of-seasonal-amplitude threshold method [1, 2, 13], aligned with operational protocols in automated national-scale monitoring systems such as Sen2-Agri [15]. Specifically, the smoothed NDVI peak is located, and the contiguous above-threshold region around it defines the active season (SOS to EOS). Within this region, a window surrounding the peak is labeled Reproductive, the ascending limb is labeled Vegetative, and the descending limb is labeled Senescence; all remaining timesteps are labeled Fallow. We explicitly disclose this as an essential scope boundary: models are evaluated on their ability to reproduce a standardized rule-based labeling rather than independently verified in-situ agronomic ground truth.</p>

<div class="table-box">
  <div class="table-caption"><strong>Table 1.</strong> Summary of the BreizhCrops dataset across initial and quality-filtered retained subsets.</div>
  <table>
    <thead>
      <tr>
        <th class="left">Crop Type</th>
        <th>Initial</th>
        <th>Retained</th>
        <th>Train (70%)</th>
        <th>Validation (15%)</th>
        <th>Test (15%)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td class="left">Barley</td><td>832</td><td>614</td><td>430</td><td>92</td><td>92</td></tr>
      <tr><td class="left">Wheat</td><td>832</td><td>626</td><td>438</td><td>94</td><td>94</td></tr>
      <tr><td class="left">Rapeseed</td><td>832</td><td>772</td><td>540</td><td>116</td><td>116</td></tr>
      <tr><td class="left">Corn</td><td>832</td><td>643</td><td>449</td><td>97</td><td>97</td></tr>
      <tr><td class="left"><strong>Total</strong></td><td><strong>3,328</strong></td><td><strong>2,655</strong></td><td><strong>1,857</strong></td><td><strong>399</strong></td><td><strong>399</strong></td></tr>
    </tbody>
  </table>
</div>

<h2><span class="subsec-num">3.3</span> Proposed CBA-PhenoNet Architecture</h2>

<p>As illustrated in Fig. 2, CBA-PhenoNet (Calendar-Biometric Attention PhenoNet) comprises three primary components: a cyclical date-aware embedding, stacked gated self-attention layers, and a linear segmentation head.</p>

<div class="figure">
  <img src="IMAGE_ARCH_PLACEHOLDER" alt="CBA-PhenoNet Architecture" style="max-height: 185px;">
  <div class="fig-caption"><strong>Fig. 2.</strong> Architectural overview of CBA-PhenoNet, featuring Cyclical DOY Positional Encoding, Multi-Head Self-Attention, and the Calendar Gating Prior for point-wise segmentation and derived phenology DOY extraction.</div>
</div>

<p><strong>Cyclical Day-of-Year Positional Encoding:</strong> Unlike standard Transformers that index sequences ordinally ($0, \dots, T-1$), CBA-PhenoNet encodes actual acquisition timing using circular harmonics:</p>

<div class="equation">
  <span class="eq-content">$$e_{\text{doy}}(t) = \left[ \sin\left(\frac{2\pi \cdot \text{DOY}_t}{365.0}\right), \; \cos\left(\frac{2\pi \cdot \text{DOY}_t}{365.0}\right) \right] \in \mathbb{R}^2$$</span>
  <span class="eq-num">(1)</span>
</div>

<p>A learnable linear layer projects $e_{\text{doy}}(t)$ to the latent embedding dimension $d_{\text{model}} = 64$ (192 parameters), which is added directly to the linear projection of the input features (`nn.Linear(3, 64)`, 256 parameters). <em>Decoupled Feature and Positional Representation:</em> Although continuous cyclical DOY encodings are included in input channels 1&ndash;2 as local content features, Eq. (1) independently injects a structural, sequence-level temporal coordinate through a learned projection into the latent space. This architectural design explicitly decouples radiometric-temporal observation features from structural cyclical position coordinates in the attention mechanisms.</p>

<p><strong>Calendar Gating Prior:</strong> To provide a calendar-conditioned prior over developmental windows across the annual sequence, we introduce a non-linear temporal gating module $g(t)$:</p>

<div class="equation">
  <span class="eq-content">$$g(t) = \sigma\left( W_2 \cdot \text{ReLU}\left(W_1 \cdot e_{\text{doy}}(t) + b_1\right) + b_2 \right) \in (0, 1)$$</span>
  <span class="eq-num">(2)</span>
</div>

<div class="equation">
  <span class="eq-content">$$h_{\text{gated}}(t) = h_{\text{encoder}}(t) \odot g(t)$$</span>
  <span class="eq-num">(3)</span>
</div>

<p>where $h_{\text{encoder}}(t)$ denotes the output of the multi-head self-attention block ($h=4$ attention heads, $d_{\text{ff}}=128$, dropout = 0.1, 33,472 parameters per layer). Across the two stacked layers, the gating modules add $2 \times 65 = 130$ parameters. Because the regularized 5-day time grid is shared across all samples, $g(t)$ acts as a learned calendar-dependent modulation over attention representations. Gated representations pass through layer normalization into a linear head outputting 4-class logits $\hat{\mathbf{y}}_t \in \mathbb{R}^4$ (260 parameters).</p>

<p><strong>Parameter Breakdown:</strong> The baseline Transformer comprises 67,460 parameters (input projection: 256; two encoder layers: 66,944; classifier: 260) with fixed sinusoidal positional encoding. CBA-PhenoNet totals 67,782 parameters, introducing an additional 322 parameters (+0.48%) that cleanly decouple into: (a) 192 parameters for the learned DOY positional projection, and (b) 130 parameters for the two calendar gating modules.</p>

<h2><span class="subsec-num">3.4</span> Evaluation Metrics and Loss Function</h2>

<p>The network is optimized using masked cross-entropy loss:</p>

<div class="equation">
  <span class="eq-content">$$\mathcal{L} = -\frac{1}{\sum_{t=1}^T m_t} \sum_{t=1}^T m_t \sum_{c=0}^3 y_{t,c} \log \hat{p}_{t,c}$$</span>
  <span class="eq-num">(4)</span>
</div>

<p>where $m_t \in \{0, 1\}$ indicates valid observations, and $\hat{p}_{t,c} = \text{softmax}(\hat{\mathbf{y}}_t)_c$ is the predicted class probability. Derived phenology dates (SOS, POS, EOS) are evaluated against targets using circular Day-of-Year absolute error:</p>

<div class="equation">
  <span class="eq-content">$$\Delta_{\text{DOY}}(y, \hat{y}) = \min\left( |y - \hat{y}|, \; 365.0 - |y - \hat{y}| \right)$$</span>
  <span class="eq-num">(5)</span>
</div>

<p>From $\Delta_{\text{DOY}}$, we compute Mean Absolute Error ($\text{MAE} = \frac{1}{N}\sum \Delta_{\text{DOY}}$), Root Mean Squared Error ($\text{RMSE} = \sqrt{\frac{1}{N}\sum (\Delta_{\text{DOY}})^2}$), Median Absolute Error ($\text{MedAE} = \text{median}(\Delta_{\text{DOY}})$), and tolerance accuracies ($\text{Acc}\pm k\text{d} = \frac{1}{N}\sum \mathbb{I}(\Delta_{\text{DOY}} \le k) \times 100\%$). Segmentation quality across timesteps is evaluated via Macro-F1: $\frac{1}{4}\sum_{c=0}^3 F_{1,c}$.</p>

<!-- SECTION 4 -->
<h1><span class="sec-num">4</span> Experiments and Results</h1>

<h2><span class="subsec-num">4.1</span> Experimental Setup</h2>

<p>All deep learning models (1D-CNN, Bi-LSTM [12], standard Transformer [5], and CBA-PhenoNet) were implemented in PyTorch and trained using the AdamW optimizer (learning rate $1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$, batch size 64) with early stopping monitored on validation Macro-F1 (patience = 15 epochs, maximum 150 epochs). Classical baselines (Random Forest [10] with 100 trees; XGBoost [11] with max depth 6) were trained on tabular representations constructed from backward lag features ($k=3$), rolling statistics ($w=3, 5$), and first-order discrete NDVI derivatives. We note that tabular baselines receive causal, per-timestep features without future trajectory context, whereas deep sequential models process the entire 73-step sequence bidirectionally. This architectural context disparity is intrinsic to comparing per-timestep tabular classifiers with full-sequence neural networks.</p>

<h2><span class="subsec-num">4.2</span> Main Benchmark Comparison</h2>

<p>Table 2 presents quantitative results across all six architectures on the 399 test seasons. For attention-based models (Transformer and CBA-PhenoNet), we report the mean and standard deviation across five random initialization seeds (42, 123, 2024, 7, 314) to evaluate reproducibility.</p>

<div class="table-box">
  <div class="table-caption"><strong>Table 2.</strong> Quantitative benchmark comparison across 6 models on 399 test crop seasons. For Transformer and CBA-PhenoNet, values are reported as mean &plusmn; std across 5 random seeds. Best results in bold.</div>
  <table>
    <thead>
      <tr>
        <th class="left">Model</th>
        <th>Macro-F1</th>
        <th>SOS MAE (d)*</th>
        <th>POS MAE (d)*</th>
        <th>EOS MAE (d)*</th>
        <th>Train Time (s)</th>
        <th>Inference (ms)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td class="left">Random Forest</td><td>0.420</td><td>116.7</td><td>45.0</td><td>92.6</td><td>32.6</td><td>62.17</td></tr>
      <tr><td class="left">XGBoost</td><td>0.403</td><td>88.4</td><td>21.2</td><td>67.7</td><td><strong>7.8</strong></td><td>9.82</td></tr>
      <tr><td class="left">1D-CNN</td><td>0.636</td><td>69.9</td><td>22.4</td><td>45.3</td><td>107.9</td><td>0.32</td></tr>
      <tr><td class="left">Bi-LSTM</td><td>0.778</td><td>36.7</td><td>12.9</td><td>23.0</td><td>166.7</td><td>0.52</td></tr>
      <tr><td class="left">Transformer</td><td><strong>0.904 &plusmn; 0.009</strong></td><td><strong>13.7 &plusmn; 1.2</strong></td><td>3.7 &plusmn; 1.5</td><td>8.5 &plusmn; 0.7</td><td>378.5</td><td>0.28</td></tr>
      <tr><td class="left"><strong>CBA-PhenoNet</strong></td><td>0.897 &plusmn; 0.016</td><td>14.4 &plusmn; 2.8</td><td><strong>2.2 &plusmn; 1.1</strong></td><td><strong>7.0 &plusmn; 1.0</strong></td><td>381.9</td><td><strong>0.27</strong></td></tr>
    </tbody>
  </table>
  <div style="font-size: 7.2pt; text-align: justify; margin-top: 3pt; line-height: 9.2pt;">*Note: Milestone MAE values are evaluated on detected transition subsets. Transition detection recall rates across the 399 test seasons are: Random Forest: SOS 64.7% (258/399), POS 96.5% (385/399), EOS 50.1% (200/399); XGBoost: SOS 59.4% (237/399), POS 81.7% (326/399), EOS 43.4% (173/399); 1D-CNN: SOS 64.2% (256/399), POS 92.5% (369/399), EOS 47.1% (188/399); Bi-LSTM: SOS 62.2% (248/399), POS 98.7% (394/399), EOS 48.1% (192/399); Transformer: SOS 63.9% (255/399), POS 99.5% (397/399), EOS 56.9% (227/399); CBA-PhenoNet: SOS 64.7% (258/399), POS 100.0% (399/399), EOS 56.6% (226/399) (for multi-seed attention models, counts reflect representative seed 314; 5-seed mean EOS recall is 56.5% for CBA-PhenoNet and 56.8% for Transformer). Attention models achieve higher EOS transition recall while sustaining substantially lower MAE.</div>
</div>

<p>The empirical results demonstrate a clear performance stratification. Classical tabular baselines fail to capture global seasonal dynamics from local lag windows, yielding lower segmentation (Macro-F1 &le; 0.420) and large phenological errors (SOS MAE &gt; 88 days). Notably, tabular models and recurrent/convolutional baselines also achieve lower transition detection recall (e.g., EOS recall 43.4&ndash;50.1% vs. 56.6&ndash;56.9% for attention models), demonstrating that attention architectures not only improve milestone timing accuracy but also detect transitions more reliably across challenging seasons. 1D-CNN improves Macro-F1 to 0.636, while Bi-LSTM achieves 0.778. Attention-based models demonstrate superior sequence modeling capability, exceeding 0.89 Macro-F1 and achieving low milestone errors on detected transitions.</p>

<p>To probe milestone regression behavior beyond mean absolute error, Table 3 reports extended error metrics across all models. Classical tabular models suffer severe variance penalties, with RMSE exceeding 90 days and MedAE above 35 days. While Bi-LSTM improves median error to 5.0 days and reaches 69.9% tolerance accuracy within &plusmn;10 days, attention architectures achieve sub-day median errors (0.7&ndash;1.0 days) and over 80% tolerance accuracy within &plusmn;5 days. Comparing the two attention models, the standard Transformer achieves slightly lower median error (0.7 &plusmn; 0.9 vs. 1.0 &plusmn; 0.9 days) and higher exact-match accuracy (73.2 &plusmn; 2.3% vs. 69.6 &plusmn; 4.5%; Acc &plusmn;5d: 84.3 &plusmn; 1.9% vs. 80.3 &plusmn; 5.2%; Acc &plusmn;10d: 87.3 &plusmn; 1.7% vs. 83.0 &plusmn; 5.3%) across milestones. However, CBA-PhenoNet secures a lower overall RMSE (25.8 &plusmn; 6.6 vs. 31.6 &plusmn; 4.5 days), reflecting fewer catastrophic outlier misses. At the EOS (senescence) boundary, CBA-PhenoNet reduces EOS RMSE from 31.6 &plusmn; 5.0 to 25.0 &plusmn; 6.8 days and MAPE from 4.4% to 3.9%.</p>

<div class="table-box">
  <div class="table-caption"><strong>Table 3.</strong> Extended phenological timing metrics across 6 models on 399 test seasons (averaged across SOS, POS, and EOS milestones). For Transformer and CBA-PhenoNet, values are reported as mean &plusmn; std across 5 random seeds for all metrics. Best results in bold.</div>
  <table>
    <thead>
      <tr>
        <th class="left">Model</th>
        <th>RMSE (d)</th>
        <th>MAPE (%)</th>
        <th>MedAE (d)</th>
        <th>Acc &plusmn;1d (%)*</th>
        <th>Acc &plusmn;5d (%)</th>
        <th>Acc &plusmn;10d (%)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td class="left">Random Forest</td><td>123.0</td><td>66.2</td><td>56.7</td><td>23.5</td><td>29.8</td><td>32.2</td></tr>
      <tr><td class="left">XGBoost</td><td>92.8</td><td>47.4</td><td>36.7</td><td>30.6</td><td>36.5</td><td>39.3</td></tr>
      <tr><td class="left">1D-CNN</td><td>85.8</td><td>47.0</td><td>13.3</td><td>41.3</td><td>54.7</td><td>58.2</td></tr>
      <tr><td class="left">Bi-LSTM</td><td>54.7</td><td>35.5</td><td>5.0</td><td>44.1</td><td>60.9</td><td>69.9</td></tr>
      <tr><td class="left">Transformer</td><td>31.6 &plusmn; 4.5</td><td>7.5 &plusmn; 1.6</td><td><strong>0.7 &plusmn; 0.9</strong></td><td><strong>73.2 &plusmn; 2.3</strong></td><td><strong>84.3 &plusmn; 1.9</strong></td><td><strong>87.3 &plusmn; 1.7</strong></td></tr>
      <tr><td class="left"><strong>CBA-PhenoNet</strong></td><td><strong>25.8 &plusmn; 6.6</strong></td><td><strong>7.4 &plusmn; 3.4</strong></td><td>1.0 &plusmn; 0.9</td><td>69.6 &plusmn; 4.5</td><td>80.3 &plusmn; 5.2</td><td>83.0 &plusmn; 5.3</td></tr>
    </tbody>
  </table>
  <div style="font-size: 7.5pt; text-align: justify; margin-top: 3pt; line-height: 9.5pt;">*Note: On a 5-day regularized grid, adjacent step errors occur in multiples of 5 days; hence, Acc &plusmn;1d serves as an exact-step proxy ($\Delta_{\text{DOY}} = 0$). Acc &plusmn;5d and &plusmn;10d evaluate transitions within &plusmn;1 and &plusmn;2 timesteps. For Transformer, per-milestone RMSE values across 5 seeds are: SOS = 35.7 &plusmn; 4.7 d, POS = 27.5 &plusmn; 8.3 d, and EOS = 31.6 &plusmn; 5.0 d. While POS exhibits 97.3% exact-step accuracy (MedAE = 0.0 d), residual secondary cloud/weed peak outliers (50&ndash;140 d) in the remaining 2.7% quadratically inflate POS RMSE. CBA-PhenoNet suppresses these outlier tails across milestones: SOS = 32.7 &plusmn; 6.4 d, POS = 19.8 &plusmn; 8.9 d, and EOS = 25.0 &plusmn; 6.8 d.</div>
</div>

<h2><span class="subsec-num">4.3</span> Multi-Seed Variance and Reproducibility Analysis</h2>

<p><strong>Methodological Discipline in Multi-Seed Reporting:</strong> A key finding of our benchmark pertains to experimental discipline. Under an isolated single-seed trial (seed 42), CBA-PhenoNet appeared to outperform the standard Transformer across nearly all metrics. However, multi-seed evaluation across five independent seeds reveals that their Macro-F1 distributions overlap substantially (0.904 &plusmn; 0.009 vs. 0.897 &plusmn; 0.016), indicating no statistically distinguishable advantage in overall segmentation quality.</p>

<p><strong>Reproducible Milestone Patterns:</strong> Multi-seed evaluation clarifies the specific empirical behavior of the calendar gating module: CBA-PhenoNet achieves lower EOS (senescence) MAE in 4 out of 5 seeds (7.7 vs 9.7, 6.5 vs 7.7, 6.6 vs 7.9, 5.6 vs 8.5 days) and a tie on seed 7 (8.6 vs 8.6 days), yielding a mean error reduction of 1.5 days (7.0 &plusmn; 1.0 vs. 8.5 &plusmn; 0.7 days; unadjusted paired <em>t</em>-test: <em>t</em> = 3.11, <em>p</em> = 0.036). When adjusting for multiple comparisons across the three milestones via Bonferroni correction, the adjusted significance is <em>p</em><sub>adj</sub> &asymp; 0.108. We therefore interpret this 1.5-day difference (which is smaller than a single 5-day grid step) as an exploratory trend rather than definitive superiority.</p>

<div class="figure">
  <img src="IMAGE_SCATTER_PLACEHOLDER" alt="Actual vs Predicted Scatter Plot" style="max-height: 150px;">
  <div class="fig-caption"><strong>Fig. 3.</strong> Actual vs. Predicted DOY scatter plots for CBA-PhenoNet (seed 314) across SOS (<em>R</em><sup>2</sup> = 0.895, MAE = 13.4d, <em>n</em> = 258), POS (<em>R</em><sup>2</sup> = 0.994, MAE = 0.9d, <em>n</em> = 399), and EOS (<em>R</em><sup>2</sup> = 0.975, MAE = 5.6d, <em>n</em> = 226) on detected test instances.</div>
</div>

<h2><span class="subsec-num">4.4</span> Accuracy vs. Computational Efficiency Trade-off</h2>

<p>As illustrated in Fig. 4, deep sequence models achieve superior Pareto efficiency compared to classical ML. Tabular feature extraction and inference require per-timestep loops that elevate latency to 9.82&ndash;62.17 ms per season. In contrast, parallel vectorized matrix multiplications in CBA-PhenoNet execute in 0.27 ms per season (equivalent to Transformer at 0.28 ms within hardware measurement variance), demonstrating that attention models can be deployed efficiently for regional parcel monitoring.</p>

<div class="figure">
  <img src="IMAGE_TRADEOFF_PLACEHOLDER" alt="Accuracy vs Latency Tradeoff" style="max-height: 140px;">
  <div class="fig-caption"><strong>Fig. 4.</strong> Macro-F1 accuracy vs. inference latency (log scale) across the six evaluated architectures.</div>
</div>

<!-- SECTION 5 -->
<h1><span class="sec-num">5</span> Discussion and Error Analysis</h1>

<p><strong>Synthesizing RQ1 and RQ2:</strong> Regarding RQ1, formulating phenology extraction as a 4-class temporal segmentation problem with bidirectional sequence models provides substantial gains over tabular ML (Macro-F1 +0.484 over Random Forest). This advantage stems primarily from bidirectional context modeling across the full growth season rather than local autoregressive features. Regarding RQ2, calendar-conditioned attention achieves sub-millisecond inference (0.27 ms) and lower RMSE outlier tails, but its modest 1.5-day reduction in EOS MAE remains exploratory given that overall Macro-F1 and tolerance accuracies remain comparable to the standard Transformer.</p>

<p><strong>Error Analysis:</strong> Inspection of the top 5% worst milestone prediction instances (44 instances out of 883 detected test milestones, pooled across SOS <em>n</em> = 258, POS <em>n</em> = 399, and EOS <em>n</em> = 226 from seed 314) reveals two primary failure modes: (1) <em>Asymmetric Error in Rapeseed:</em> Rapeseed accounted for 18 of the 44 most severe error instances (40.9%) despite representing 29.1% (116/399) of the test cohort (expected count &asymp; 12.8). In Western Europe, winter rapeseed is autumn-sown and establishes significant overwinter green canopy, producing early vegetative pseudo-peaks that confound models trained on standard spring growth profiles. (2) <em>SOS Boundary Ambiguity:</em> Across all models, SOS proved systematically harder to detect (MAE ~14 days) than POS (~2&ndash;4 days) or EOS (~7&ndash;9 days). During green-up, emerging weeds and variable soil moisture create low signal-to-noise ratios in NDVI, causing gradual slope changes that are harder to demarcate than the rapid spectral drops characteristic of canopy senescence.</p>

<p><strong>Limitations:</strong> The primary limitation of this study is the reliance on pseudo-ground-truth phenological labels derived from a 20% seasonal amplitude threshold on smoothed NDVI. The reported metrics evaluate the capacity of sequence models to emulate and automate a deterministic rule-based heuristic, rather than field-validated agronomic observations (such as ground BBCH records). Furthermore, pre-filtering flat trajectories (&Delta;NDVI &lt; 0.15) and edge-peaking seasons removes damaged or non-standard growth cycles, which represents a disclosed selection bias. Future work should validate against PhenoCam camera networks, evaluate multi-modal SAR (Sentinel-1) fusion for persistent cloud cover, and assess transferability to complex tropical multi-cropped rice ecosystems [9].</p>

<!-- SECTION 6 -->
<h1><span class="sec-num">6</span> Conclusion and Future Work</h1>

<p>This paper presented an open benchmark for Sentinel-2 NDVI crop phenology extraction, formulating the challenge as a 4-class temporal segmentation problem across 2,655 quality-filtered seasons (<em>T</em> = 73) from BreizhCrops. We investigated CBA-PhenoNet, combining cyclical DOY positional encoding with a calendar gating prior. While multi-seed evaluation demonstrates that overall segmentation Macro-F1 is statistically comparable between Transformer (0.904 &plusmn; 0.009) and CBA-PhenoNet (0.897 &plusmn; 0.016), CBA-PhenoNet demonstrates exploratory suppression of severe outlier timing errors at the senescence boundary (EOS MAE 7.0 &plusmn; 1.0 vs. 8.5 &plusmn; 0.7 days; lower in 4/5 seeds with 1 tie) with sub-millisecond inference latency (0.27 ms). Future extensions include validating on ground-truth PhenoCam networks and extending the framework to tropical multi-cropped rice ecosystems [9].</p>

<div class="disclosure">
  <h3>Disclosure of Interests.</h3> The author has no competing interests to declare that are relevant to the content of this article.
</div>

<div class="disclosure">
  <h3>Code and Data Availability.</h3> The complete PyTorch codebase, trained weights, and benchmark evaluation scripts are openly available at <a href="https://github.com/vuhahodo/crop-phenology-prediction" style="color: inherit; text-decoration: none;">https://github.com/vuhahodo/crop-phenology-prediction</a>. The BreizhCrops dataset is publicly available at <a href="https://github.com/dl4sits/BreizhCrops" style="color: inherit; text-decoration: none;">https://github.com/dl4sits/BreizhCrops</a>.
</div>

<!-- REFERENCES -->
<h1>References</h1>

<div class="ref-list">
  <div class="ref-item">1. J&ouml;nsson, P., Eklundh, L.: TIMESAT&mdash;a program for analyzing time-series of satellite sensor data. Computers &amp; Geosciences 30(8), 833&ndash;845 (2004)</div>
  <div class="ref-item">2. Zhang, X., Friedl, M.A., Schaaf, C.B., Strahler, A.H., Hodges, J.C., Gao, F., Reed, B.C., Huete, A.: Monitoring vegetation phenology using MODIS. Remote Sensing of Environment 84(3), 471&ndash;475 (2003)</div>
  <div class="ref-item">3. Zeng, L., Wardlow, B.D., Xiang, D., Hu, S., Li, D.: A review of vegetation phenological metrics extraction using time-series, multispectral satellite data. Remote Sensing of Environment 237, 111511 (2020)</div>
  <div class="ref-item">4. Pelletier, C., Webb, G.I., Petitjean, F.: Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series. Remote Sensing 11(5), 523 (2019)</div>
  <div class="ref-item">5. Sainte Fare Garnot, V., Landrieu, L., Giordano, S., Chehata, N.: Satellite image time series classification with pixel-set encoders and temporal self-attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12325&ndash;12334 (2020)</div>
  <div class="ref-item">6. Tarasiou, M., Chavez, E., Zafeiriou, S.: ViTs for SITS: Vision Transformers for Satellite Image Time Series. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10418&ndash;10428 (2023)</div>
  <div class="ref-item">7. Beck, P.S., Atzberger, C., H&oslash;gda, K.A., Johansen, B., Skidmore, A.K.: Improved monitoring of vegetation dynamics at very high latitudes: A new method using MODIS NDVI. Remote Sensing of Environment 100(3), 321&ndash;334 (2006)</div>
  <div class="ref-item">8. Ru&szlig;wurm, M., Pelletier, C., Zollner, M., Lef&egrave;vre, S., K&ouml;rner, M.: BreizhCrops: A Time Series Dataset for Crop Type Mapping. Int. Arch. Photogramm. Remote Sens. Spat. Inf. Sci. XLIII-B2-2020, 1545&ndash;1551 (2020)</div>
  <div class="ref-item">9. Boschetti, M., Nutini, F., Brivio, P.A., Bartholome, E., Stroppiana, D.: Identifying rice cropping systems and phenology in the Mekong Delta using MODIS time series. Remote Sensing 1(4), 1147&ndash;1170 (2009)</div>
  <div class="ref-item">10. Breiman, L.: Random Forests. Machine Learning 45(1), 5&ndash;32 (2001)</div>
  <div class="ref-item">11. Chen, T., Guestrin, C.: XGBoost: A scalable tree boosting system. In: Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., pp. 785&ndash;794 (2016)</div>
  <div class="ref-item">12. Schuster, M., Paliwal, K.K.: Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing 45(11), 2673&ndash;2681 (1997); and Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural Computation 9(8), 1735&ndash;1780 (1997)</div>
  <div class="ref-item">13. Savitzky, A., Golay, M.J.: Smoothing and differentiation of data by simplified least squares procedures. Analytical Chemistry 36(8), 1627&ndash;1639 (1964)</div>
  <div class="ref-item">14. Jakubauskas, M.E., Peterson, D.L., Kastens, J.H.: Harmonic analysis of time-series AVHRR NDVI data for characterizing US Great Plains land use/land cover. Int. J. Remote Sens. 22(18), 3739&ndash;3761 (2001)</div>
  <div class="ref-item">15. Defourny, P., Bontemps, S., Bellemans, N., Cara, C., Dedieu, G., Guzzonato, E., Hagolle, O., Inglada, J., Nicola, L., Rabaute, T.: Near real-time agriculture monitoring at national scale at parcel resolution: Performance assessment of the Sen2-Agri automated system in various cropping systems around the world. Remote Sens. Environ. 221, 551&ndash;568 (2019)</div>
</div>

</body>
</html>"""

def stamp_springer_headers(input_pdf: Path, output_pdf: Path):
    """Post-processes PDF with PyMuPDF to stamp exact Springer LNCS running headers.
    - Page 1: NO header, NO footer.
    - Even pages (2, 4, 6...): Left-aligned page number, Centered author name ('Ha Do-Phuc-Vu').
    - Odd pages (3, 5, 7...): Centered short title ('Crop Phenology Detection from Sentinel-2 NDVI Time Series'), Right-aligned page number.
    - Footer: completely clean on all pages.
    """
    doc = fitz.open(input_pdf)
    num_pages = len(doc)
    
    author_text = "Ha Do-Phuc-Vu"
    title_text = "Crop Phenology Detection from Sentinel-2 NDVI Time Series"
    font_name = "times-roman"
    font_size = 10.0
    
    left_x = 124.7
    right_x = 472.6
    center_x = (left_x + right_x) / 2.0  # 298.65 pt
    header_y = 125.0  # ~44 mm from top edge
    
    for page_idx in range(num_pages):
        page_num = page_idx + 1
        if page_num == 1:
            continue
        
        page = doc[page_idx]
        
        if page_num % 2 == 0:
            page.insert_text((left_x, header_y), str(page_num), fontname=font_name, fontsize=font_size)
            author_w = fitz.get_text_length(author_text, fontname=font_name, fontsize=font_size)
            page.insert_text((center_x - author_w / 2.0, header_y), author_text, fontname=font_name, fontsize=font_size)
        else:
            title_w = fitz.get_text_length(title_text, fontname=font_name, fontsize=font_size)
            page.insert_text((center_x - title_w / 2.0, header_y), title_text, fontname=font_name, fontsize=font_size)
            num_str = str(page_num)
            num_w = fitz.get_text_length(num_str, fontname=font_name, fontsize=font_size)
            page.insert_text((right_x - num_w, header_y), num_str, fontname=font_name, fontsize=font_size)

    doc.save(output_pdf)
    doc.close()

def main():
    img_arch = image_to_base64(PLOTS_DIR / "cba_phenonet_architecture.png")
    img_eda = image_to_base64(PLOTS_DIR / "eda_distribution.png")
    img_scatter = image_to_base64(PLOTS_DIR / "cba_phenonet_actual_vs_predicted.png")
    img_tradeoff = image_to_base64(PLOTS_DIR / "accuracy_vs_latency_tradeoff.png")

    html = HTML_TEMPLATE.replace("IMAGE_ARCH_PLACEHOLDER", img_arch)
    html = html.replace("IMAGE_EDA_PLACEHOLDER", img_eda)
    html = html.replace("IMAGE_SCATTER_PLACEHOLDER", img_scatter)
    html = html.replace("IMAGE_TRADEOFF_PLACEHOLDER", img_tradeoff)

    html_path = DOCS_DIR / "manuscript_springer_lncs.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote HTML preview to: {html_path}")

    raw_pdf_path = DOCS_DIR / "raw_springer_unheaded.pdf"
    final_pdf_path = DOCS_DIR / "DoPhucVuHa_CropPhenology_Springer.pdf"
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        page.wait_for_timeout(2000)
        # Remove hidden KaTeX MathML elements so Chromium does not dump them at the bottom of pages in PDF text layer
        page.evaluate('() => { document.querySelectorAll(".katex-mathml").forEach(el => el.remove()); document.querySelectorAll(".katex-html").forEach(el => el.removeAttribute("aria-hidden")); }')
        
        # Output clean body without any chromium default headers/footers
        page.pdf(
            path=str(raw_pdf_path),
            format="A4",
            margin={"top": "52mm", "bottom": "52mm", "left": "44mm", "right": "44mm"},
            print_background=True,
            display_header_footer=False
        )
        browser.close()
    
    # Stamp the exact Springer headers using PyMuPDF
    stamp_springer_headers(raw_pdf_path, final_pdf_path)
    if raw_pdf_path.exists():
        raw_pdf_path.unlink()
        
    print(f"Successfully generated publication-grade Springer PDF at: {final_pdf_path}")

if __name__ == "__main__":
    main()
