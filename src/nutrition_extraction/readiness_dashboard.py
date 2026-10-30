"""Cached pre-inference evidence UI: no model outputs, no OCR or tokenizer in server."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data(show_spinner='Loading cached pre-inference readiness…')
def _load(directory, signature):
    del signature
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['prediction_caches_read'] or manifest['model_inference_runs'] or manifest['ocr_runs']:
        raise ValueError('Readiness artifact contains a nonzero inference/OCR dependency.')
    root = directory.parents[1]
    for path in (root / 'outputs/part1_part2_integration').glob('*.parquet'):
        expected = next((digest for name, digest in manifest['source_sha256'].items() if Path(name) == path), None)
        if expected != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Integrated inputs changed; rerun the separate preprocessing notebook.')
    tables = {name: pd.read_parquet(directory / f'{name}.parquet') for name in ['images', 'token_examples', 'bio_transition_examples', 'entities', 'label_counts', 'metadata', 'proxy_experiments']}
    return manifest, tables, pd.read_csv(directory / 'recommendations.csv')


def load(directory):
    directory = Path(directory)
    inputs = sorted(directory.glob('*.parquet')) + [directory / 'manifest.json', directory / 'recommendations.csv']
    inputs += sorted((directory.parents[1] / 'outputs/part1_part2_integration').glob('*.parquet'))
    return _load(str(directory), tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in inputs))


def render(root):
    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.preprocessing_readiness import filtered_images, labels
    st.subheader('Pre-Inference EDA & Readiness')
    directory = root / 'outputs/preprocessing_readiness'
    if not (directory / 'manifest.json').exists():
        st.info('Run part2_preprocessing_eda.ipynb to build the pre-inference audit. No readiness statistics are inferred from model predictions.')
        return
    try:
        manifest, data, recommendations = load(directory)
    except (OSError, ValueError, ImportError) as error:
        st.error(f'Readiness cache unavailable: {error}')
        return
    st.info('Before inference: source image/metadata, supplied OCR words, original boxes, gold annotation integrity and local-tokenizer lengths only. Zero model predictions, F1 scores, OCR runs or model forward passes. Gold labels are diagnostic annotations—not runtime features for unseen inputs.')
    kpis = st.columns(4)
    kpis[0].metric('Annotated images audited', f"{manifest['total_annotated_images']:,}")
    kpis[1].metric('Barcode-matched products', f"{manifest['matched_products']:,}", f"{manifest['unmatched_images']} unmatched images retained")
    kpis[2].metric('Pixel-measured originals', str(manifest['pixel_measured_images']), 'cached convenience sample only')
    kpis[3].metric('Overlength tokenizer inputs', str(manifest['sliding_window_images']), f"of {manifest['tokenizer_measured_images']:,} measured")
    st.caption('Fixed whole-audit KPIs; filters below affect diagnostic charts and tables only. Sidebar EDA filters do not apply on this page. No full-source scan, image download or inference is performed by the dashboard.')
    images = data['images']
    controls = st.columns(4)
    with controls[0]: splits = st.multiselect('Readiness official split', ['train', 'test'], key='readiness_split')
    with controls[1]: nutrients = st.multiselect('Readiness nutrient', sorted({value for text in images['nutrients'] for value in text.split('|') if value}), key='readiness_nutrient')
    with controls[2]: countries = st.multiselect('Readiness country', sorted({value for text in images['country'] for value in labels(text)}), key='readiness_country')
    with controls[3]: flags = st.multiselect('Readiness quality flag', sorted({value for text in images['quality_flags'] for value in text.split('|') if value}), key='readiness_flag')
    selected = filtered_images(images, splits=splits, countries=countries, nutrients=nutrients, quality_flags=flags)
    keys = set(selected['image_key'])
    pixel = selected.loc[selected['pixel_status'].eq('measured_cached_original')]
    st.caption(f"Selection: {len(selected):,} source images, {len(pixel)} with measured local pixels, {selected['needs_sliding_window'].eq(True).sum()} overlength. Country filters exclude missing/unmatched product context; split assignments never change.")
    if selected.empty:
        st.info('No source images match these readiness filters. No zero-quality or score statistics are invented.')
        return

    st.subheader('Image resolution, aspect ratio and pixel-quality proxies')
    st.plotly_chart(px.histogram(selected, x='width', color='split', nbins=40, hover_data=['height', 'match_status']), width='stretch')
    st.caption('Resolution comes from metadata for the complete selected inventory. A short side below 400px is a legibility-review flag; upscaling cannot restore missing text. It is not proof of OCR failure.')
    st.plotly_chart(px.histogram(selected, x='aspect_ratio', color='split', nbins=40), width='stretch')
    st.caption('Aspect ratio identifies long/portrait panels and potential layout complexity. It does NOT identify sideways or upside-down text; rotation correction needs EXIF or reviewed text orientation evidence.')
    st.dataframe(selected.groupby(['split', 'match_status']).agg(images=('image_key', 'size'), short_side_review=('short_side_review', 'sum'), crop_margin_review=('large_margin_crop_review', 'sum')).reset_index(), width='stretch', hide_index=True)
    if not pixel.empty:
        for field in ['brightness_mean', 'contrast_std', 'laplacian_variance']:
            st.plotly_chart(px.histogram(pixel, x=field, color='split', nbins=25, hover_data=['barcode', 'measure_width', 'measure_height']), width='stretch')
        st.caption(f"Pixel scope: {len(pixel)}/{len(selected)} selected images, from {manifest['pixel_measured_images']} cached originals globally (not random). Brightness/contrast are grayscale mean/std; blur indicator is Laplacian variance at max-side 1000px without upscaling. Thresholds are uncalibrated review screens, not OCR accuracy. White backgrounds can raise brightness and detailed packaging can raise sharpness.")
        st.dataframe(pixel[['image_key', 'barcode', 'exif_orientation', 'source_mode', 'brightness_mean', 'contrast_std', 'laplacian_variance', 'quality_flags']], width='stretch', hide_index=True)
    else:
        st.info('No cached pixels for this selection. Brightness, contrast, blur and EXIF remain unmeasured, not assumed good.')
    st.caption('Crop review uses the envelope of ALL supplied OCR boxes, not an annotated nutrition-table region. Small envelopes justify inspection only; automatic cropping could delete context/targets.')

    st.subheader('OCR readiness · supplied corpus OCR, not new EasyOCR output')
    diagnostic_columns = ['decimal_comma_tokens', 'decimal_point_tokens', 'merged_number_unit_tokens', 'multiple_numeric_tokens', 'malformed_numeric_review_tokens', 'long_tokens', 'empty_tokens', 'backward_y_jumps']
    diagnostic_summary = pd.DataFrame([{'diagnostic': name, 'affected_images': int(selected[name].gt(0).sum()), 'image_denominator': len(selected), 'fraction': float(selected[name].gt(0).mean()), 'occurrences': int(selected[name].sum())} for name in diagnostic_columns])
    st.plotly_chart(px.bar(diagnostic_summary, x='diagnostic', y='affected_images', hover_data=['image_denominator', 'fraction', 'occurrences']), width='stretch')
    st.caption('Decimal commas and merged number/unit strings require robust parsing—not mutation of OCR/gold tokens. Multiple numbers, long strings and glyph anomalies indicate segmentation review. Backward y-jumps also occur in valid multi-column layouts; never sort annotated words blindly.')
    st.plotly_chart(px.histogram(selected, x='token_count', color='split', nbins=40), width='stretch')
    st.caption('Token-count distribution quantifies text density and expected segmentation workload before encoding. Token lengths are preserved, not re-tokenized as gold corrections.')
    examples = data['token_examples'].loc[data['token_examples']['image_key'].isin(keys)]
    st.dataframe(examples.head(200), width='stretch', hide_index=True)
    st.dataframe(selected[['image_key', 'mean_token_characters', 'max_token_characters', 'backward_y_jumps']], width='stretch', hide_index=True)

    st.subheader('Bounding-box validity and token/box/label alignment')
    box_columns = ['invalid_box_count', 'out_of_range_box_count', 'zero_area_boxes', 'reversed_boxes', 'duplicate_box_excess', 'overlapping_pairs', 'high_iou_pairs']
    box_summary = pd.DataFrame([{'diagnostic': name, 'affected_images': int(selected[name].gt(0).sum()), 'image_denominator': len(selected), 'box_or_pair_occurrences': int(selected[name].sum())} for name in box_columns])
    st.plotly_chart(px.bar(box_summary, x='diagnostic', y='affected_images', hover_data=['image_denominator', 'box_or_pair_occurrences']), width='stretch')
    st.caption('Original boxes must be integer [x0,y0,x1,y1] in [0,1000] with positive area and aligned words/labels. Repeated/high-IoU boxes are review flags, not permission to run NMS on gold. Normalization is already present; do not normalize a second time.')
    st.dataframe(selected[['image_key', 'barcode', 'token_box_alignment', 'token_index_contiguous', 'label_ids_consistent', *box_columns, 'invalid_box_indices']], width='stretch', hide_index=True)

    st.subheader('Model input length · tokenizer only, no LayoutLM forward pass')
    measured = selected.loc[selected['subword_tokens_with_specials'].notna()]
    st.caption(f"Exact local fast-tokenizer length measured on {len(measured)}/{len(selected)} selected images. Raw geometry is not approved by a successful tokenizer call. Length includes special tokens, not image patches; model text limit is 512.")
    if not measured.empty:
        st.plotly_chart(px.scatter(measured, x='token_count', y='subword_tokens_with_specials', color='split', symbol='needs_sliding_window', hover_data=['barcode', 'words_missed_by_single_pass', 'max_subwords_per_word']), width='stretch')
        st.caption('A word count below 512 can still overflow after subword splitting. Required action for measured overlength records: keep overlapping windows and word_ids alignment; no silent truncation. Single-pass coverage reserves both special tokens and counts word starts, not complete last-word subwords or inference accuracy.')
        st.dataframe(measured.groupby(['split', 'match_status']).agg(images=('image_key', 'size'), sliding_window=('needs_sliding_window', 'sum'), max_subwords=('subword_tokens_with_specials', 'max'), single_pass_missed_words=('words_missed_by_single_pass', 'sum')).reset_index(), width='stretch', hide_index=True)

    st.subheader('BIO annotation integrity and entity/basis coverage')
    counts = data['label_counts'].loc[data['label_counts']['image_key'].isin(keys)]
    if nutrients:
        nutrient_identity = counts['label'].str.replace(r'^[BI]-', '', regex=True).str.replace(r'_(100G|SERVING)$', '', regex=True).str.lower()
        counts = counts.loc[nutrient_identity.isin(nutrients)]
    support = counts.groupby(['label', 'split'])['count'].sum().reset_index()
    st.plotly_chart(px.bar(support.loc[support['label'].ne('O')], x='count', y='label', color='split', orientation='h', barmode='group'), width='stretch')
    st.caption('Gold BIO frequencies describe annotation availability, not model results. O is omitted from the chart for readability. Invalid I- continuations are checked but never repaired; official splits/labels remain unchanged.')
    st.dataframe(selected[['image_key', 'entity_count', 'multi_token_entities', 'per_100g_entities', 'per_serving_entities', 'serving_size_entities', 'invalid_bio_transitions']], width='stretch', hide_index=True)
    transition_rows = data['bio_transition_examples'].loc[data['bio_transition_examples']['image_key'].isin(keys)]
    st.dataframe(transition_rows, width='stretch', hide_index=True)
    st.caption('Multi-token spans and separate per-100g/per-serving counts justify word-level alignment and basis preservation. Absence of one class is not proof that its nutrient is absent from the image.')

    st.subheader('Product metadata readiness · matched versus unmatched')
    product_rows = data['metadata'].loc[data['metadata']['barcode'].isin(selected['barcode'])]
    if nutrients:
        product_rows = product_rows.loc[product_rows['nutrient'].isin(nutrients)]
    st.dataframe(product_rows.groupby(['match_status', 'nutrient', 'source_unit_convention']).agg(products=('barcode', 'nunique'), finite_values=('finite_numeric', 'sum'), serving_size_available=('serving_size_available', 'sum'), unit_packaging_verified=('unit_verified_from_packaging', 'sum')).reset_index(), width='stretch', hide_index=True)
    st.caption('Product fields use export unit/basis conventions (mass g, energy kJ/kcal); none are declared packaging-verified. Serving context cannot be inferred when missing. Unmatched records are retained and cannot be called extraction failures. Implausible product values are downstream review guards, not a reason to alter image pixels or labels.')
    st.dataframe(product_rows.loc[product_rows['review_flag'].ne('')], width='stretch', hide_index=True)

    st.subheader('Dataset differences · official split and match-status cohorts')
    st.dataframe(selected.groupby(['split', 'match_status']).agg(images=('image_key', 'size'), mean_words=('token_count', 'mean'), median_subwords=('subword_tokens_with_specials', 'median'), median_width=('width', 'median'), geometry_review=('zero_area_boxes', lambda values: int(values.gt(0).sum()))).reset_index(), width='stretch', hide_index=True)
    st.caption('Small unmatched cohorts and the highly test-heavy cached pixel subset can bias quality impressions. Compare denominators before proposing split-specific preprocessing. Do not select/tune preprocessing from model test errors.')

    st.subheader('Representative cached images · measured issues only')
    if not pixel.empty:
        keys_available = pixel.sort_values(['quality_flags', 'image_key'], ascending=[False, True])['image_key'].tolist()
        selected_key = st.selectbox('Readiness image example', keys_available, key='readiness_example')
        example = pixel.loc[pixel['image_key'].eq(selected_key)].iloc[0]
        path = Path(example['local_image_path'])
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == example['image_hash']:
            st.image(str(path), caption=f"{selected_key} · {example['barcode']} · {example['quality_flags'] or 'no threshold flag; not a quality certificate'}", width=600)
        else:
            st.warning('Cached image changed or moved; rerun pixel measurements before displaying evidence.')
        st.link_button('View original source image', example['image_url'])
        st.caption('Original image is unchanged. Flags illustrate evidence for review, not automatic rotation/crop/enhancement instructions.')
    st.subheader('Small copy-only preprocessing experiment')
    experiment = data['proxy_experiments'].loc[data['proxy_experiments']['image_key'].isin(keys)]
    st.dataframe(experiment, width='stretch', hide_index=True)
    st.caption('Eight cached originals were tested with the existing resize+contrast configuration on copies only. Before/after pixel proxies do NOT measure OCR quality. Zero OCR runs were performed: the sample lacks independently aligned manual transcription references. A paired manual-reference OCR experiment is required before claiming benefit or making enhancement unconditional.')
    if not experiment.empty:
        experiment_key = st.selectbox('Readiness preprocessing-copy example', experiment['image_key'].tolist(), key='readiness_proxy_example')
        preview = experiment.loc[experiment['image_key'].eq(experiment_key)].iloc[0]
        col1, col2 = st.columns(2)
        if Path(preview['original_path']).is_file():
            col1.image(preview['original_path'], caption='Original (unchanged)', width='stretch')
        if Path(preview['transformed_path']).is_file():
            col2.image(preview['transformed_path'], caption='Copy: resize + contrast 1.15 (not validated OCR improvement)', width='stretch')

    st.subheader('Observed issue → affected count → preprocessing decision')
    st.caption('Recommendation counts refer to the full audit, not the current interactive selection; scope explicitly separates all-image versus cached-pixel denominators. Required checks, optional enhancements, data-quality flags and downstream validation are distinct.')
    decision_class = st.multiselect('Readiness decision class', sorted(recommendations['decision_class'].unique()), key='readiness_decision_class')
    decisions = recommendations.loc[recommendations['decision_class'].isin(decision_class)] if decision_class else recommendations
    st.dataframe(decisions, width='stretch', hide_index=True)
    with st.expander('Measurement rules, cache provenance and limitations'):
        st.json({key: value for key, value in manifest.items() if key not in ['source_sha256','measured_image_sha256']})
    st.download_button('Download preprocessing recommendations', recommendations.to_csv(index=False).encode('utf-8'), file_name='preprocessing_recommendations.csv', mime='text/csv', key='readiness_download')
    st.download_button('Download selected readiness records', selected.to_csv(index=False).encode('utf-8'), file_name='selected_readiness_images.csv', mime='text/csv', key='readiness_selected_download')
