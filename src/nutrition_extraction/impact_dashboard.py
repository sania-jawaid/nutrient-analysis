"""Read cached EDA→extraction evidence; no full database scan or inference in UI."""
from pathlib import Path
import json
import hashlib
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data(show_spinner='Loading cached EDA → Extraction analysis…')
def _cached(directory, signature):
    del signature
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    root = directory.parents[1]
    required_inputs = [*sorted((root / 'outputs/part1_part2_integration').glob('*.parquet')),
                       root / 'data/part2_cache/layoutlm_bio/latest_test_bundle.json']
    for path in required_inputs:
        expected = next((digest for name, digest in manifest.get('input_sha256', {}).items() if Path(name) == path), None)
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Integrated inputs or prediction bundle changed after this analysis; rerun the separate impact notebook before showing cached findings.')
    tables = {name: pd.read_parquet(directory / f'{name}.parquet') for name in [
        'representativeness', 'label_distributions', 'metadata_coverage', 'distribution_bins',
        'performance_rows', 'coverage_rows', 'anomaly_rows', 'agreement_rows', 'image_rows',
        'anomaly_thresholds', 'error_examples']}
    tables['findings'] = pd.read_csv(directory / 'key_findings.csv')
    return manifest, tables


def load_results(directory):
    directory = Path(directory)
    paths = sorted(directory.glob('*.parquet')) + [directory / 'manifest.json', directory / 'key_findings.csv']
    root = directory.parents[1]
    paths += sorted((root / 'outputs/part1_part2_integration').glob('*.parquet'))
    paths += [root / 'data/part2_cache/layoutlm_bio/latest_test_bundle.json']
    return _cached(str(directory), tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths))


def render(root):
    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.eda_extraction_impact import filtered_rows, coverage_summary, performance_groups, metric_summary, labels
    st.subheader('Post-Inference Diagnostics')
    st.caption('Retained EDA → Extraction Insights analysis: cached model performance, error-driver strata and cross-source agreement. Preprocessing evidence before inference is now in the separate Pre-Inference EDA & Readiness section.')
    directory = root / 'outputs/eda_extraction_impact'
    if not (directory / 'manifest.json').exists():
        st.info('Run part1_eda_impact_on_part2.ipynb to compute the offline analysis. This dashboard never scans the complete export or runs model inference.')
        return
    try:
        manifest, data = load_results(directory)
    except (OSError, ValueError, ImportError) as error:
        st.error(f'Cached insight results unavailable: {error}')
        return
    st.info('Observational evidence audit, not a causal experiment. Uses the complete local product snapshot, the 3,075 matched products and aligned cached official-test predictions. Product metadata is not image-verified gold. Existing EDA and official model evaluation are unchanged.')
    if manifest.get('existing_notebooks_and_inputs_unchanged') is not True or manifest.get('model_inference_runs') != 0:
        st.error('Unexpected analysis provenance; inspect the notebook before interpreting results.')
        return
    columns = st.columns(4)
    columns[0].metric('Matched products', f"{manifest['matched_products']:,}")
    columns[1].metric('Barcode overlap', f"{manifest['integration_summary']['overlap_percent_of_part2']:.2f}%", 'of official Part 2 barcodes')
    all_coverage = data['coverage_rows']
    unique_images = data['image_rows']
    numeric_products = all_coverage.loc[all_coverage['structured_numeric'], 'barcode'].nunique()
    columns[2].metric('Mapped nutrient coverage', f"{numeric_products/manifest['matched_products']:.1%}", f'{numeric_products:,} products with ≥1 finite mapped field')
    columns[3].metric('Evaluated images', str(manifest['evaluated_images']), 'cached official test; not all matched images')
    st.caption('KPIs above describe the fixed analysed snapshot. Dedicated filters below apply to linked charts/tables, not the global integration KPIs or fixed full-background aggregates. Sidebar EDA filters do not apply on this page.')
    controls = st.columns(4)
    with controls[0]:
        nutrients = st.multiselect('Insights nutrient', sorted(data['coverage_rows']['nutrient'].unique()), key='impact_nutrient')
    with controls[1]:
        countries = st.multiselect('Insights country', sorted({value for raw in unique_images['country'] for value in labels(raw)}), key='impact_country')
    with controls[2]:
        categories = st.multiselect('Insights category', sorted({value for raw in unique_images['category'] for value in labels(raw)}), key='impact_category')
    with controls[3]:
        splits = st.multiselect('Insights official split', ['train', 'test'], key='impact_split')
    filters = dict(nutrients=nutrients, countries=countries, categories=categories, splits=splits)
    coverage = filtered_rows(data['coverage_rows'], **filters)
    perf = filtered_rows(data['performance_rows'], **filters)
    anomaly = filtered_rows(data['anomaly_rows'], **filters)
    agreement = filtered_rows(data['agreement_rows'], **filters)
    visible_images = filtered_rows(unique_images, countries=countries, categories=categories, splits=splits)
    st.caption(f"Selection: {visible_images['image_key'].nunique():,} matched images, {perf['image_key'].nunique()} evaluated images. No prediction for an image means not evaluated—not a failed extraction.")
    if manifest['prediction_audit']['rejected']:
        st.warning('Some cached samples failed alignment; excluded reasons are shown in provenance.')

    st.subheader('Representativeness · full Part 1 versus annotation-selected subset')
    representation = data['representativeness']
    if nutrients:
        representation = representation.loc[representation['nutrient'].isin(nutrients)]
    st.caption('Full-background aggregates are fixed, row-weighted snapshots. These distribution, country/category and completeness comparisons support nutrient selection only; local category/country/split filters are not applied to full-source aggregates because joint background strata were not cached. Matched and broad-50K snapshots are shown separately.')
    st.dataframe(representation, width='stretch', hide_index=True)
    bins = data['distribution_bins']
    histogram_nutrients = sorted(bins['nutrient'].unique())
    chosen = st.selectbox('Background nutrient distribution', nutrients or histogram_nutrients, key='impact_distribution')
    histogram = bins.loc[bins['nutrient'].eq(chosen)].sort_values('bin_index')
    st.plotly_chart(px.bar(histogram, x='bin_label', y='fraction', color='dataset', barmode='group', hover_data=['count', 'numeric_denominator'], title='Identical nutrient-specific bins across sources'), width='stretch')
    st.caption('Why it matters: different source distributions can change the mix of nutrient values seen by extraction demonstrations. Full bin counts are exact; full P1/P99/IQR quantiles use a seed42 uniform sample capped at 20,000 finite values per nutrient and are approximate. The complete-source scan includes 56 repeated barcode rows.')
    label_dimension = st.selectbox('Representativeness labels', ['countries', 'categories'], key='impact_labels')
    label_data = data['label_distributions'].loc[data['label_distributions']['dimension'].eq(label_dimension)]
    top = label_data.loc[label_data['dataset'].eq('Matched annotated products')].nlargest(12, 'count')['label']
    label_data = label_data.loc[label_data['label'].isin(top)]
    st.plotly_chart(px.bar(label_data, x='label', y='fraction', color='dataset', barmode='group', hover_data=['count', 'denominator']), width='stretch')
    st.caption('Why it matters: annotation-selected countries/categories may be over- or under-represented. Labels overlap, so fractions need not sum to 100%; missing labels are not reinterpreted as a country or category.')
    completeness = data['metadata_coverage']
    st.plotly_chart(px.bar(completeness, x='field', y='fraction', color='dataset', barmode='group', hover_data=['count', 'denominator']), width='stretch')
    st.caption('Why it matters: sparse serving/category/country metadata limits context-based parsing and subgroup evaluation even when the image is annotated. Completeness is not extraction accuracy.')

    st.subheader('Nutrient coverage · metadata, gold annotations and predictions')
    coverage_stats = coverage_summary(coverage)
    if not coverage_stats.empty:
        matrix = coverage_stats.pivot(index='nutrient', columns='source', values='coverage')
        st.plotly_chart(px.imshow(matrix, zmin=0, zmax=1, color_continuous_scale='Blues', aspect='auto', labels={'color': 'Fraction with field/span'}), width='stretch')
        st.dataframe(coverage_stats, width='stretch', hide_index=True)
        st.caption('Why it matters: a missing structured value can coexist with gold image text. All-matched metadata/gold use selected matched images; evaluated-cohort columns share the smaller evaluated-image denominator. Model availability is never divided by all 3,075 images. Absent gold spans do not prove the nutrient is absent from the panel.')
        states = coverage[['nutrient', 'metadata_state', 'annotation_state', 'prediction_state']].value_counts().rename('image_nutrient_pairs').reset_index()
        st.dataframe(states, width='stretch', hide_index=True)
    else:
        st.info('No coverage rows match these filters.')

    st.subheader('Extraction performance by nutrient · cached official test only')
    summary = metric_summary(perf)
    st.write(summary)
    nutrient_performance = performance_groups(perf, 'nutrient')
    if not nutrient_performance.empty:
        st.dataframe(nutrient_performance, width='stretch', hide_index=True)
        valid = nutrient_performance.loc[nutrient_performance['sufficient_support']]
        if not valid.empty:
            melted = valid.melt(id_vars=['group', 'images', 'entity_contributing_images', 'gold_spans'], value_vars=['precision', 'recall', 'f1', 'parsing_success'], var_name='measure', value_name='score')
            st.plotly_chart(px.bar(melted, x='group', y='score', color='measure', barmode='group', hover_data=['images', 'entity_contributing_images', 'gold_spans']), width='stretch')
        st.caption('Why it matters: nutrient identity/basis errors and parser failures are different bottlenecks. Scores are exact fine-grained BIO spans; parser success is parseable predicted spans / predicted spans, not value correctness. Groups require ≥5 entity-contributing images and ≥5 gold spans; sparse scores are blank while counts remain visible.')
        failures = perf.loc[perf['parsing_failures'].gt(0), ['barcode', 'nutrient', 'pred_entities', 'parsed_predictions', 'parsing_failures', 'parsing_reasons']]
        with st.expander('Actual numerical parsing failures in the selected cached cohort'):
            st.dataframe(failures, width='stretch', hide_index=True)
    else:
        st.info('No evaluated predictions for this selection. Train-only filters are expected to have no model metrics; this is not zero accuracy.')

    st.subheader('Anomaly analysis · unit-aware review, not automatic corrections')
    st.caption('Units: source energy-kcal in kcal, energy-kj in kJ, mapped mass fields in grams under the export’s _100g convention. Review bounds: negative values; mass >100 g/100 g; energy >1000 kcal or >4200 kJ/100 g. No per-100g upper bound is applied to serving quantities. Beverage per-volume/density ambiguity remains unverified.')
    if not anomaly.empty:
        st.dataframe(anomaly.groupby('nutrient').agg(numeric_products=('value', 'size'), physical_review=('physical_review', 'sum'), iqr_review=('iqr_outlier', 'sum'), p01_p99_tail_review=('tail_p01_p99', 'sum')).reset_index(), width='stretch', hide_index=True)
        anomaly_nutrient = st.selectbox('Anomaly nutrient', sorted(anomaly['nutrient'].unique()), key='impact_anomaly')
        chart = anomaly.loc[anomaly['nutrient'].eq(anomaly_nutrient)]
        st.plotly_chart(px.histogram(chart, x='value', color='physical_review', nbins=45, hover_data=['unit']), width='stretch')
        st.caption('Why it matters: extreme values may require a unit/basis or packaging review before cross-source agreement is meaningful. Histogram uses raw selected product values; flag counts and numeric support are retained.')
        st.dataframe(chart[['barcode', 'nutrient', 'value', 'unit', 'basis', 'rules', 'iqr_outlier', 'tail_p01_p99', 'evaluated']], width='stretch', hide_index=True)
        thresholds = data['anomaly_thresholds']
        st.dataframe(thresholds.loc[thresholds['nutrient'].eq(anomaly_nutrient)], width='stretch', hide_index=True)
        st.caption('IQR/P1/P99 thresholds describe the complete matched metadata cohort and are not retuned by dashboard filters or test errors. IQR/tail flags are not physical-error labels.')
    else:
        st.info('No finite mapped metadata values for this selection.')

    st.subheader('Potential error drivers · observational strata')
    dimension = st.selectbox('Error-driver grouping', ['category', 'country', 'token_count', 'aspect_ratio', 'width', 'metadata_fields_present', 'structured_numeric'], key='impact_driver')
    drivers = performance_groups(perf, dimension)
    if not drivers.empty:
        st.dataframe(drivers.sort_values('images', ascending=False), width='stretch', hide_index=True)
        supported = drivers.loc[drivers['sufficient_support']]
        if not supported.empty:
            st.plotly_chart(px.bar(supported, x='group', y='f1', hover_data=['images', 'entity_contributing_images', 'gold_spans', 'token_errors']), width='stretch')
        else:
            st.info('All selected strata have insufficient observations; no score bars are drawn.')
        st.caption('Why it matters: text length, image geometry and metadata gaps are candidate review priorities, not established causes. Groups can overlap for countries/categories and are not independent; pixel width is not a measured blur/OCR-quality score. The model uses supplied corpus OCR, so these metrics do not establish EasyOCR accuracy.')
    else:
        st.info('No error-driver scores are available for this cohort.')

    st.subheader('Cross-source agreement · recorded values are not gold')
    if not agreement.empty:
        compatible = agreement.loc[agreement['compatible']]
        st.caption(f"{len(compatible)} compatible predicted fields among {len(agreement)} parsed candidates in this selection. Predeclared tolerance: 1% relative plus 1e-8 absolute; qualified bounds are not compared as point values. Missing/serving/unit-incompatible fields remain explicitly unscored.")
        if not compatible.empty:
            agreement_nutrient = st.selectbox('Agreement nutrient (one unit scale)', sorted(compatible['nutrient'].unique()), key='impact_agreement_nutrient')
            agreement_plot = compatible.loc[compatible['nutrient'].eq(agreement_nutrient)]
            st.plotly_chart(px.scatter(agreement_plot, x='product_value', y='predicted_value', color='nutrient', symbol='within_one_percent', hover_data=['barcode', 'unit', 'basis', 'raw_text']), width='stretch')
            st.caption('Why it matters: discrepancies can reflect model/BIO/OCR issues, rounded declarations, changed packaging or database edits. Unit/basis field-name compatibility does not verify the photographed packaging; no discrepancy automatically becomes a model error. Mixed nutrient units are shown explicitly and must not be pooled into one numeric error statistic.')
        st.dataframe(agreement.sort_values('comparison_status'), width='stretch', hide_index=True)
    else:
        st.info('No parsed cached predictions for this selection; no agreement score is invented.')

    st.subheader('Key findings · EDA finding → Part 2 implication → action')
    st.caption('These five computed findings refer to the full analysed cohort, not the current interactive selection. Observations and hypotheses are explicitly separated; no causal claims.')
    st.dataframe(data['findings'].rename(columns={'eda_finding': 'EDA Finding', 'impact': 'Impact on Part 2', 'action': 'Recommended Action'}), width='stretch', hide_index=True)
    with st.expander('Provenance, rejected caches and limitations'):
        st.json({key: manifest[key] for key in ['version', 'generated_at_utc', 'background', 'prediction_audit', 'limitations', 'model_inference_runs']})
    st.download_button('Download computed findings', data['findings'].to_csv(index=False).encode('utf-8'), file_name='eda_extraction_findings.csv', mime='text/csv', key='impact_findings_download')
    st.download_button('Download filtered performance counts', perf.to_csv(index=False).encode('utf-8'), file_name='eda_extraction_filtered_performance.csv', mime='text/csv', key='impact_perf_download')
