"""Lightweight display of the checkpointed 3,075-image BIO inference run."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


@st.fragment(run_every='30s')
def render(root):
    directory = Path(root) / 'outputs/full_cohort_bio'
    st.subheader('Complete matched dataset · 3,075-image BIO inference')
    path = directory / 'summary.json'
    if not path.exists():
        st.info('No full-cohort checkpoint yet. The separate CPU worker publishes results after each image.')
        return
    summary = json.loads(path.read_text(encoding='utf-8'))
    done, target = summary['completed_images'], summary['target_images']
    columns = st.columns(4)
    columns[0].metric('Full-cohort completed', f'{done:,}', f'of {target:,} matched images')
    columns[1].metric('Full-cohort pending', f"{summary['pending_images']:,}")
    columns[2].metric('Full-cohort failures', str(summary['failed_images']))
    columns[3].metric('Full-cohort status', summary['status'])
    st.progress(done / target, text=f'{done:,}/{target:,} saved · updated {summary["updated_at_utc"]}')
    if done != target:
        st.warning('Partial run: scores below describe completed images only, not the complete 3,075-image cohort. This panel refreshes every 30 seconds; existing checkpoints survive interruptions.')
    st.caption('2,876 matched train + 199 official test images; eight unmatched images excluded. Original supplied OCR and normalized boxes, CPU-only existing checkpoint, overlapping windows, no retraining or OCR run. Train scores are descriptive—not held-out generalization. The original seeded 20-image evaluation remains separately below.')
    metrics = pd.DataFrame([dict(split=split, **values) for split, values in summary['metrics_by_split'].items()])
    # Do not present an unprocessed cohort as zero model accuracy.
    metrics.loc[metrics['images'].eq(0), ['precision','recall','f1','token_accuracy']] = float('nan')
    st.dataframe(metrics, hide_index=True, width='stretch')
    results_path = directory / 'image_results.csv'
    if done and results_path.exists():
        try:
            rows = pd.read_csv(results_path, dtype={'barcode':'string'})
        except pd.errors.EmptyDataError:
            st.info('First result is being published.'); return
        split = st.selectbox('Full-cohort result split', ['All','test','train'], key='full_bio_split')
        selected = rows if split == 'All' else rows.loc[rows['split'].eq(split)]
        st.dataframe(selected, hide_index=True, width='stretch')
        st.download_button('Download full-cohort image results', rows.to_csv(index=False).encode(), file_name='full_cohort_bio_image_results.csv', key='full_bio_csv')
        if not selected.empty:
            key = st.selectbox('Full-cohort image result', selected['image_key'].tolist(), key='full_bio_image')
            sample = json.loads((directory/'samples'/f"{key.replace(':','_')}.json").read_text(encoding='utf-8'))
            prediction = sample['prediction']
            st.write(dict(image_key=key, barcode=sample['barcode'], split=sample['split'], token_coverage=sample['summary_row']['token_coverage']))
            st.dataframe(pd.DataFrame(dict(token_index=range(len(sample['tokens'])), ocr_token=sample['tokens'], gold_bio=prediction['gold_labels'], predicted_bio=prediction['predicted_labels'])), hide_index=True, width='stretch')
            structured = prediction['structured_output']
            st.write('Extracted nutrient fields')
            st.dataframe(pd.DataFrame(structured.get('parsed_fields', [])), hide_index=True, width='stretch')
            with st.expander('Full-cohort structured JSON and uncertain fields'):
                st.json(structured)
            st.download_button('Download selected full-cohort prediction', json.dumps(sample,ensure_ascii=False).encode(), file_name=f"{key.replace(':','_')}_bio.json", mime='application/json', key='full_bio_json')
    if summary['failed_images']:
        with st.expander('Full-cohort failures · not scored as zero accuracy'):
            st.json(json.loads((directory/'failures.json').read_text(encoding='utf-8')))
    st.divider()