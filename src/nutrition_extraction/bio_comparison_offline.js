/* BIO_COMPARISON_START */
// Reuse the captured UI controller; only the new image-pair event is handled here.
const renderWithoutBioOverlays = render;
render = function(events, parent) {
  let pending = [];
  const flush = () => { if (pending.length) renderWithoutBioOverlays(pending, parent); pending = []; };
  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    if (event.kind !== 'bio_overlay') { pending.push(event); continue; }
    flush();
    const grid = node('div', parent);
    grid.className = 'bio-overlay-grid';
    grid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:20px;align-items:start';
    for (let panel = 0; panel < 2 && events[i]?.kind === 'bio_overlay'; panel++, i++) {
      const e = events[i], figure = node('figure', grid);
      figure.style.margin = '0';
      node('figcaption', figure, e.caption);
      const img = node('img', figure);
      img.src = e.uri;
      img.alt = e.caption + ' on supplied original OCR boxes; green exact label, amber mismatch';
      img.style.cssText = 'width:100%;height:auto;display:block;margin-top:8px';
      const provenance = node('details', figure);
      node('summary', provenance, 'Overlay PNG · SHA-256');
      node('pre', provenance, JSON.stringify({overlay_sha256:e.sha256, original_sha256:e.original_sha256,
        original_file:e.original_file, source_size:e.source_size}, null, 2));
    }
    i--;
  }
  flush();
};
/* BIO_COMPARISON_END */