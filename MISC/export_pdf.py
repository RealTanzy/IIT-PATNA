from playwright.sync_api import sync_playwright
import os

HTML_PATH = "/home/dibyanayan/tanzeel/weekly_update_dr_asif_ekbal.html"
PDF_PATH  = "/home/dibyanayan/tanzeel/weekly_update_dr_asif_ekbal.pdf"

INJECT_JS = """
() => {
  // ── Remove carousel layout ──
  const body    = document.body;
  const shell   = document.querySelector('.shell');
  const vp      = document.querySelector('.viewport');
  const track   = document.querySelector('.track');
  const nav     = document.querySelector('.controls');
  const slides  = [...document.querySelectorAll('.slide')];

  body.style.cssText   = 'margin:0;padding:0;background:#0d0f1a;height:auto;overflow:auto;display:block;';
  shell.style.cssText  = 'width:100%;height:auto;border-radius:0;box-shadow:none;border:none;display:block;overflow:visible;';
  vp.style.cssText     = 'overflow:visible;height:auto;';
  track.style.cssText  = 'display:block;height:auto;transform:none;transition:none;';
  nav.style.display    = 'none';

  slides.forEach((s, i) => {
    s.style.cssText = [
      'min-width:100%',
      'width:100%',
      'height:100vh',
      'box-sizing:border-box',
      'page-break-after:always',
      'break-after:page',
      'position:relative',
      'overflow:hidden',
    ].join(';') + ';';
    s.classList.add('active');           // trigger entry animations immediately
  });
}
"""

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 720})

    page.goto(f"file://{HTML_PATH}", wait_until="networkidle")

    # Let Google Fonts load (up to 3s)
    try:
        page.wait_for_load_state("networkidle", timeout=3000)
    except Exception:
        pass

    # Apply print layout
    page.evaluate(INJECT_JS)
    page.wait_for_timeout(600)   # allow CSS transitions to settle

    page.pdf(
        path=PDF_PATH,
        width="1280px",
        height="720px",
        print_background=True,
        margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
    )

    browser.close()

size_kb = os.path.getsize(PDF_PATH) // 1024
print(f"PDF saved: {PDF_PATH}  ({size_kb} KB)")
