// static/app.js
(() => {
  // helpers
  const $ = (s) => document.querySelector(s);
  const showToast = (msg, timeout=3000) => {
    const t = $('#toast');
    if(!t) return;
    t.textContent = msg; t.style.display='block';
    setTimeout(()=> t.style.display='none', timeout);
  };

  // dark mode
  const themeToggle = $('#theme-toggle');
  const applyTheme = (dark) => {
    document.body.classList.toggle('dark', !!dark);
    localStorage.setItem('mhoc_dark', !!dark ? '1':'0');
  };
  const initTheme = () => {
    const saved = localStorage.getItem('mhoc_dark');
    if(saved === null) { applyTheme(false); } else { applyTheme(saved === '1'); }
  };
  if(themeToggle){ themeToggle.addEventListener('click', ()=> applyTheme(!document.body.classList.contains('dark'))); }
  initTheme();

  // elements
  const form = $('#predict-form');
  const btn = $('#predict-btn');
  const spinner = $('#btn-spinner');
  const btnText = $('#btn-text');
  const copyBtn = $('#copy-btn');
  const resetBtn = $('#reset-btn');
  const resultsArea = $('#results-area');
  const viz = $('#viz');
  const hardnessValueEl = $('#hardness-value');
  const oxidationValueEl = $('#oxidation-value');

  // bootstrap initial server-render state
  const initial = window.__INITIAL__ || {};
  if(initial.hardness || initial.oxidation) {
    // ensure the values show in results area
    setTimeout(()=> {
      renderResults(initial.hardness, initial.oxidation, initial.hardness_error, initial.oxidation_error);
      plotResults(initial.hardness, initial.oxidation);
    }, 50);
  }

  function disableUI(disabled=true) {
    btn.disabled = disabled;
    if(disabled) {
      spinner.style.display = 'inline-block';
      btnText.textContent = 'Predicting…';
    } else {
      spinner.style.display = 'none';
      btnText.textContent = 'Predict';
    }
  }

  // build payload from form fields
  function gatherPayload() {
    return {
      Material: $('#Material').value,
      Current: $('#Current').value,
      Heat_Input: $('#Heat_Input').value,
      Soaking_Time: $('#Soaking_Time').value,
      Carbon: $('#Carbon').value,
      Manganese: $('#Manganese').value
    };
  }

  // display results into DOM
  function renderResults(hardness, oxidation, hardness_error, oxidation_error) {
    resultsArea.innerHTML = '';
    const box = document.createElement('div'); box.className='result-item';
    if(hardness !== null && hardness !== undefined) {
      const r = document.createElement('div'); r.className='result-row';
      r.innerHTML = `<strong>Predicted Hardness:</strong> <span id="hardness-value">${Number(hardness).toFixed(4)}</span>`;
      box.appendChild(r);
    }
    if(oxidation !== null && oxidation !== undefined) {
      const r2 = document.createElement('div'); r2.className='result-row';
      r2.innerHTML = `<strong>Predicted Oxidation Rate:</strong> <span id="oxidation-value">${Number(oxidation)}</span>`;
      box.appendChild(r2);
    }
    if(hardness_error) {
      const e = document.createElement('div'); e.className='error'; e.textContent = `Hardness Error: ${hardness_error}`; box.appendChild(e);
    }
    if(oxidation_error) {
      const e2 = document.createElement('div'); e2.className='error'; e2.textContent = `Oxidation Error: ${oxidation_error}`; box.appendChild(e2);
    }
    resultsArea.appendChild(box);
  }

  // Plotly small bar chart (normalized)
  function plotResults(hardness, oxidation) {
    if(!viz) return;
    // if missing values, clear
    if((hardness === null || hardness === undefined) && (oxidation === null || oxidation === undefined)) {
      viz.innerHTML = '';
      return;
    }

    // create a normalized scale for visualization (not meaningful absolute)
    const hv = hardness ? Number(hardness) : 0;
    const ov = oxidation ? Number(oxidation) : 0;

    // simple normalization (min-max within small ranges to look good)
    const hv_scaled = (hv - 300) / 100; // approx
    const ov_scaled = ov * 1000; // bring to comparable range visually

    const data = [{
      x: ['Hardness','Oxidation (x1000)'],
      y: [Math.max(0, hv_scaled), Math.max(0, ov_scaled)],
      type: 'bar',
      marker: {color: ['#1f77b4','#ff7f0e']}
    }];

    const layout = {
      margin: {l:30,r:20,t:10,b:30},
      height: 220,
      yaxis: {visible:true, title:''},
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };

    try {
      Plotly.react(viz, data, layout, {displayModeBar:false});
    } catch(e) {
      viz.innerHTML = '<div class="muted">Visualization unavailable.</div>';
    }
  }

  // copy to clipboard
  if(copyBtn) {
    copyBtn.addEventListener('click', async () => {
      const h = $('#hardness-value') ? $('#hardness-value').textContent : '';
      const o = $('#oxidation-value') ? $('#oxidation-value').textContent : '';
      const txt = `Hardness: ${h}\nOxidation Rate: ${o}`;
      try {
        await navigator.clipboard.writeText(txt);
        showToast('Results copied to clipboard');
      } catch(e) {
        showToast('Copy failed');
      }
    });
  }

  // reset button
  if(resetBtn){
    resetBtn.addEventListener('click', () => {
      form.reset();
      resultsArea.innerHTML = '<div class="muted">No predictions yet. Fill inputs and click Predict.</div>';
      viz.innerHTML = '';
    });
  }

  // async submit
  if(form){
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      // client-side basic validation
      const required = ['Material','Current','Heat_Input','Soaking_Time','Carbon','Manganese'];
      for(let id of required) {
        const el = document.getElementById(id);
        if(!el || el.value === '' || el.value === null) { el && el.focus(); showToast('Please fill all required fields'); return; }
      }

      disableUI(true);
      const payload = gatherPayload();

      try {
        const res = await fetch('/api/v1/predict', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        renderResults(data.hardness, data.oxidation, data.hardness_error, data.oxidation_error);
        plotResults(data.hardness, data.oxidation);
        showToast('Prediction complete');

      } catch(err) {
        showToast('Request failed');
        console.error(err);
      } finally {
        disableUI(false);
      }
    });
  }

  // utility: disable UI
  function disableUI(off=false) {
    btn.disabled = off;
    if(off) {
      spinner.style.display = 'inline-block'; btnText.textContent = 'Predicting…';
    } else {
      spinner.style.display = 'none'; btnText.textContent = 'Predict';
    }
  }

  // init toast DOM if missing
  if(!$('#toast')) {
    const t = document.createElement('div'); t.id='toast'; t.className='toast'; t.style.display='none'; document.body.appendChild(t);
  }
})();
