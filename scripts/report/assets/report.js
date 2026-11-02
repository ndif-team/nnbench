// Report charts: horizontal grouped bars, dumbbells and column strips in plain SVG.
// Every chart gets a legend (2+ series), a hover/focus tooltip per row, and a table twin.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs || {}) {
      if (k === "style") Object.assign(n.style, attrs[k]);
      else n.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(n);
    return n;
  }
  function htmlEl(tag, cls, parent, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  function svgText(parent, x, y, text, attrs) {
    const t = svgEl("text", Object.assign({ x, y }, attrs || {}), parent);
    t.textContent = text;
    return t;
  }

  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    return (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  }
  function ticks(min, max, count) {
    if (max === min) max = min + 1;
    const step = niceStep(max - min, count);
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return { values: out, step };
  }

  // Bar path from the baseline: square at the baseline, 4px rounded data end.
  function barPath(x0, x1, y, h) {
    const w = Math.abs(x1 - x0);
    const r = Math.min(4, w / 2, h / 2);
    if (x1 >= x0) {
      return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
    }
    return `M${x0},${y}H${x1 + r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 + r},${y + h}H${x0}Z`;
  }
  function colBarPath(x, y0, y1, w) {
    const h = Math.abs(y1 - y0);
    const r = Math.min(4, w / 2, h / 2);
    return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
  }

  function tooltip(plot) {
    const tip = htmlEl("div", "tip", plot);
    tip.hidden = true;
    return {
      show(title, rows, px, py) {
        tip.replaceChildren();
        htmlEl("div", "h", tip, title);
        for (const r of rows) {
          const row = htmlEl("div", "r", tip);
          const key = htmlEl("i", null, row);
          key.style.background = r.color || "var(--ink-2)";
          htmlEl("b", null, row, r.value);
          htmlEl("span", null, row, r.label);
        }
        tip.hidden = false;
        const pw = plot.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
        let x = px + 14, y = py - th / 2;
        if (x + tw > pw) x = Math.max(0, px - tw - 14);
        tip.style.left = `${x}px`;
        tip.style.top = `${Math.max(0, y)}px`;
      },
      hide() { tip.hidden = true; },
    };
  }

  function legend(fig, series) {
    const box = fig.querySelector(".legend");
    if (!box || series.length < 2) return;
    box.replaceChildren();
    for (const s of series) {
      const item = htmlEl("span", null, box);
      const sw = htmlEl("i", s.line ? "line" : null, item);
      sw.style.background = s.color;
      item.appendChild(document.createTextNode(s.label));
    }
  }

  function tableTwin(fig, head, rows) {
    let tools = fig.querySelector(".chart-tools");
    if (!tools) return;
    let wrap = fig.querySelector(".twin");
    if (!wrap) {
      const btn = htmlEl("button", null, tools, "Show table");
      btn.type = "button";
      wrap = htmlEl("div", "table-wrap twin", fig);
      wrap.hidden = true;
      btn.addEventListener("click", () => {
        wrap.hidden = !wrap.hidden;
        btn.textContent = wrap.hidden ? "Show table" : "Hide table";
      });
    }
    wrap.replaceChildren();
    const t = htmlEl("table", null, wrap);
    const tr = htmlEl("tr", null, htmlEl("thead", null, t));
    head.forEach((h, i) => htmlEl("th", i ? "num" : null, tr, h));
    const tb = htmlEl("tbody", null, t);
    for (const r of rows) {
      const row = htmlEl("tr", null, tb);
      r.forEach((c, i) => htmlEl("td", i ? "num" : null, row, c));
    }
  }

  // ---- horizontal grouped bars --------------------------------------------------------------
  function hbars(plot, spec) {
    const W = Math.max(300, plot.clientWidth);
    const nS = spec.series.length;
    const barH = nS > 1 ? 11 : 15, gap = 2, pad = 9;
    const labelW = Math.min(spec.labelWidth || 220, Math.round(W * 0.4));
    const right = 62, axisH = 24;
    const bandH = nS * barH + (nS - 1) * gap + pad * 2;
    const H = spec.categories.length * bandH + axisH + 4;
    const vals = [];
    spec.categories.forEach((c, ci) => spec.series.forEach((s, si) => {
      const v = spec.value(ci, si);
      if (v != null && isFinite(v)) vals.push(v);
    }));
    let dmin = Math.min(0, ...(spec.domain ? [spec.domain[0]] : []), ...vals);
    let dmax = Math.max(0, ...(spec.domain ? [spec.domain[1]] : []), ...vals);
    if (spec.refLine) { dmin = Math.min(dmin, spec.refLine.value); dmax = Math.max(dmax, spec.refLine.value); }
    const tk = ticks(dmin, dmax, Math.max(3, Math.floor((W - labelW - right) / 90)));
    dmin = Math.min(dmin, tk.values[0]);
    dmax = Math.max(dmax, tk.values[tk.values.length - 1]);
    const x = (v) => labelW + ((v - dmin) / (dmax - dmin || 1)) * (W - labelW - right);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.aria || "" });
    const grid = svgEl("g", {}, svg);
    for (const t of tk.values) {
      svgEl("line", { x1: x(t), x2: x(t), y1: 0, y2: H - axisH, style: { stroke: t === 0 ? "var(--rule-strong)" : "var(--rule)", strokeWidth: 1 } }, grid);
      svgText(grid, x(t), H - axisH + 15, spec.tick ? spec.tick(t) : String(t), { "text-anchor": "middle", style: { fill: "var(--muted)", fontSize: "11px", fontVariantNumeric: "tabular-nums" } });
    }
    if (spec.refLine) {
      svgEl("line", { x1: x(spec.refLine.value), x2: x(spec.refLine.value), y1: 0, y2: H - axisH, style: { stroke: "var(--ink-2)", strokeWidth: 1 } }, grid);
    }
    const tip = tooltip(plot);
    spec.categories.forEach((c, ci) => {
      const y0 = ci * bandH;
      const g = svgEl("g", { tabindex: 0, role: "listitem" }, svg);
      const lbl = svgText(g, labelW - 10, y0 + bandH / 2 + (c.sub ? -2 : 4), c.label, { "text-anchor": "end", style: { fill: "var(--ink)", fontSize: "12px", fontWeight: 600 } });
      if (c.sub) svgText(g, labelW - 10, y0 + bandH / 2 + 12, c.sub, { "text-anchor": "end", style: { fill: "var(--muted)", fontSize: "11px" } });
      const rows = [];
      spec.series.forEach((s, si) => {
        const v = spec.value(ci, si);
        const by = y0 + pad + si * (barH + gap);
        const txt = v == null || !isFinite(v) ? (spec.missing ? spec.missing(ci, si) : "n/a") : spec.fmt(v);
        rows.push({ color: s.color, value: txt, label: s.label });
        if (v == null || !isFinite(v)) {
          svgText(g, x(0) + 6, by + barH - 2, txt, { style: { fill: "var(--muted)", fontSize: "11px", fontStyle: "italic" } });
          return;
        }
        svgEl("path", { d: barPath(x(0), x(v), by, barH), style: { fill: s.color } }, g);
        if (spec.labelTips !== false) {
          const tx = v >= 0 ? x(v) + 5 : x(v) - 5;
          svgText(g, tx, by + barH - 2, txt, { "text-anchor": v >= 0 ? "start" : "end", style: { fill: "var(--ink-2)", fontSize: "11px", fontVariantNumeric: "tabular-nums" } });
        }
      });
      g.setAttribute("aria-label", `${c.label}: ` + rows.map((r) => `${r.label} ${r.value}`).join(", "));
      const hit = svgEl("rect", { x: 0, y: y0, width: W, height: bandH, style: { fill: "transparent" } }, g);
      const showAt = (px, py) => tip.show(c.label + (c.sub ? ` · ${c.sub}` : ""), rows, px, py);
      hit.addEventListener("pointermove", (e) => {
        const r = plot.getBoundingClientRect();
        showAt(e.clientX - r.left, e.clientY - r.top);
      });
      hit.addEventListener("pointerleave", () => tip.hide());
      g.addEventListener("focus", () => showAt(Math.min(plot.clientWidth - 40, x(Math.max(0, dmax)) * plot.clientWidth / W), (y0 + bandH / 2) * plot.clientWidth / W));
      g.addEventListener("blur", () => tip.hide());
      void lbl;
    });
    svg.setAttribute("role", "list");
    plot.replaceChildren(svg, plot.querySelector(".tip") || document.createElement("span"));
    const fig = plot.closest("figure");
    legend(fig, spec.series);
    tableTwin(fig, [spec.catHead || "", ...spec.series.map((s) => s.label)],
      spec.categories.map((c, ci) => [c.label + (c.sub ? ` (${c.sub})` : ""), ...spec.series.map((s, si) => {
        const v = spec.value(ci, si);
        return v == null || !isFinite(v) ? (spec.missing ? spec.missing(ci, si) : "n/a") : spec.fmt(v);
      })]));
    // keep the tooltip element (replaceChildren above re-attached it or a placeholder)
    if (!plot.querySelector(".tip")) tooltip(plot);
  }

  // ---- dumbbell: before -> after per row -----------------------------------------------------
  function dumbbell(plot, spec) {
    const W = Math.max(300, plot.clientWidth);
    const labelW = Math.min(spec.labelWidth || 230, Math.round(W * 0.42));
    const right = 64, rowH = 30, axisH = 24;
    const H = spec.categories.length * rowH + axisH + 6;
    const vals = [];
    spec.categories.forEach((_, i) => { for (const v of [spec.a.value(i), spec.b.value(i)]) if (v != null) vals.push(v); });
    let dmax = Math.max(...vals) * 1.04;
    const tk = ticks(0, dmax, Math.max(3, Math.floor((W - labelW - right) / 90)));
    dmax = Math.max(dmax, tk.values[tk.values.length - 1]);
    const x = (v) => labelW + (v / dmax) * (W - labelW - right);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "list" });
    for (const t of tk.values) {
      svgEl("line", { x1: x(t), x2: x(t), y1: 0, y2: H - axisH, style: { stroke: t === 0 ? "var(--rule-strong)" : "var(--rule)", strokeWidth: 1 } }, svg);
      svgText(svg, x(t), H - axisH + 15, spec.tick ? spec.tick(t) : String(t), { "text-anchor": "middle", style: { fill: "var(--muted)", fontSize: "11px", fontVariantNumeric: "tabular-nums" } });
    }
    const tip = tooltip(plot);
    spec.categories.forEach((c, i) => {
      const cy = i * rowH + rowH / 2;
      const g = svgEl("g", { tabindex: 0, role: "listitem" }, svg);
      svgText(g, labelW - 10, cy + 4, c.label, { "text-anchor": "end", style: { fill: "var(--ink)", fontSize: "12px", fontWeight: 600 } });
      const va = spec.a.value(i), vb = spec.b.value(i);
      if (va != null && vb != null) {
        svgEl("line", { x1: x(va), x2: x(vb), y1: cy, y2: cy, style: { stroke: "var(--rule-strong)", strokeWidth: 2, strokeLinecap: "round" } }, g);
      }
      for (const [s, v] of [[spec.a, va], [spec.b, vb]]) {
        if (v == null) continue;
        svgEl("circle", { cx: x(v), cy, r: 5, style: { fill: s.color, stroke: "var(--surface)", strokeWidth: 2 } }, g);
      }
      const delta = spec.delta ? spec.delta(i) : null;
      if (delta) svgText(g, W - right + 8, cy + 4, delta, { style: { fill: "var(--ink-2)", fontSize: "11px", fontVariantNumeric: "tabular-nums" } });
      const rows = [
        { color: spec.a.color, value: va == null ? "n/a" : spec.fmt(va), label: spec.a.label },
        { color: spec.b.color, value: vb == null ? "n/a" : spec.fmt(vb), label: spec.b.label },
      ];
      if (delta) rows.push({ color: "transparent", value: delta, label: "change" });
      g.setAttribute("aria-label", `${c.label}: ` + rows.map((r) => `${r.label} ${r.value}`).join(", "));
      const hit = svgEl("rect", { x: 0, y: i * rowH, width: W, height: rowH, style: { fill: "transparent" } }, g);
      hit.addEventListener("pointermove", (e) => {
        const r = plot.getBoundingClientRect();
        tip.show(c.label, rows, e.clientX - r.left, e.clientY - r.top);
      });
      hit.addEventListener("pointerleave", () => tip.hide());
      g.addEventListener("focus", () => tip.show(c.label, rows, x(Math.max(va || 0, vb || 0)) * plot.clientWidth / W, cy * plot.clientWidth / W));
      g.addEventListener("blur", () => tip.hide());
    });
    plot.replaceChildren(svg, plot.querySelector(".tip"));
    const fig = plot.closest("figure");
    legend(fig, [spec.a, spec.b]);
    tableTwin(fig, [spec.catHead || "", spec.a.label, spec.b.label, "change"],
      spec.categories.map((c, i) => [c.label, spec.a.value(i) == null ? "n/a" : spec.fmt(spec.a.value(i)),
        spec.b.value(i) == null ? "n/a" : spec.fmt(spec.b.value(i)), spec.delta ? spec.delta(i) || "" : ""]));
  }

  // ---- column strip: one value per category, emphasis by a flag --------------------------------
  function columns(plot, spec) {
    const W = Math.max(260, plot.clientWidth);
    const left = 34, right = 8, top = 10, axisH = 34, plotH = spec.height || 120;
    const H = top + plotH + axisH;
    const n = spec.categories.length;
    const slot = (W - left - right) / n;
    const bw = Math.min(24, Math.max(4, slot - 4));
    const ymax = spec.yMax || 1;
    const y = (v) => top + plotH - (v / ymax) * plotH;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "list" });
    for (const t of spec.yTicks || [0, ymax / 2, ymax]) {
      svgEl("line", { x1: left, x2: W - right, y1: y(t), y2: y(t), style: { stroke: t === 0 ? "var(--rule-strong)" : "var(--rule)", strokeWidth: 1 } }, svg);
      svgText(svg, left - 6, y(t) + 4, spec.tick ? spec.tick(t) : String(t), { "text-anchor": "end", style: { fill: "var(--muted)", fontSize: "11px", fontVariantNumeric: "tabular-nums" } });
    }
    const tip = tooltip(plot);
    spec.categories.forEach((c, i) => {
      const v = spec.values[i];
      const cx = left + slot * i + (slot - bw) / 2;
      const g = svgEl("g", { tabindex: 0, role: "listitem" }, svg);
      const color = spec.emph[i] ? spec.emphColor : spec.restColor;
      if (v > 0) svgEl("path", { d: colBarPath(cx, y(0), y(Math.min(v, ymax)), bw), style: { fill: color } }, g);
      else svgEl("rect", { x: cx, y: y(0) - 1.5, width: bw, height: 1.5, style: { fill: color } }, g);
      svgText(g, cx + bw / 2, top + plotH + 14, c.label, { "text-anchor": "middle", style: { fill: "var(--ink-2)", fontSize: "11px" } });
      if (c.sub) svgText(g, cx + bw / 2, top + plotH + 27, c.sub, { "text-anchor": "middle", style: { fill: "var(--muted)", fontSize: "10px", fontVariantNumeric: "tabular-nums" } });
      const rows = [{ color, value: spec.fmt(v), label: spec.emph[i] ? spec.emphLabel : spec.restLabel }];
      g.setAttribute("aria-label", `${spec.catName || ""} ${c.label}: ${spec.fmt(v)}`);
      const hit = svgEl("rect", { x: left + slot * i, y: top, width: slot, height: plotH + axisH, style: { fill: "transparent" } }, g);
      const title = `${spec.catName || ""} ${c.label}${c.sub ? ` · ${c.subName || ""}${c.sub}` : ""}`;
      hit.addEventListener("pointermove", (e) => {
        const r = plot.getBoundingClientRect();
        tip.show(title, rows, e.clientX - r.left, e.clientY - r.top);
      });
      hit.addEventListener("pointerleave", () => tip.hide());
      g.addEventListener("focus", () => tip.show(title, rows, (cx + bw) * plot.clientWidth / W, y(Math.min(v, ymax)) * plot.clientWidth / W));
      g.addEventListener("blur", () => tip.hide());
    });
    plot.replaceChildren(svg, plot.querySelector(".tip"));
    const fig = plot.closest("figure");
    legend(fig, [{ label: spec.emphLabel, color: spec.emphColor }, { label: spec.restLabel, color: spec.restColor }]);
    tableTwin(fig, [spec.catName || "", spec.subName || "", spec.valueName || "value"],
      spec.categories.map((c, i) => [c.label, c.sub || "", spec.fmt(spec.values[i])]));
  }

  const KINDS = { hbars, dumbbell, columns };
  function mount(id, kind, spec) {
    const plot = document.getElementById(id);
    if (!plot) return;
    let last = 0;
    const draw = () => {
      const w = plot.clientWidth;
      if (Math.abs(w - last) < 2) return;
      last = w;
      KINDS[kind](plot, spec);
    };
    draw();
    if (window.ResizeObserver) new ResizeObserver(() => requestAnimationFrame(draw)).observe(plot);
  }
  window.Report = { mount };
})();
