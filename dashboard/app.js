// Live anchor feed for agora-perp-agent.
//
// Reads our agent wallet's transactions on Arc testnet from Blockscout's
// REST API, filters for ERC-8183 `createJob` calls to the commerce proxy,
// decodes the on-chain `description` field, and renders the table.
//
// Zero build step, zero dependencies — just static fetch() + manual ABI
// decoding for the one function we care about. Auto-refreshes every
// 30 s.

const WALLET = "0x5b09fc1d4a519b4591217fb5d06247159b9fcf44";
const ERC8183_PROXY = "0x0747EEf0706327138c69792bF28Cd525089e4583";

// Function selector for `createJob(address,address,uint256,string,address)`
// — precomputed off-chain. Anything else from this wallet to the proxy is
// a different ERC-8183 method (fund / submit / complete / reject) and is
// not shown here at MVP.
const SELECTOR_CREATE_JOB = "0x41528812";

const BLOCKSCOUT_BASE = "https://testnet.arcscan.app";
const REFRESH_MS = 30_000;
const SNAPSHOT_REFRESH_MS = 10_000;

// On deployed GitHub Pages, the bundled `data/snapshot.json` lags by the
// next Pages rebuild (~60s after the bot pushes). We bypass that by
// fetching the file from raw.githubusercontent.com directly, which
// reflects the latest commit on main with only a short CDN cache
// (defeated by the cache-buster query param).
//
// On localhost (dev preview via `python -m http.server -d dashboard`),
// the relative path serves the on-disk file the bot is writing in
// real-time, which is even faster — no GitHub round trip at all.
const IS_LOCAL =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";
const SNAPSHOT_PATH = IS_LOCAL
  ? "data/snapshot.json"
  : "https://raw.githubusercontent.com/greetingromansoldier/agora-perp-agent/main/dashboard/data/snapshot.json";

// ---------------------------------------------------------------- fetch

async function fetchTransactions() {
  const url = `${BLOCKSCOUT_BASE}/api/v2/addresses/${WALLET}/transactions?filter=from`;
  const resp = await fetch(url, { headers: { accept: "application/json" } });
  if (!resp.ok) {
    throw new Error(`blockscout ${resp.status}`);
  }
  const body = await resp.json();
  return body.items || [];
}

// ---------------------------------------------------------------- decode

/**
 * Pull the `description` string out of a `createJob(...)` calldata blob.
 * Returns null if the input doesn't match the expected layout.
 *
 * createJob ABI layout (after the 4-byte selector):
 *   [  0:32]  provider     (address, left-padded)
 *   [ 32:64]  evaluator    (address)
 *   [ 64:96]  expiredAt    (uint256)
 *   [ 96:128] descr offset (uint256, bytes from start of params)
 *   [128:160] hook         (address)
 *   ... at offset:
 *   [off : off+32]   length
 *   [off+32 : off+32+len]  utf-8 bytes
 */
function decodeCreateJobDescription(rawInput) {
  if (!rawInput || typeof rawInput !== "string") return null;
  if (!rawInput.toLowerCase().startsWith(SELECTOR_CREATE_JOB)) return null;
  const hex = rawInput.slice(10); // strip 0x + 4-byte selector
  if (hex.length < 320) return null;

  const offsetBytes = parseInt(hex.slice(192, 256), 16);
  const cursor = offsetBytes * 2;
  if (hex.length < cursor + 64) return null;
  const len = parseInt(hex.slice(cursor, cursor + 64), 16);
  if (len <= 0 || hex.length < cursor + 64 + len * 2) return null;

  const dataHex = hex.slice(cursor + 64, cursor + 64 + len * 2);
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = parseInt(dataHex.slice(i * 2, i * 2 + 2), 16);
  }
  return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
}

/**
 * Parse the on-chain description string. Three formats supported:
 *
 *   v1 (smoke-test legacy):
 *     "agora:<audit_id_36>:0x<hash_64>"
 *
 *   v2 (intermediate pipe-separated):
 *     "agora|<short_id>|<ASSET>|<L|S>|...|hash=0x<hash_64>"
 *
 *   v3 (current readable sentence):
 *     "agora-perp-agent · ASSET side qty @ lev x · $X notional ·
 *      stop $Y take $Z · tier T · regime A/B/C · audit <id> ·
 *      keccak 0x<hash_64>"
 *
 * Returns `{auditId, anchorHash, summary?}` or null on no match.
 */
function parseDescription(desc) {
  if (!desc || !/^agora/i.test(desc)) return null;

  // Pull the 64-char hex anchor hash out, regardless of label.
  const hashMatch = desc.match(/0x[0-9a-f]{64}/i);
  const anchorHash = hashMatch ? hashMatch[0] : null;

  // Try each known format's audit-id extractor.
  let auditId = null;
  // v1: 36-char uuid after `agora:`
  const v1 = desc.match(/^agora:([0-9a-f-]{36}):/i);
  if (v1) {
    auditId = v1[1];
  } else {
    // v2: 8-char hex after `agora|`
    const v2 = desc.match(/^agora\|([0-9a-f]{8})\|/i);
    if (v2) {
      auditId = v2[1];
    } else {
      // v3: 8-char hex after the word "audit"
      const v3 = desc.match(/\baudit\s+([0-9a-f]{8})\b/i);
      if (v3) auditId = v3[1];
    }
  }

  // Build a compact summary from the middle of v2 / v3 descriptions.
  let summary = null;
  if (desc.startsWith("agora|")) {
    const parts = desc.split("|");
    summary = parts.slice(2, -1).join(" · ");
  } else if (/^agora-perp-agent\s+·\s+/i.test(desc)) {
    // Strip leading "agora-perp-agent · " and trailing keccak/hash token.
    const stripped = desc.replace(/^agora-perp-agent\s+·\s+/, "");
    const cut = stripped.search(/\s+·\s+(?:keccak|hash)\s+/i);
    summary = cut > 0 ? stripped.slice(0, cut) : stripped;
  }

  return { auditId, anchorHash, summary };
}

// ---------------------------------------------------------------- format

function formatTimestamp(iso) {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  const now = Date.now();
  const diff = (now - d.getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function abbreviate(s, head = 6, tail = 4) {
  if (!s) return "—";
  if (s.length <= head + tail + 1) return s;
  return `${s.slice(0, head)}…${s.slice(-tail)}`;
}

// ---------------------------------------------------------------- render

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function renderHeader() {
  document.getElementById("wallet-link").href =
    `https://testnet.arcscan.app/address/${WALLET}`;
  document.getElementById("contract-link").href =
    `https://testnet.arcscan.app/address/${ERC8183_PROXY}`;
  const walletStat = document.getElementById("stat-wallet");
  walletStat.textContent = abbreviate(WALLET, 8, 6);
  walletStat.href = `https://testnet.arcscan.app/address/${WALLET}`;
}

function renderError(message) {
  const tbody = document.querySelector("#feed tbody");
  tbody.innerHTML = `
    <tr class="error">
      <td colspan="4">${message}</td>
    </tr>`;
}

function renderEmpty() {
  const tbody = document.querySelector("#feed tbody");
  tbody.innerHTML = `
    <tr class="placeholder">
      <td colspan="4">no anchors yet — agent has not run on Arc.</td>
    </tr>`;
}

function renderRows(rows) {
  if (rows.length === 0) {
    renderEmpty();
    return;
  }
  const tbody = document.querySelector("#feed tbody");
  tbody.innerHTML = rows
    .map((r) => {
      const txUrl = `https://testnet.arcscan.app/tx/${r.txHash}`;
      const auditLabel = r.auditId
        ? (r.auditId.length > 12 ? abbreviate(r.auditId, 8, 4) : r.auditId)
        : "";
      const audit = r.auditId
        ? `<code title="${r.auditId}">${auditLabel}</code>${r.summary ? `<div class="row-sub">${escapeHtml(r.summary)}</div>` : ""}`
        : `<span class="muted">—</span>`;
      const hash = r.anchorHash
        ? `<span title="${r.anchorHash}">${abbreviate(r.anchorHash, 10, 6)}</span>`
        : `<span class="muted">—</span>`;
      return `
        <tr>
          <td class="when" title="${r.timestamp}">${formatTimestamp(r.timestamp)}</td>
          <td class="audit">${audit}</td>
          <td class="hash">${hash}</td>
          <td><a href="${txUrl}" target="_blank" rel="noopener">${abbreviate(r.txHash, 8, 6)} ↗</a></td>
        </tr>`;
    })
    .join("");
}

function renderStats(rows) {
  setText("stat-anchors", rows.length.toString());
  const cutoff = Date.now() - 24 * 3600 * 1000;
  const last24h = rows.filter((r) => new Date(r.timestamp).getTime() >= cutoff).length;
  setText("stat-24h", last24h.toString());
  setText("stat-updated", new Date().toLocaleTimeString());
}

// ---------------------------------------------------------------- pipeline

async function refresh() {
  try {
    const txs = await fetchTransactions();
    const rows = txs
      .filter((tx) => {
        const to = (tx.to?.hash || tx.to)?.toLowerCase?.();
        return to === ERC8183_PROXY.toLowerCase();
      })
      .map((tx) => {
        const desc = decodeCreateJobDescription(tx.raw_input || tx.input);
        const parsed = parseDescription(desc);
        return {
          txHash: tx.hash,
          timestamp: tx.timestamp,
          description: desc,
          auditId: parsed?.auditId || null,
          anchorHash: parsed?.anchorHash || null,
          summary: parsed?.summary || null,
        };
      })
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    renderRows(rows);
    renderStats(rows);
  } catch (err) {
    console.error(err);
    renderError(`failed to load anchors: ${err.message}`);
    setText("stat-updated", "error");
  }
}

// ---------------------------------------------------------------- snapshot

/**
 * Fetch the local dashboard snapshot.json (written by the running
 * agent's `agent/snapshot.py`). Returns the parsed payload or null if
 * the file is missing (e.g. the agent has never run on this checkout).
 */
async function fetchSnapshot() {
  try {
    const resp = await fetch(`${SNAPSHOT_PATH}?_=${Date.now()}`, {
      cache: "no-store",
      headers: { accept: "application/json" },
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch (_) {
    return null;
  }
}

function fmtUsd(value, sign = false) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const s = Math.abs(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  if (!sign) return `$${s}`;
  return value >= 0 ? `+$${s}` : `−$${s}`;
}

function fmtNum(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function renderEquityChart(curve, startingBalance) {
  const svg = document.getElementById("equity-chart");
  if (!svg) return;
  svg.innerHTML = "";
  if (!Array.isArray(curve) || curve.length < 2) {
    document.getElementById("equity-meta-left").textContent =
      "(equity curve unavailable — agent has not run yet)";
    document.getElementById("equity-meta-right").textContent = "";
    return;
  }
  const w = 800;
  const h = 200;
  const pad = 12;
  const values = curve.map((p) => p.equity_usd);
  const minV = Math.min(...values, startingBalance);
  const maxV = Math.max(...values, startingBalance);
  const range = Math.max(maxV - minV, 1e-6);
  const stepX = (w - pad * 2) / (curve.length - 1);
  const sy = (v) => h - pad - ((v - minV) / range) * (h - pad * 2);

  // starting-balance reference line
  const refY = sy(startingBalance);
  const refLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  refLine.setAttribute("x1", pad);
  refLine.setAttribute("y1", refY);
  refLine.setAttribute("x2", w - pad);
  refLine.setAttribute("y2", refY);
  refLine.setAttribute("stroke", "#3a3f4a");
  refLine.setAttribute("stroke-dasharray", "4 4");
  svg.appendChild(refLine);

  // gradient under the curve
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `<linearGradient id="equity-grad" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="rgba(125,209,129,0.4)"/>
    <stop offset="100%" stop-color="rgba(125,209,129,0)"/>
  </linearGradient>`;
  svg.appendChild(defs);

  const points = curve
    .map((p, i) => `${pad + stepX * i},${sy(p.equity_usd)}`)
    .join(" ");
  const fill = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  fill.setAttribute(
    "points",
    `${pad},${h - pad} ${points} ${w - pad},${h - pad}`
  );
  fill.setAttribute("fill", "url(#equity-grad)");
  svg.appendChild(fill);

  const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  path.setAttribute("points", points);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "#7dd181");
  path.setAttribute("stroke-width", "2");
  svg.appendChild(path);

  const firstIso = curve[0].timestamp_iso;
  const lastIso = curve[curve.length - 1].timestamp_iso;
  const span = (new Date(lastIso).getTime() - new Date(firstIso).getTime()) / 1000;
  const spanStr =
    span < 60 ? `${Math.round(span)}s`
      : span < 3600 ? `${Math.round(span / 60)}m`
        : `${(span / 3600).toFixed(1)}h`;
  document.getElementById("equity-meta-left").textContent =
    `${curve.length} samples · span ${spanStr}`;
  document.getElementById("equity-meta-right").textContent =
    `${formatTimestamp(firstIso)} → now`;
}

function renderPositions(positions) {
  const tbody = document.querySelector("#positions tbody");
  if (!tbody) return;
  if (!Array.isArray(positions) || positions.length === 0) {
    tbody.innerHTML = `<tr class="placeholder"><td colspan="10">(no open positions)</td></tr>`;
    return;
  }
  tbody.innerHTML = positions
    .map((p) => {
      const pnlClass = p.unrealized_pnl_usd >= 0 ? "pnl-pos" : "pnl-neg";
      const tx = p.arc_tx_hash
        ? `<a href="${p.arcscan_url || `https://testnet.arcscan.app/tx/${p.arc_tx_hash}`}" target="_blank">${abbreviate(p.arc_tx_hash, 6, 4)} ↗</a>`
        : (p.anchor_state === "pending" ? `<span class="muted">…</span>` : `<span class="muted">—</span>`);
      const regime = p.regime ?? "—";
      const lev = p.leverage != null ? `${p.leverage.toFixed(1)}×` : "—";
      return `<tr>
        <td><strong>${p.asset}</strong></td>
        <td class="${p.side === 'long' ? 'side-long' : 'side-short'}">${p.side.toUpperCase()}</td>
        <td>${lev}</td>
        <td>${fmtNum(p.qty, 6)}</td>
        <td>${fmtNum(p.mark_price, 2)}</td>
        <td>${p.stop_price != null ? fmtNum(p.stop_price, 2) : "—"}</td>
        <td>${p.take_price != null ? fmtNum(p.take_price, 2) : "—"}</td>
        <td class="${pnlClass}">${fmtUsd(p.unrealized_pnl_usd, true)}</td>
        <td><code>${regime}</code></td>
        <td>${tx}</td>
      </tr>`;
    })
    .join("");
}

function renderDecisions(decisions) {
  const root = document.getElementById("decisions");
  if (!root) return;
  if (!Array.isArray(decisions) || decisions.length === 0) {
    root.innerHTML = `<div class="placeholder muted">(no decisions yet)</div>`;
    return;
  }
  root.innerHTML = decisions
    .slice(0, 20)
    .map((d) => {
      const txLink = d.arc_tx_hash
        ? `<a href="${d.arcscan_url || `https://testnet.arcscan.app/tx/${d.arc_tx_hash}`}" target="_blank" class="tx-link">arcscan ${abbreviate(d.arc_tx_hash, 6, 4)} ↗</a>`
        : (d.anchor_state === "pending"
          ? `<span class="anchor-pending">anchoring…</span>`
          : d.anchor_state === "off"
            ? `<span class="muted">no anchor</span>`
            : "");
      const sizeBits = [];
      if (d.tier) sizeBits.push(d.tier);
      if (d.regime) sizeBits.push(d.regime);
      if (d.leverage != null) sizeBits.push(`${d.leverage.toFixed(1)}× lev`);
      if (d.notional_usd != null) sizeBits.push(`${fmtUsd(d.notional_usd)} notional`);
      if (d.stop_price != null) sizeBits.push(`stop ${fmtNum(d.stop_price, 2)}`);
      if (d.take_price != null) sizeBits.push(`take ${fmtNum(d.take_price, 2)}`);

      return `<div class="decision">
        <div class="decision-head">
          <span class="verdict verdict-${(d.verdict || "EXECUTE").toLowerCase()}">${d.verdict || "EXECUTE"}</span>
          <strong>${d.asset}</strong>
          <span class="side-${d.side || ""}">${(d.side || "").toUpperCase()}</span>
          <span class="muted">${d.decision_id || ""}</span>
          <span class="muted">${formatTimestamp(d.decided_at_iso)}</span>
          ${txLink}
        </div>
        ${sizeBits.length ? `<div class="decision-size">${sizeBits.map(b => `<code>${b}</code>`).join(" · ")}</div>` : ""}
        ${d.reasoning ? `<div class="decision-reasoning">${escapeHtml(d.reasoning)}</div>` : ""}
      </div>`;
    })
    .join("");
}

function renderHistory(history) {
  const tbody = document.querySelector("#history tbody");
  if (!tbody) return;
  if (!Array.isArray(history) || history.length === 0) {
    tbody.innerHTML = `<tr class="placeholder"><td colspan="8">(no closed trades yet)</td></tr>`;
    return;
  }
  tbody.innerHTML = history
    .slice(0, 50)
    .map((t) => {
      const pnlClass = t.realized_pnl_usd >= 0 ? "pnl-pos" : "pnl-neg";
      const regime = t.regime || "—";
      return `<tr>
        <td class="when">${formatTimestamp(t.exit_time_iso)}</td>
        <td><strong>${t.asset}</strong></td>
        <td class="${t.side === 'long' ? 'side-long' : 'side-short'}">${(t.side || "").toUpperCase()}</td>
        <td>${fmtNum(t.qty, 6)}</td>
        <td>${fmtNum(t.entry_price, 2)}</td>
        <td>${fmtNum(t.exit_price, 2)}</td>
        <td class="${pnlClass}">${fmtUsd(t.realized_pnl_usd, true)}</td>
        <td><code>${regime}</code></td>
      </tr>`;
    })
    .join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function refreshSnapshot() {
  const snap = await fetchSnapshot();
  if (!snap) {
    setText("stat-equity", "—");
    setText("stat-realized", "—");
    setText("stat-open", "—");
    return;
  }
  setText("stat-equity", fmtUsd(snap.equity_usd));
  setText(
    "stat-realized",
    fmtUsd(snap.realized_pnl_usd, true)
  );
  const realizedEl = document.getElementById("stat-realized");
  if (realizedEl) {
    realizedEl.className = "value " + (snap.realized_pnl_usd >= 0 ? "pnl-pos" : "pnl-neg");
  }
  setText("stat-open", (snap.open_positions || []).length.toString());

  // Snapshot freshness — overrides the generic "last update" stat so
  // viewers see "bot's data is X minutes old" rather than just when the
  // browser last refreshed.
  if (snap.generated_at_iso) {
    setText("stat-updated", formatTimestamp(snap.generated_at_iso));
  }

  renderEquityChart(snap.equity_curve, snap.starting_balance_usd);
  renderPositions(snap.open_positions);
  renderDecisions(snap.recent_decisions);
  renderHistory(snap.trade_history);
}

// ---------------------------------------------------------------- boot

renderHeader();
refresh();
refreshSnapshot();
setInterval(refresh, REFRESH_MS);
setInterval(refreshSnapshot, SNAPSHOT_REFRESH_MS);
