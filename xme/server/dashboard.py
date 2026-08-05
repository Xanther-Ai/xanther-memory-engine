"""XME Dashboard — FastAPI app on port 8001.

Single-file SPA: no npm, no build step, vis.js + Tailwind via CDN.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML (self-contained SPA)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>XME Memory Dashboard</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet"/>
<style>
  #graph-container { height: 500px; border: 1px solid #e2e8f0; border-radius: 8px; }
  .node-decision { background: #ebf8ff; }
  .node-attempt  { background: #fff5f5; }
  .node-pref     { background: #f0fff4; }
</style>
</head>
<body class="bg-gray-50 text-gray-800">
<div class="max-w-7xl mx-auto p-6">
  <div class="flex items-center justify-between mb-6">
    <h1 class="text-2xl font-bold text-indigo-700">⚡ XME Memory Dashboard</h1>
    <select id="project-select" class="border rounded px-3 py-1 text-sm" onchange="loadProject()">
      <option value="">Select project...</option>
    </select>
  </div>

  <!-- Stats bar -->
  <div id="stats-bar" class="grid grid-cols-4 gap-4 mb-6 hidden">
    <div class="bg-white rounded-lg p-4 shadow-sm text-center">
      <div class="text-3xl font-bold text-indigo-600" id="stat-facts">0</div>
      <div class="text-sm text-gray-500">Facts</div>
    </div>
    <div class="bg-white rounded-lg p-4 shadow-sm text-center">
      <div class="text-3xl font-bold text-green-600" id="stat-episodes">0</div>
      <div class="text-sm text-gray-500">Sessions</div>
    </div>
    <div class="bg-white rounded-lg p-4 shadow-sm text-center">
      <div class="text-3xl font-bold text-blue-600" id="stat-users">0</div>
      <div class="text-sm text-gray-500">Users</div>
    </div>
    <div class="bg-white rounded-lg p-4 shadow-sm text-center">
      <div class="text-sm font-medium text-gray-700" id="stat-last">—</div>
      <div class="text-sm text-gray-500">Last activity</div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="flex gap-2 mb-4">
    <button onclick="showTab('graph')"    id="tab-graph"    class="tab-btn active-tab px-4 py-2 rounded text-sm">Graph</button>
    <button onclick="showTab('timeline')" id="tab-timeline" class="tab-btn px-4 py-2 rounded text-sm bg-white">Timeline</button>
    <button onclick="showTab('search')"   id="tab-search"   class="tab-btn px-4 py-2 rounded text-sm bg-white">Search</button>
    <button onclick="showTab('context')"  id="tab-context"  class="tab-btn px-4 py-2 rounded text-sm bg-white">Context</button>
  </div>

  <!-- Graph tab -->
  <div id="tab-content-graph" class="tab-content bg-white rounded-lg p-4 shadow-sm">
    <div id="graph-container"></div>
    <div id="graph-legend" class="flex gap-4 mt-3 text-xs text-gray-500">
      <span class="flex items-center gap-1"><span class="w-3 h-3 rounded-full bg-blue-400 inline-block"></span> Decision</span>
      <span class="flex items-center gap-1"><span class="w-3 h-3 rounded-full bg-red-400 inline-block"></span> Attempt</span>
      <span class="flex items-center gap-1"><span class="w-3 h-3 rounded-full bg-green-400 inline-block"></span> Preference</span>
      <span class="flex items-center gap-1"><span class="w-3 h-3 rounded-full bg-yellow-400 inline-block"></span> Convention</span>
      <span class="flex items-center gap-1"><span class="w-3 h-3 rounded-full bg-gray-400 inline-block"></span> Entity</span>
    </div>
  </div>

  <!-- Timeline tab -->
  <div id="tab-content-timeline" class="tab-content hidden bg-white rounded-lg p-4 shadow-sm">
    <div id="timeline-list" class="space-y-3"></div>
  </div>

  <!-- Search tab -->
  <div id="tab-content-search" class="tab-content hidden bg-white rounded-lg p-4 shadow-sm">
    <div class="flex gap-2 mb-4">
      <input id="search-input" type="text" placeholder="Search memory..."
             class="flex-1 border rounded px-3 py-2 text-sm"
             onkeydown="if(event.key==='Enter') runSearch()"/>
      <button onclick="runSearch()" class="bg-indigo-600 text-white px-4 py-2 rounded text-sm">Search</button>
    </div>
    <div id="search-results" class="space-y-2"></div>
  </div>

  <!-- Context tab -->
  <div id="tab-content-context" class="tab-content hidden bg-white rounded-lg p-4 shadow-sm">
    <div id="context-list" class="space-y-4"></div>
  </div>
</div>

<script>
const COLOR = {decision:'#60a5fa',attempt:'#f87171',preference:'#34d399',convention:'#fbbf24',entity:'#9ca3af'};
let currentProject = '';
let network = null;

async function api(path) {
  const r = await fetch('/api' + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

window.onload = async () => {
  try {
    const projects = await api('/projects');
    const sel = document.getElementById('project-select');
    projects.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.project_id; opt.textContent = p.project_id;
      sel.appendChild(opt);
    });
    if (projects.length === 1) { sel.value = projects[0].project_id; loadProject(); }
  } catch(e) { console.error(e); }
};

async function loadProject() {
  currentProject = document.getElementById('project-select').value;
  if (!currentProject) return;
  document.getElementById('stats-bar').classList.remove('hidden');
  await Promise.all([loadStats(), loadGraph(), loadTimeline(), loadContexts()]);
}

async function loadStats() {
  const s = await api(`/${currentProject}/stats`);
  document.getElementById('stat-facts').textContent = s.total_facts;
  document.getElementById('stat-episodes').textContent = s.total_episodes;
  document.getElementById('stat-users').textContent = s.active_users;
  document.getElementById('stat-last').textContent = s.last_activity ? s.last_activity.slice(0,10) : '—';
}

async function loadGraph() {
  const g = await api(`/${currentProject}/graph`);
  const nodes = new vis.DataSet(g.nodes.map(n => ({
    id: n.id, label: n.label.slice(0,25),
    color: { background: COLOR[n.group] || '#9ca3af', border: '#374151' },
    title: n.title, font: { size: 11 }
  })));
  const edges = new vis.DataSet(g.edges.map(e => ({
    from: e.from, to: e.to, label: e.label,
    arrows: 'to', color: { color: '#6b7280' }, font: { size: 9 }
  })));
  const container = document.getElementById('graph-container');
  if (network) network.destroy();
  network = new vis.Network(container, { nodes, edges }, {
    physics: { stabilization: { iterations: 100 } },
    interaction: { hover: true },
  });
}

async function loadTimeline() {
  const data = await api(`/${currentProject}/timeline?limit=30`);
  const el = document.getElementById('timeline-list');
  el.innerHTML = data.episodes.map(ep => `
    <div class="border-l-4 border-indigo-300 pl-4 py-1">
      <div class="text-sm font-medium">${ep.started_at ? ep.started_at.slice(0,10) : ''} — ${ep.outcome || ''}</div>
      <div class="text-xs text-gray-500">${ep.user_id || ''}</div>
      <div class="text-sm text-gray-700 mt-1">${ep.summary || '(no summary)'}</div>
    </div>`).join('');
}

async function loadContexts() {
  const data = await api(`/${currentProject}/context`);
  const el = document.getElementById('context-list');
  el.innerHTML = data.contexts.map(c => `
    <div class="border rounded p-3">
      <div class="font-medium text-sm mb-1">👤 ${c.user_id}</div>
      <div class="text-sm"><b>Task:</b> ${c.current_task || '—'}</div>
      <div class="text-sm"><b>Next:</b> ${c.next_steps || '—'}</div>
      ${c.recent_decisions.length ? '<div class="text-xs text-gray-500 mt-1">Decisions: ' + c.recent_decisions.slice(0,3).join(', ') + '</div>' : ''}
    </div>`).join('');
}

async function runSearch() {
  if (!currentProject) return;
  const q = document.getElementById('search-input').value;
  if (!q) return;
  const r = await api(`/${currentProject}/search?q=${encodeURIComponent(q)}`);
  const el = document.getElementById('search-results');
  const all = [...(r.facts||[]), ...(r.episodic||[]), ...(r.context||[])];
  el.innerHTML = all.map(item => `
    <div class="border rounded p-3">
      <span class="text-xs font-medium px-2 py-0.5 rounded bg-gray-100 mr-2">${item.layer}</span>
      <span class="text-sm font-medium">${item.summary || ''}</span>
      ${item.highlight ? '<div class="text-xs text-gray-500 mt-1">' + item.highlight + '</div>' : ''}
    </div>`).join('') || '<p class="text-gray-400 text-sm">No results</p>';
}

function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.getElementById('tab-content-' + name).classList.remove('hidden');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('bg-indigo-600','text-white','active-tab'));
  document.getElementById('tab-' + name).classList.add('bg-indigo-600','text-white','active-tab');
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_dashboard_app() -> FastAPI:
    app = FastAPI(title="XME Memory Dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return HTMLResponse(_DASHBOARD_HTML)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return HTMLResponse(_DASHBOARD_HTML)

    @app.get("/api/projects")
    async def list_projects():
        from xme.engine import get_engine
        engine = await get_engine()
        # List all projects known from context store
        assert engine.context._conn is not None
        rows = engine.context._conn.execute(
            "SELECT DISTINCT project_id FROM working_context ORDER BY project_id"
        ).fetchall()
        projects = [{"project_id": r["project_id"]} for r in rows]
        # Also check facts store
        assert engine.facts._conn is not None
        rows2 = engine.facts._conn.execute(
            "SELECT DISTINCT project_id FROM xme_facts ORDER BY project_id"
        ).fetchall()
        seen = {p["project_id"] for p in projects}
        for r in rows2:
            if r["project_id"] not in seen:
                projects.append({"project_id": r["project_id"]})
        return projects

    @app.get("/api/{project_id}/stats")
    async def project_stats(project_id: str):
        from xme.engine import get_engine
        engine = await get_engine()
        return await engine.stats(project_id)

    @app.get("/api/{project_id}/graph")
    async def project_graph(project_id: str):
        from xme.engine import get_engine
        engine = await get_engine()
        return await engine.facts.get_graph_data(project_id)

    @app.get("/api/{project_id}/timeline")
    async def project_timeline(project_id: str, limit: int = 30):
        from xme.engine import get_engine
        engine = await get_engine()
        episodes = await engine.episodic.list_episodes(project_id, limit=limit)
        return {"episodes": [
            {
                "episode_id": ep.episode_id,
                "user_id": ep.user_id,
                "started_at": ep.started_at,
                "ended_at": ep.ended_at,
                "summary": ep.summary,
                "outcome": ep.outcome,
                "turn_count": ep.turn_count,
            }
            for ep in episodes
        ]}

    @app.get("/api/{project_id}/context")
    async def project_context(project_id: str):
        from xme.engine import get_engine
        engine = await get_engine()
        users = engine.context.list_users(project_id)
        contexts = []
        for uid in users:
            ctx = engine.context.get(project_id, uid)
            if ctx:
                contexts.append(ctx.to_dict())
        return {"contexts": contexts}

    @app.get("/api/{project_id}/search")
    async def project_search(project_id: str, q: str = "", limit: int = 10):
        from xme.engine import get_engine
        engine = await get_engine()
        if not q:
            return {"facts": [], "episodic": [], "context": []}
        results = await engine.search(q, project_id, limit=limit)
        return {
            "facts": [{"layer": r.layer, "id": r.item_id, "summary": r.summary,
                       "score": r.score, "highlight": r.highlight}
                      for r in results.facts],
            "episodic": [{"layer": r.layer, "id": r.item_id, "summary": r.summary,
                          "score": r.score, "highlight": r.highlight}
                         for r in results.episodic],
            "context": [{"layer": r.layer, "id": r.item_id, "summary": r.summary,
                         "score": r.score}
                        for r in results.context],
        }

    @app.post("/api/{project_id}/export")
    async def project_export(project_id: str, request: Request):
        body = await request.json()
        fmt = body.get("format", "obsidian")
        from xme.engine import get_engine
        from xme.export import run_export
        engine = await get_engine()
        path = await run_export(engine, project_id, fmt=fmt)
        return {"status": "ok", "output_path": str(path)}

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "xme-dashboard"}

    return app
