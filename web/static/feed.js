const POLL_MS = 10000;
const CHART_LIBRARY_URLS = [
  "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js",
  "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js",
];
const list = document.getElementById("feed");
const statusLine = document.getElementById("status");
const filterButtons = [...document.querySelectorAll(".filter button")];

const STORAGE_KEY = "feed-source";

let source = "all";
let seen = null;
let latestRequest = 0;
let chart = null;
let latestDays = null;

function renderFeed(articles) {
  const keys = articles.map(articleKey);
  const nodes = articles.map((article) => {
    const node = articleNode(article);
    if (seen && !seen.has(articleKey(article))) node.classList.add("is-new");
    return node;
  });
  list.replaceChildren(...nodes);
  list.removeAttribute("aria-busy");
  seen = new Set(keys);
  // Fade the highlight back to the card colour once the new rows are on screen.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    setTimeout(() => list.querySelectorAll(".is-new").forEach((node) => node.classList.remove("is-new")), 3000);
  }));
}

// How many articles each filter holds, so the With content count shows how far
// the collector has got with Hacker News story bodies.
function renderCounts(counts) {
  if (!counts) return;
  filterButtons.forEach((button) => {
    const count = counts[button.dataset.source];
    button.querySelector(".count").textContent = Number.isInteger(count) ? `(${count.toLocaleString("en-US")})` : "";
  });
}

function showLoading() {
  list.setAttribute("aria-busy", "true");
  list.replaceChildren(...skeletonCards(4));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function dayLabel(day) {
  return new Date(`${day}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

function renderTable(days) {
  document.getElementById("pulse-rows").replaceChildren(...days.map((day) => el("tr", {},
    el("td", { textContent: dayLabel(day.day) }),
    el("td", { textContent: day.arxiv }),
    el("td", { textContent: day.hn }),
    el("td", { textContent: day.devto ?? 0 }),
  )));
}

function series(label, values, color, page) {
  return {
    label, data: values, backgroundColor: color, borderColor: page,
    borderWidth: { top: 2 }, borderRadius: 4, borderSkipped: "bottom", barPercentage: 0.7, categoryPercentage: 0.9,
  };
}

function loadChartLibrary(urls) {
  if (!urls.length) {
    // Without the library the table carries the same numbers.
    document.querySelector(".pulse-chart").hidden = true;
    document.querySelector(".pulse-table").open = true;
    return;
  }
  const script = el("script", { src: urls[0] });
  script.onload = () => { if (latestDays) renderChart(latestDays); };
  script.onerror = () => loadChartLibrary(urls.slice(1));
  document.head.append(script);
}

function renderChart(days) {
  latestDays = days;
  if (!window.Chart) return;
  const labels = days.map((day) => dayLabel(day.day));
  const arxiv = days.map((day) => day.arxiv);
  const hn = days.map((day) => day.hn);
  const devto = days.map((day) => day.devto ?? 0);
  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = arxiv;
    chart.data.datasets[1].data = hn;
    chart.data.datasets[2].data = devto;
    chart.update("none");
    return;
  }
  const page = cssVar("--page");
  const soft = cssVar("--ink-soft");
  chart = new Chart(document.getElementById("pulse"), {
    type: "bar",
    data: { labels, datasets: [series("arXiv", arxiv, cssVar("--arxiv"), page), series("Hacker News", hn, cssVar("--hn"), page),
      series("Dev.to", devto, cssVar("--devto"), page)] },
    options: {
      animation: false,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { align: "start", labels: { color: soft, boxWidth: 12, boxHeight: 12 } } },
      scales: {
        x: { stacked: true, grid: { display: false }, border: { display: false }, ticks: { color: soft, maxRotation: 0, autoSkipPadding: 16 } },
        y: { stacked: true, beginAtZero: true, grid: { color: cssVar("--line") }, border: { display: false }, ticks: { color: soft, maxTicksLimit: 4, precision: 0 } },
      },
    },
  });
}

async function refresh() {
  const request = ++latestRequest;
  try {
    const [feed, stats] = await Promise.all([
      fetchJson(`/api/feed?source=${source}`),
      fetchJson("/api/stats"),
    ]);
    if (request !== latestRequest) return; // A newer filter choice has taken over.
    renderFeed(feed.articles);
    renderCounts(stats.counts);
    renderTable(stats.days);
    renderChart(stats.days);
    statusLine.classList.remove("is-error");
    statusLine.textContent = `${stats.total_articles.toLocaleString("en-US")} articles collected. Updated ${new Date().toLocaleTimeString("en-GB")}`;
  } catch {
    if (request !== latestRequest) return;
    if (list.hasAttribute("aria-busy")) {
      list.replaceChildren(); // Placeholders would suggest articles are on the way.
      list.removeAttribute("aria-busy");
    }
    statusLine.classList.add("is-error");
    statusLine.textContent = "Can't reach the server. Trying again in 10 seconds.";
  }
}

function selectSource(choice) {
  source = choice;
  filterButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.source === choice)));
}

filterButtons.forEach((button) => button.addEventListener("click", () => {
  selectSource(button.dataset.source);
  save(STORAGE_KEY, source);
  seen = null; // A different list isn't "new" articles.
  showLoading();
  refresh();
}));

// Chart.js paints with the colours it was given, so a theme switch redraws it.
document.addEventListener("themechange", () => {
  if (!chart) return;
  chart.destroy();
  chart = null;
  if (latestDays) renderChart(latestDays);
});

const savedSource = loadSaved(STORAGE_KEY, "all");
if (filterButtons.some((button) => button.dataset.source === savedSource)) selectSource(savedSource);

loadChartLibrary(CHART_LIBRARY_URLS);
showLoading();
refresh();
setInterval(() => { if (!document.hidden) refresh(); }, POLL_MS);
