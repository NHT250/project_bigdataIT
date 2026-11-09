const thread = document.getElementById("thread");
const form = document.getElementById("ask");
const input = document.getElementById("question");
const askButton = form.querySelector("button");

const INLINE = /(\*\*[^*]+\*\*|\[\s*\d+(?:\s*,\s*\d+)*\s*\])/;

let archiveSize = null;

function citation(group, sources) {
  const numbers = group.slice(1, -1).split(",").map((n) => Number(n.trim()));
  const parts = ["["];
  numbers.forEach((number, index) => {
    if (index) parts.push(", ");
    const source = sources.find((candidate) => candidate.number === number);
    parts.push(source ? externalLink(source.url, { className: "cite", textContent: number, title: source.title }) : String(number));
  });
  parts.push("]");
  return parts;
}

function inline(text, sources) {
  return text.split(INLINE).filter(Boolean).flatMap((part) => {
    if (part.startsWith("**")) return [el("strong", {}, ...inline(part.slice(2, -2), sources))];
    if (/^\[\s*\d/.test(part)) return citation(part, sources);
    return [part];
  });
}

function answerNode(text, sources) {
  const body = el("div", { className: "answer" });
  let bullets = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) { bullets = null; continue; }
    const bullet = line.match(/^[*-]\s+(.*)$/);
    if (bullet) {
      if (!bullets) { bullets = el("ul"); body.append(bullets); }
      bullets.append(el("li", {}, ...inline(bullet[1], sources)));
    } else {
      bullets = null;
      body.append(el("p", {}, ...inline(line, sources)));
    }
  }
  return body;
}

function sourceList(sources) {
  if (!sources.length) return null;
  return el("ol", { className: "sources" }, ...sources.map((source) => {
    const item = el("li", {},
      `[${source.number}] `,
      externalLink(source.url, { textContent: source.title }),
      source.summary ? el("p", { textContent: source.summary }) : null,
    );
    item.style.setProperty("--source", `var(--${source.origin})`);
    return item;
  }));
}

function plural(count) {
  return `${count.toLocaleString("en-US")} article${count === 1 ? "" : "s"}`;
}

async function loadArchiveSize() {
  try {
    archiveSize = (await fetchJson("/api/stats")).indexed_articles;
    document.getElementById("reach").textContent =
      `Answers come only from the ${plural(archiveSize)} in the archive, with a numbered link to each source.`;
  } catch {
    archiveSize = null;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  const reach = el("p", {
    className: "reach",
    textContent: archiveSize === null ? "Searching the archive" : `Searching ${plural(archiveSize)}`,
  });
  reach.setAttribute("role", "status");
  const turn = el("section", { className: "turn" }, el("p", { className: "question", textContent: question }), reach);
  thread.append(turn);
  turn.scrollIntoView({ block: "nearest" });
  askButton.disabled = true;
  input.value = "";

  try {
    const result = await fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    reach.textContent = `Searched ${plural(result.searched_articles)}`;
    turn.append(answerNode(result.text, result.sources));
    const sources = sourceList(result.sources);
    if (sources) turn.append(sources);
  } catch (error) {
    reach.classList.add("is-error");
    reach.textContent = error.message;
  } finally {
    askButton.disabled = false;
    input.focus();
    turn.scrollIntoView({ block: "nearest" });
  }
});

loadArchiveSize();
