const thread = document.getElementById("thread");
const welcome = document.getElementById("welcome");
const form = document.getElementById("ask");
const input = document.getElementById("question");
const askButton = form.querySelector("button");
const newChatButton = document.getElementById("new-chat");

const INLINE = /(\*\*[^*]+\*\*|\[\s*\d+(?:\s*,\s*\d+)*\s*\])/;
// The server only reads the last few turns, so sending more is wasted bandwidth.
const HISTORY_TURNS = 3;
// What the page keeps and restores, so stored history can't grow without limit.
const SAVED_TURNS = 20;
const STORAGE_KEY = "chat";

let archiveSize = null;
let turns = []; // { question, result } for every answered question, oldest first
let inFlight = null; // AbortController of the question being answered

// Saved history

function isSource(source) {
  return source && Number.isInteger(source.number) && typeof source.title === "string"
    && typeof source.url === "string" && Object.hasOwn(ORIGIN_NAMES, source.origin)
    && (source.summary === null || typeof source.summary === "string");
}

function isTurn(turn) {
  return turn && typeof turn.question === "string" && turn.result
    && typeof turn.result.text === "string" && Number.isInteger(turn.result.searched_articles)
    && Array.isArray(turn.result.sources) && turn.result.sources.every(isSource)
    && (turn.result.suggestions === undefined
      || (Array.isArray(turn.result.suggestions) && turn.result.suggestions.every((q) => typeof q === "string")));
}

function serverHistory() {
  return turns.slice(-HISTORY_TURNS).map(({ question, result }) => ({
    question,
    answer: result.text,
    sources: result.sources.map((source) => ({ number: source.number, title: source.title })),
  }));
}

function remember(question, result) {
  turns.push({ question, result });
  turns = turns.slice(-SAVED_TURNS);
  save(STORAGE_KEY, turns);
}

function rememberSuggestions(questions) {
  const last = turns[turns.length - 1];
  if (!last) return;
  last.result.suggestions = questions;
  save(STORAGE_KEY, turns);
}

// Follow-up questions: under the latest answer only, each one sent as the next turn when clicked.
// The server keeps only questions the archive answers, so a click doesn't lead to "no information".

function clearFollowUps() {
  for (const node of thread.querySelectorAll(".follow-ups")) node.remove();
}

function followUps(questions) {
  const list = el("ul", {}, ...questions.map((question) => {
    const button = el("button", { type: "button", textContent: question });
    button.addEventListener("click", () => {
      if (!inFlight) ask(question);
    });
    return el("li", {}, button);
  }));
  return el("section", { className: "follow-ups", ariaLabel: "Follow-up questions" },
    el("p", { className: "follow-ups-label", textContent: "Ask next" }), list);
}

function followUpsPending() {
  const note = el("p", { className: "follow-ups-label is-pending", textContent: "Finding follow-up questions…" });
  return el("section", { className: "follow-ups" }, note);
}

// Answer text: paragraphs, bullets, bold and [n] citations, built as DOM nodes

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
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return [el("strong", {}, ...inline(part.slice(2, -2), sources))];
    }
    if (/^\[\s*\d/.test(part)) return citation(part, sources);
    return [part];
  });
}

function fillAnswer(body, text, sources) {
  body.replaceChildren();
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
}

function sourceList(sources) {
  if (!sources.length) return null;
  const list = el("ol", { className: "sources" }, ...sources.map((source) => el("li", {},
    el("span", { className: "number", textContent: `[${source.number}]` }),
    el("div", {},
      externalLink(source.url, { textContent: source.title }),
      " ",
      el("span", { className: `pill pill-${source.origin}`, textContent: ORIGIN_NAMES[source.origin] }),
      source.summary ? el("p", { textContent: source.summary }) : null,
    ),
  )));
  const count = `${sources.length} source${sources.length === 1 ? "" : "s"}`;
  return el("details", { className: "source-block", open: true }, el("summary", { textContent: count }), list);
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

// Messages: the question on the right, the answer on the left

function typingDots() {
  const dots = el("span", { className: "typing" }, el("span"), el("span"), el("span"));
  dots.setAttribute("role", "img");
  dots.setAttribute("aria-label", "Writing the answer");
  return dots;
}

function messagePair(question) {
  const asked = el("div", { className: "message from-user" }, el("p", { className: "bubble", textContent: question }));
  const searching = el("p", { className: "searching" });
  const body = el("div", { className: "answer" });
  const reply = el("div", { className: "message from-archive" }, searching, el("div", { className: "bubble" }, body));
  thread.append(asked, reply);
  return { reply, searching, body };
}

function searchingLine(node, question, searchedFor, done, searchedArticles) {
  const rewritten = searchedFor && searchedFor !== question;
  const count = searchedArticles ?? archiveSize;
  const scope = count === null || count === undefined ? "the archive" : plural(count);
  node.replaceChildren(done ? `Searched ${scope}` : `Searching ${scope}`);
  if (rewritten) node.append(el("span", { className: "searched-for", textContent: ` for “${searchedFor}”` }));
}

function showResult(view, question, result) {
  searchingLine(view.searching, question, result.searched_for, true, result.searched_articles);
  fillAnswer(view.body, result.text, result.sources);
  view.reply.classList.toggle("is-empty", result.status !== "answered");
  view.reply.querySelector(".source-block")?.remove();
  const sources = sourceList(result.sources);
  if (sources) view.reply.querySelector(".bubble").append(sources);
}

// Scrolling: follow the newest message unless the reader scrolled up to read

function nearBottom() {
  return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
}

function scrollToEnd(force) {
  if (force || nearBottom()) window.scrollTo({ top: document.documentElement.scrollHeight });
}

function showWelcome() {
  welcome.hidden = turns.length > 0 || thread.childElementCount > 0;
}

function restore() {
  const saved = loadSaved(STORAGE_KEY, []);
  turns = Array.isArray(saved) ? saved.filter(isTurn).slice(-SAVED_TURNS) : [];
  for (const { question, result } of turns) showResult(messagePair(question), question, result);
  const last = turns[turns.length - 1];
  if (last?.result.suggestions?.length) thread.append(followUps(last.result.suggestions));
  showWelcome();
  scrollToEnd(true);
}

// Server-sent events from /api/chat/stream, one JSON object per "data:" line

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of block.split("\n")) {
        if (line.startsWith("data: ")) yield JSON.parse(line.slice(6));
      }
    }
    if (done) return;
  }
}

async function ask(question) {
  clearFollowUps();
  const view = messagePair(question);
  view.body.append(typingDots());
  searchingLine(view.searching, question, null, false);
  showWelcome();
  scrollToEnd(true);

  inFlight = new AbortController();
  const controller = inFlight;
  askButton.disabled = true;
  let text = "";
  let finished = false;
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: serverHistory() }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `The server returned ${response.status}.`);
    }
    for await (const event of readEvents(response)) {
      // Decided before the message grows, so a long piece doesn't count as the reader scrolling away.
      const follow = nearBottom();
      if (event.type === "searching") {
        // A second search starts the answer over, so anything shown so far goes.
        text = "";
        view.body.replaceChildren(typingDots());
        searchingLine(view.searching, question, event.searched_for, false);
      } else if (event.type === "restart") {
        // Another LLM took over midway and writes the answer again from the start.
        text = "";
        view.body.replaceChildren(typingDots());
      } else if (event.type === "delta") {
        text += event.text;
        fillAnswer(view.body, text, []);
      } else if (event.type === "done") {
        showResult(view, question, event.answer);
        remember(question, event.answer);
        finished = true;
        // The answer is complete: the reader can ask again while follow-ups are found.
        inFlight = null;
        askButton.disabled = false;
        if (event.answer.status === "answered") thread.append(followUpsPending());
      } else if (event.type === "suggestions") {
        if (thread.lastElementChild?.classList.contains("follow-ups") && thread.lastElementChild.previousElementSibling === view.reply) {
          thread.lastElementChild.replaceWith(followUps(event.questions));
          rememberSuggestions(event.questions);
        }
      } else if (event.type === "error") {
        throw new Error(event.message);
      }
      scrollToEnd(follow);
    }
    if (!finished) throw new Error("The answer stopped before it was finished. Try again.");
    // No follow-ups came (none the archive could answer): drop the waiting note.
    const pending = thread.querySelector(".follow-ups .is-pending");
    if (pending && pending.closest(".follow-ups").previousElementSibling === view.reply) pending.closest(".follow-ups").remove();
  } catch (error) {
    if (controller.signal.aborted) return;
    if (finished) {
      // The answer arrived; only the follow-ups failed, so they are simply left out.
      if (thread.lastElementChild?.classList.contains("follow-ups")
        && thread.lastElementChild.previousElementSibling === view.reply) thread.lastElementChild.remove();
      return;
    }
    const follow = nearBottom();
    view.body.replaceChildren(el("p", { className: "notice", textContent: error.message }));
    view.reply.classList.add("is-empty");
    const retry = el("button", { type: "button", className: "retry", textContent: "Try again" });
    retry.addEventListener("click", () => {
      if (inFlight) return;
      view.reply.previousElementSibling?.remove();
      view.reply.remove();
      ask(question);
    });
    view.body.append(retry);
    scrollToEnd(follow);
  } finally {
    if (inFlight === controller) {
      inFlight = null;
      askButton.disabled = false;
      input.focus({ preventScroll: true });
    }
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || inFlight) return;
  input.value = "";
  ask(question);
});

for (const button of welcome.querySelectorAll(".suggestions button")) {
  button.addEventListener("click", () => {
    if (!inFlight) ask(button.textContent);
  });
}

newChatButton.addEventListener("click", () => {
  inFlight?.abort();
  inFlight = null;
  askButton.disabled = false;
  turns = [];
  forget(STORAGE_KEY);
  thread.replaceChildren();
  showWelcome();
  input.value = "";
  input.focus();
  window.scrollTo({ top: 0 });
});

restore();
loadArchiveSize();
