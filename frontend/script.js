// ---- Config ----
// Backend ready hone par USE_MOCK = false karo aur API_URL set karo.
const API_URL = "http://localhost:8000";
const USE_MOCK = true;

const copy = {
  flight: ["Know the fare before you book.", "Pick a route and date. Our model estimates the ticket price."],
  gender: ["Predict gender from user details.", "Enter a name and age to see what the classifier returns."],
  hotel:  ["Find a hotel that fits your budget.", "Choose a city and a nightly budget to get ranked picks."]
};
const inr = n => "₹" + Number(n).toLocaleString("en-IN");
const $ = s => document.querySelector(s);

// ---- Tabs ----
document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {
  const t = btn.dataset.tab;
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".pane").forEach(p => p.classList.toggle("active", p.id === t));
  $("#title").textContent = copy[t][0];
  $("#sub").textContent = copy[t][1];
}));

$("#mode").textContent = USE_MOCK ? "Showing sample data." : "";
document.querySelector('input[name="date"]').min = new Date().toISOString().split("T")[0];
const range = document.querySelector('input[name="budget"]');
range.addEventListener("input", () => $("#budgetVal").textContent = inr(range.value));

// ---- API ----
const endpoints = { flight: "/predict-price", gender: "/predict-gender", hotel: "/recommend-hotel" };

const mocks = {
  flight: () => ({ predicted_price: 4000 + Math.floor(Math.random() * 4000) }),
  gender: d => ({ gender: /a$|i$/i.test(d.name.trim()) ? "Female" : "Male", confidence: 0.82 }),
  hotel:  d => ({ hotels: [
    { name: "Sunrise Residency", price: Math.round(d.budget * 0.85), rating: 4.4 },
    { name: "Blue Lagoon Stay", price: Math.round(d.budget * 0.95), rating: 4.2 },
    { name: "Palm Court Hotel", price: Math.round(d.budget * 0.7), rating: 4.0 }] })
};

async function callApi(kind, payload) {
  if (USE_MOCK) { await new Promise(r => setTimeout(r, 500)); return mocks[kind](payload); }
  const res = await fetch(API_URL + endpoints[kind], {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error("Server returned " + res.status);
  return res.json();
}

// ---- Renderers ----
const render = {
  flight: (r, d) => `<div class="ticket">
      <div class="main"><small>${d.airline} · ${d.date}</small>
        <div class="route">${d.source.slice(0,3).toUpperCase()}<span class="line"></span>✈<span class="line"></span>${d.destination.slice(0,3).toUpperCase()}</div>
        <small>${d.source} to ${d.destination}</small></div>
      <div class="price"><small style="color:#fff">Estimated fare</small><b>${inr(r.predicted_price)}</b></div></div>`,
  gender: r => `<div class="glass big"><small>Predicted gender</small><b>${r.gender}</b>
      ${r.confidence ? `<small>Confidence ${Math.round(r.confidence * 100)}%</small>` : ""}</div>`,
  hotel: r => r.hotels.map(h => `<div class="glass hotel"><div><h3>${h.name}</h3><small>★ ${h.rating}</small></div>
      <span class="p">${inr(h.price)} / night</span></div>`).join("")
};

// ---- Forms ----
document.querySelectorAll("form").forEach(form => form.addEventListener("submit", async e => {
  e.preventDefault();
  const kind = form.dataset.pane, out = $("#out-" + kind), btn = form.querySelector(".cta");
  const data = Object.fromEntries(new FormData(form));
  if (data.age) data.age = Number(data.age);
  if (data.budget) data.budget = Number(data.budget);
  btn.disabled = true; const label = btn.textContent; btn.textContent = "Working…";
  try {
    out.innerHTML = render[kind](await callApi(kind, data), data);
  } catch (err) {
    out.innerHTML = `<p class="error">Could not reach the API at ${API_URL}. Check that the backend is running. (${err.message})</p>`;
  } finally { btn.disabled = false; btn.textContent = label; }
}));
