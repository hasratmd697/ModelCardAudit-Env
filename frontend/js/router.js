const routes = new Set(["overview", "audit", "report", "cards"]);

export function getRouteFromHash() {
  const raw = window.location.hash.replace(/^#\/?/, "") || "overview";
  return routes.has(raw) ? raw : "overview";
}

export function navigate(route) {
  const safeRoute = routes.has(route) ? route : "overview";
  window.location.hash = `#/${safeRoute}`;
}

export function initRouter(onChange) {
  const handleChange = () => onChange(getRouteFromHash());

  window.addEventListener("hashchange", handleChange);

  if (!window.location.hash) {
    navigate("overview");
  } else {
    handleChange();
  }

  return () => window.removeEventListener("hashchange", handleChange);
}
